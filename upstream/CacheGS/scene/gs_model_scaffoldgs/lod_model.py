#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import logging
import warnings
import time
import torch
import math
from numbers import Integral
from pathlib import Path
import fvdb
import numpy as np
from dataclasses import dataclass
from torch import nn
from einops import repeat
from functools import reduce
from typing import Dict, Iterable, Optional, Tuple, Union
from torch_scatter import scatter_max
from gaussian_renderer.neural_gaussians import BundleMetadata, NeuralGaussianBatch
from gaussian_renderer.fvdb_native_renderer import FvdbNativeRenderer
from utils.fvdb_grid_attributes import GridAttributeHandle, GridAttributeSpec, GridAttributeStore
from utils.fvdb_visibility import JaggedVisibilityDescriptor, validate_anchor_ids
from utils.general_utils import get_expon_lr_func, knn
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.graphics_utils import BasicPointCloud
from scene.embedding import Embedding
from scene.basic_model import BasicModel


@dataclass(frozen=True)
class PoseLocalState:
    """LoD decisions owned by one pose, independent of legacy model state."""

    anchor_mask: torch.Tensor
    prog_ratio: Optional[torch.Tensor] = None
    transition_mask: Optional[torch.Tensor] = None

    def __post_init__(self):
        if self.anchor_mask.dtype != torch.bool or self.anchor_mask.ndim != 1:
            raise ValueError("Pose anchor_mask must be a one-dimensional boolean tensor.")
        if (self.prog_ratio is None) != (self.transition_mask is None):
            raise ValueError("Progressive ratios and transition mask must be supplied together.")
        if self.prog_ratio is not None:
            count = self.anchor_mask.numel()
            if self.prog_ratio.shape != (count, 1) or not self.prog_ratio.is_floating_point():
                raise ValueError("Pose prog_ratio must be a floating-point tensor with shape [N, 1].")
            if self.transition_mask.shape != (count,) or self.transition_mask.dtype != torch.bool:
                raise ValueError("Pose transition_mask must be a boolean tensor with shape [N].")
            if self.prog_ratio.device != self.anchor_mask.device or self.transition_mask.device != self.anchor_mask.device:
                raise ValueError("Pose-local tensors must share a device.")


class GaussianLoDModel(BasicModel):

    _ACCUMULATOR_ATTRIBUTE_NAMES = (
        "opacity_accum",
        "anchor_demon",
        "offset_gradient_accum",
        "offset_denom",
    )

    def __init__(self, **model_kwargs):

        for key, value in model_kwargs.items():
            setattr(self, key, value)

        self._grid_attributes: Optional[GridAttributeStore] = None
        self._grid_attribute_specs: Dict[str, GridAttributeSpec] = {}
        self._grid_attribute_sources: Dict[str, str] = {}
        self._init_grid_attribute_store()
        
        self._anchor = torch.empty(0)
        self._level = torch.empty(0)
        self._extra_level = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)

        # fVDB integration cache
        self._fvdb_grid = None
        self._fvdb_dirty = True
        self._fvdb_coords: Optional[torch.Tensor] = None
        self._fvdb_levels: Optional[torch.Tensor] = None
        self._last_visibility_descriptor: Optional[JaggedVisibilityDescriptor] = None
        self.fvdb_renderer = FvdbNativeRenderer()
        self.opacity_accum = torch.empty(0)
        self.anchor_demon = torch.empty(0)
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)
                
        self.optimizer = None
        self.spatial_lr_scale = 0
        self.setup_functions()
    
        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(self.view_dim, self.feat_dim),
                nn.ReLU(True),
                nn.Linear(self.feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()
            
        self.mlp_opacity = nn.Sequential(
            nn.Linear(self.feat_dim+self.view_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, self.n_offsets),
            nn.Tanh()
        ).cuda()
        
        self.mlp_cov = nn.Sequential(
            nn.Linear(self.feat_dim+self.view_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, 7*self.n_offsets),
        ).cuda()
    
        self.mlp_color = nn.Sequential(
            nn.Linear(self.feat_dim+self.view_dim+self.appearance_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()

    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()
        if self.appearance_dim > 0:
            self.embedding_appearance.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        if self.use_feat_bank:                   
            self.mlp_feature_bank.train()
        if self.appearance_dim > 0:
            self.embedding_appearance.train()

    def capture(self):
        param_dict = {}
        param_dict['optimizer'] = self.optimizer.state_dict()
        param_dict['opacity_mlp'] = self.mlp_opacity.state_dict()
        param_dict['cov_mlp'] = self.mlp_cov.state_dict()
        param_dict['color_mlp'] = self.mlp_color.state_dict()
        if self.use_feat_bank:
            param_dict['feature_bank_mlp'] = self.mlp_feature_bank.state_dict()
        if self.appearance_dim > 0:
            param_dict['appearance'] = self.embedding_appearance.state_dict()
        param_dict['fvdb_grid'] = self._serialize_fvdb_grid()
        if self._grid_attributes_enabled():
            param_dict["grid_attributes"] = self._grid_attributes.state_dict()
        return (
            self.voxel_size,
            self.standard_dist,
            self._anchor,
            self._level,
            self._extra_level,
            self._offset,
            self._scaling,
            self._rotation,
            self.opacity_accum, 
            self.anchor_demon,
            self.offset_gradient_accum,
            self.offset_denom,
            param_dict,
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.voxel_size,
        self.standard_dist,
        self._anchor,
        self._level,
        self._extra_level,
        self._offset,
        self._scaling,
        self._rotation,
        self.opacity_accum, 
        self.anchor_demon,
        self.offset_gradient_accum,
        self.offset_denom,
        param_dict, 
        self.spatial_lr_scale) = model_args
        self._sync_grid_attributes(
            [
                "anchor",
                "offset",
                "anchor_feat",
                "scaling",
                "rotation",
                "level",
                "extra_level",
                "anchor_mask",
            ]
        )
        self.training_setup(training_args)
        self.optimizer.load_state_dict(param_dict['optimizer'])
        self.mlp_opacity.load_state_dict(param_dict['opacity_mlp'])
        self.mlp_cov.load_state_dict(param_dict['cov_mlp'])
        self.mlp_color.load_state_dict(param_dict['color_mlp'])
        if self.use_feat_bank:
            self.mlp_feature_bank.load_state_dict(param_dict['feature_bank_mlp'])
        if self.appearance_dim > 0:
            self.embedding_appearance.load_state_dict(param_dict['appearance'])
        fvdb_snapshot = param_dict.get("fvdb_grid")
        if fvdb_snapshot is None:
            raise RuntimeError("Checkpoint is missing fvdb_grid; retrain or re-export with fvdb metadata.")
        self._hydrate_fvdb_from_serialized(fvdb_snapshot)
        grid_attr_state = param_dict.get("grid_attributes")
        if grid_attr_state and self._grid_attributes_enabled():
            self._grid_attributes.load_state_dict(grid_attr_state)

    @property
    def get_anchor(self):
        return self._anchor
    
    @property
    def get_level(self):
        return self._level
    
    @property
    def get_extra_level(self):
        return self._extra_level
        
    @property
    def get_anchor_feat(self):
        return self._anchor_feat

    @property
    def get_offset(self):
        return self._offset
    
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def fvdb_grid(self):
        """Expose the cached fVDB grid, refreshing on demand."""
        self._ensure_fvdb_ready()
        return self._fvdb_grid

    @property
    def _opacity_accum_attr(self):
        return self.opacity_accum

    @property
    def _anchor_demon_attr(self):
        return self.anchor_demon

    @property
    def _offset_gradient_attr(self):
        return self._reshape_offset_tensor(self.offset_gradient_accum)

    @property
    def _offset_denom_attr(self):
        return self._reshape_offset_tensor(self.offset_denom)

    def fvdb_anchor_lookup(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Deprecated hook for anchor lookup; returns None in grid-index mode."""
        return self._fvdb_anchor_lookup()

    def fvdb_anchor_tables(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return per-anchor quantized coordinates and level ids."""
        self._ensure_fvdb_ready()
        coords = self._fvdb_coords
        levels = self._fvdb_levels
        anchor_total = int(self._anchor.shape[0])
        if coords is None or levels is None or coords.shape[0] != anchor_total:
            coords, levels = self._fvdb_coordinate_jagged()
            self._fvdb_coords = coords
            self._fvdb_levels = levels
        return coords, levels

    def set_appearance(self, num_cameras):
        if self.appearance_dim > 0:
            self.embedding_appearance = Embedding(num_cameras, self.appearance_dim).cuda()
        else:
            self.embedding_appearance = None

    @property
    def get_appearance(self):
        return self.embedding_appearance
    
    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity   

    @property
    def get_cov_mlp(self):
        return self.mlp_cov
    
    @property
    def get_color_mlp(self):
        return self.mlp_color
    
    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank

    @property
    def last_visibility_descriptor(self) -> Optional[JaggedVisibilityDescriptor]:
        """Return the descriptor produced by the most recent visibility query."""
        return self._last_visibility_descriptor

    def _mark_fvdb_dirty(self):
        self._fvdb_dirty = True

    def _ensure_fvdb_ready(self):
        if self._fvdb_grid is None or self._fvdb_dirty:
            self._refresh_fvdb_cache()

    def _fvdb_anchor_lookup(self) -> Optional[Dict[str, torch.Tensor]]:
        return None

    def _grid_attributes_enabled(self) -> bool:
        return self._grid_attributes is not None

    def _init_grid_attribute_store(self) -> None:
        self._grid_attributes = GridAttributeStore()
        self._grid_attribute_specs = self._build_grid_attribute_specs()
        self._grid_attribute_sources = {
            "anchor": "_anchor",
            "offset": "_offset",
            "anchor_feat": "_anchor_feat",
            "scaling": "_scaling",
            "rotation": "_rotation",
            "level": "_level",
            "extra_level": "_extra_level",
            "anchor_mask": "_anchor_mask",
            "opacity_accum": "_opacity_accum_attr",
            "anchor_demon": "_anchor_demon_attr",
            "offset_gradient_accum": "_offset_gradient_attr",
            "offset_denom": "_offset_denom_attr",
        }

    def _build_grid_attribute_specs(self) -> Dict[str, GridAttributeSpec]:
        return {
            "anchor": GridAttributeSpec(
                name="anchor",
                shape=(3,),
                dtype=torch.float32,
                requires_grad=True,
                description="Anchor xyz positions.",
            ),
            "offset": GridAttributeSpec(
                name="offset",
                shape=(self.n_offsets, 3),
                dtype=torch.float32,
                requires_grad=True,
                description="Per-anchor offset bank.",
            ),
            "anchor_feat": GridAttributeSpec(
                name="anchor_feat",
                shape=(self.feat_dim,),
                dtype=torch.float32,
                requires_grad=True,
                description="SH feature bank per anchor.",
            ),
            "scaling": GridAttributeSpec(
                name="scaling",
                dtype=torch.float32,
                requires_grad=True,
                description="Anisotropic scale logits.",
            ),
            "rotation": GridAttributeSpec(
                name="rotation",
                shape=(4,),
                dtype=torch.float32,
                requires_grad=True,
                description="Quaternion rotation parameters.",
            ),
            "level": GridAttributeSpec(
                name="level",
                shape=(1,),
                dtype=torch.int16,
                requires_grad=False,
                description="Integer LoD level per anchor.",
            ),
            "extra_level": GridAttributeSpec(
                name="extra_level",
                shape=(1,),
                dtype=torch.float32,
                requires_grad=False,
                description="Progressive LoD bonus per anchor.",
            ),
            "anchor_mask": GridAttributeSpec(
                name="anchor_mask",
                shape=(1,),
                dtype=torch.bool,
                requires_grad=False,
                description="Visibility/progressive mask per anchor.",
            ),
            "opacity_accum": GridAttributeSpec(
                name="opacity_accum",
                shape=(1,),
                dtype=torch.float32,
                requires_grad=False,
                description="EMA opacity accumulator used for pruning heuristics.",
            ),
            "anchor_demon": GridAttributeSpec(
                name="anchor_demon",
                shape=(1,),
                dtype=torch.float32,
                requires_grad=False,
                description="Counts how often each anchor participates in rendering.",
            ),
            "offset_gradient_accum": GridAttributeSpec(
                name="offset_gradient_accum",
                shape=(self.n_offsets, 1),
                dtype=torch.float32,
                requires_grad=False,
                description="Per-offset gradient accumulator used during densification.",
            ),
            "offset_denom": GridAttributeSpec(
                name="offset_denom",
                shape=(self.n_offsets, 1),
                dtype=torch.float32,
                requires_grad=False,
                description="Per-offset normalisation factor for densification statistics.",
            ),
        }

    def _sync_grid_attributes(self, names: Optional[Iterable[str]] = None) -> None:
        if not self._grid_attributes_enabled():
            return
        target_names = names or self._grid_attribute_sources.keys()
        for name in target_names:
            source = self._grid_attribute_sources.get(name)
            if not source or not hasattr(self, source):
                continue
            tensor = getattr(self, source)
            if tensor is None or (isinstance(tensor, torch.Tensor) and tensor.numel() == 0):
                continue
            spec = self._grid_attribute_specs.get(name)
            self._grid_attributes.register_tensor(name, tensor, spec=spec)

    def _bind_attributes_to_grid(self) -> None:
        if not self._grid_attributes_enabled():
            return
        self._grid_attributes.bind_grid(self._fvdb_grid)

    def _reshape_offset_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.numel() == 0:
            return tensor
        total_offsets = tensor.shape[0]
        if total_offsets == 0 or self.n_offsets <= 0:
            return tensor
        if total_offsets % self.n_offsets != 0:
            raise RuntimeError(
                f"Offset tensor of length {total_offsets} cannot be reshaped with n_offsets={self.n_offsets}; "
                "check densify/prune bookkeeping."
            )
        anchor_count = total_offsets // self.n_offsets
        return tensor.view(anchor_count, self.n_offsets, -1)

    def _grid_attribute_handle(self, name: str):
        if not self._grid_attributes_enabled():
            return None
        return self._grid_attributes.handle(name)

    def _grid_attribute_push_update(self, names: Iterable[str], mask: torch.Tensor) -> None:
        if not self._grid_attributes_enabled():
            return
        if mask is None:
            return
        bool_mask = mask.to(device=self._anchor.device, dtype=torch.bool).view(-1)
        if not bool_mask.any():
            return
        indices = torch.nonzero(bool_mask, as_tuple=False).flatten()
        for name in names:
            handle = self._grid_attribute_handle(name)
            if handle is None:
                continue
            source = self._grid_attribute_sources.get(name)
            if not source or not hasattr(self, source):
                continue
            tensor = getattr(self, source)
            if tensor is None or tensor.numel() == 0:
                continue
            if tensor.shape[0] != bool_mask.shape[0]:
                continue
            payload = tensor[bool_mask]
            jagged = getattr(handle, "jagged", None)
            if jagged is None or not hasattr(jagged, "scatter_update"):
                continue
            try:
                jagged.scatter_update(indices.to(dtype=torch.long), payload)
            except Exception:
                continue

    def _append_optimizer_accumulators(self, new_count: int) -> None:
        if new_count <= 0:
            return
        device = self._anchor.device
        if self.anchor_demon.numel() == 0:
            self.anchor_demon = torch.zeros((new_count, 1), device=device, dtype=torch.float32)
        else:
            extra = torch.zeros((new_count, 1), device=device, dtype=self.anchor_demon.dtype)
            self.anchor_demon = torch.cat([self.anchor_demon, extra], dim=0)
        if self.opacity_accum.numel() == 0:
            self.opacity_accum = torch.zeros((new_count, 1), device=device, dtype=torch.float32)
        else:
            extra = torch.zeros((new_count, 1), device=device, dtype=self.opacity_accum.dtype)
            self.opacity_accum = torch.cat([self.opacity_accum, extra], dim=0)
        offset_extra = torch.zeros(
            (new_count * self.n_offsets, 1),
            device=device,
            dtype=self.offset_gradient_accum.dtype if self.offset_gradient_accum.numel() else torch.float32,
        )
        self.offset_gradient_accum = (
            offset_extra if self.offset_gradient_accum.numel() == 0 else torch.cat([self.offset_gradient_accum, offset_extra], dim=0)
        )
        offset_extra = torch.zeros(
            (new_count * self.n_offsets, 1),
            device=device,
            dtype=self.offset_denom.dtype if self.offset_denom.numel() else torch.float32,
        )
        self.offset_denom = (
            offset_extra if self.offset_denom.numel() == 0 else torch.cat([self.offset_denom, offset_extra], dim=0)
        )
        self._sync_grid_attributes(self._ACCUMULATOR_ATTRIBUTE_NAMES)
        new_mask = torch.zeros(self.get_anchor.shape[0], dtype=torch.bool, device=device)
        new_mask[-new_count:] = True
        self._grid_attribute_push_update(self._ACCUMULATOR_ATTRIBUTE_NAMES, new_mask)

    def _zero_anchor_statistics(self, mask: torch.Tensor) -> None:
        if mask is None or mask.numel() == 0:
            return
        bool_mask = mask.to(device=self._anchor.device, dtype=torch.bool).view(-1)
        if not bool_mask.any():
            return
        self.opacity_accum[bool_mask] = 0
        self.anchor_demon[bool_mask] = 0
        self._sync_grid_attributes(("opacity_accum", "anchor_demon"))
        self._grid_attribute_push_update(("opacity_accum", "anchor_demon"), bool_mask)

    def _reset_offset_statistics(self, mask: torch.Tensor) -> None:
        if mask is None or mask.numel() == 0:
            return
        bool_mask = mask.to(device=self._anchor.device, dtype=torch.bool).view(-1)
        if not bool_mask.any():
            return
        self.offset_denom[bool_mask] = 0
        self.offset_gradient_accum[bool_mask] = 0
        anchor_mask = bool_mask.view(-1, self.n_offsets).any(dim=1)
        self._sync_grid_attributes(("offset_gradient_accum", "offset_denom"))
        self._grid_attribute_push_update(("offset_gradient_accum", "offset_denom"), anchor_mask)

    def _prune_optimizer_accumulators(self, prune_mask: torch.Tensor) -> None:
        if prune_mask is None or prune_mask.numel() == 0:
            return
        mask = prune_mask.to(device=self._anchor.device, dtype=torch.bool).view(-1)
        if not mask.any():
            return
        keep = ~mask
        self.opacity_accum = self.opacity_accum[keep]
        self.anchor_demon = self.anchor_demon[keep]
        keep_offsets = keep.unsqueeze(1).repeat(1, self.n_offsets).view(-1)
        self.offset_gradient_accum = self.offset_gradient_accum[keep_offsets]
        self.offset_denom = self.offset_denom[keep_offsets]
        self._sync_grid_attributes(self._ACCUMULATOR_ATTRIBUTE_NAMES)

    def _normalize_visibility_input(
        self,
        visibility: Union[None, torch.Tensor, JaggedVisibilityDescriptor],
        *,
        strict: bool = False,
    ) -> Tuple[torch.Tensor, Optional[JaggedVisibilityDescriptor]]:
        """Resolve the incoming visibility payload into anchor indices."""
        anchor_count = int(self._anchor.shape[0])
        device = self._anchor.device
        descriptor: Optional[JaggedVisibilityDescriptor] = None

        if isinstance(visibility, JaggedVisibilityDescriptor):
            descriptor = visibility
            indices = descriptor.anchor_indices()
        elif isinstance(visibility, torch.Tensor):
            vis = visibility.to(device=device)
            if vis.dtype == torch.bool:
                if strict and vis.ndim != 1:
                    raise ValueError("Explicit visibility masks must be one-dimensional.")
                if vis.ndim > 1:
                    vis = vis.squeeze(-1)
                if vis.numel() != anchor_count:
                    raise ValueError(
                        f"Visibility mask has length {vis.numel()} but model holds {anchor_count} anchors."
                    )
                indices = torch.nonzero(vis, as_tuple=False).flatten()
            else:
                indices = vis if strict else vis.to(device=device, dtype=torch.long).flatten()
        elif visibility is None:
            indices = torch.arange(anchor_count, device=device, dtype=torch.long)
        else:
            raise TypeError(f"Unsupported visibility payload type: {type(visibility)!r}")
        if strict:
            indices = validate_anchor_ids(indices, anchor_count, device=device)
        else:
            indices = indices.to(device=device, dtype=torch.long)
        return indices, descriptor

    @staticmethod
    def _scalar(value: Union[float, torch.Tensor]) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    def _quantize_anchor_positions(
        self,
        anchor: torch.Tensor,
        level: torch.Tensor,
    ) -> torch.Tensor:
        """Convert anchor positions into GridBatch ijk coordinates."""
        if anchor.numel() == 0:
            return torch.empty_like(anchor, dtype=torch.int32)
        if not hasattr(self, "init_pos"):
            raise RuntimeError("GaussianLoDModel missing init_pos; create_from_pcd must run first.")

        device = anchor.device
        level = level.view(-1).to(device=device, dtype=torch.float32)
        base_voxel = self._scalar(self.voxel_size)
        per_anchor_voxel = (base_voxel / (self.fork ** level)).unsqueeze(-1)
        init_pos = self.init_pos.to(device=device, dtype=torch.float32)
        coords = torch.round(((anchor - init_pos) / per_anchor_voxel) - self.padding).to(torch.int32)
        return coords.contiguous()

    def _infer_voxel_size(self, anchor_tensor: torch.Tensor) -> float:
        """Best-effort voxel size reconstruction for legacy checkpoints."""
        if anchor_tensor.numel() == 0:
            return 1.0
        device = anchor_tensor.device
        neighbor_count = min(int(anchor_tensor.shape[0]), 8)
        if neighbor_count > 3:
            try:
                distances = knn(anchor_tensor, neighbor_count)[:, 1:]
                dist2 = (distances ** 2).mean(dim=-1)
                kth_index = max(int(dist2.shape[0] * 0.5), 1)
                median_dist, _ = torch.kthvalue(dist2, kth_index)
                candidate = float(median_dist.detach().to(device="cpu").item())
                if math.isfinite(candidate) and candidate > 0.0:
                    return candidate
            except RuntimeError as err:  # knn can OOM on massive clouds; fall back silently
                warnings.warn(f"GaussianLoDModel: knn fallback for voxel_size failed ({err}); using bbox heuristic.",
                              RuntimeWarning)
        span = torch.max(anchor_tensor) - torch.min(anchor_tensor)
        span_value = float(span.detach().to(device="cpu").item())
        span_value = span_value if math.isfinite(span_value) and span_value > 0 else 1.0
        fork = float(getattr(self, "fork", 2.0)) or 2.0
        base_layer = float(getattr(self, "base_layer", 1.0)) or 1.0
        denom = fork ** max(base_layer, 1.0)
        fallback = span_value / max(denom, 1.0)
        return fallback if fallback > 0 else 1.0

    def _ensure_voxel_size(self, anchor_tensor: torch.Tensor) -> None:
        voxel_attr = getattr(self, "voxel_size", None)
        if voxel_attr is None:
            raise RuntimeError(
                "GaussianLoDModel: voxel_size metadata missing from PLY. "
                "Please re-export the checkpoint with the current fvdb-native training pipeline."
            )
        try:
            scalar = self._scalar(voxel_attr)
        except Exception as exc:
            raise RuntimeError(
                "GaussianLoDModel: voxel_size metadata could not be parsed; retrain or re-export the checkpoint "
                "with a modern fvdb config."
            ) from exc
        if not math.isfinite(scalar) or scalar <= 0:
            raise RuntimeError(
                f"GaussianLoDModel: voxel_size must be positive; got {scalar}. "
                "Re-export the checkpoint via the current fvdb pipeline so valid metadata is persisted."
            )
        self.voxel_size = scalar

    def _fvdb_coordinate_jagged(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return per-anchor ijk coordinates and level ids for GridBatch construction."""
        if self._anchor.numel() == 0:
            raise RuntimeError("Cannot build fVDB coordinates before anchors are initialised.")

        coords = self._quantize_anchor_positions(self._anchor.detach(), self._level.detach())
        jidx = self._level.view(-1).to(dtype=torch.int16, device=coords.device)
        return coords, jidx.contiguous()

    def _fvdb_batch_size(self) -> int:
        """Return the JaggedTensor batch size that covers all active LoD levels."""
        level_cap = int(getattr(self, "levels", 1) or 1)
        if hasattr(self, "_level") and isinstance(self._level, torch.Tensor) and self._level.numel() > 0:
            data_cap = int(self._level.max().item()) + 1
        else:
            data_cap = 1
        return max(level_cap, data_cap, 1)

    def _refresh_fvdb_cache(self):
        if not self._fvdb_dirty and self._fvdb_grid is not None:
            return
        anchor_count = self._anchor.shape[0]
        if anchor_count == 0:
            raise RuntimeError("fVDB integration requires anchors to be initialized before use.")

        device = self._anchor.device
        if device.type != "cuda":
            raise RuntimeError("fVDB integration requires anchors to be initialized before use.")

        ijk_coords, ijk_jidx = self._fvdb_coordinate_jagged()
        ijk_jt = fvdb.JaggedTensor.from_data_and_jidx(ijk_coords, ijk_jidx, batch_size=self._fvdb_batch_size())

        voxel_sizes = torch.tensor(
            [[self.voxel_size, self.voxel_size, self.voxel_size]],
            device=device,
            dtype=torch.float64,
        )
        origins = self.init_pos.view(1, 3).to(device=device, dtype=torch.float64)

        self._fvdb_grid = fvdb.sparse_grid_from_ijk(
            ijk_jt,
            pad_min=[0, 0, 0],
            pad_max=[0, 0, 0],
            voxel_sizes=voxel_sizes,
            origins=origins,
            mutable=True,
        )
        self._fvdb_dirty = False
        self._fvdb_coords = ijk_coords
        self._fvdb_levels = ijk_jidx
        self._bind_attributes_to_grid()

    def _grid_chunked_mutation(
        self,
        coords: torch.Tensor,
        levels: torch.Tensor,
        *,
        op: str,
        chunk_size: int = 65536,
    ) -> None:
        """Apply enable/disable mutations in manageable batches."""
        if coords.numel() == 0 or self._fvdb_grid is None:
            return
        grid = self._fvdb_grid
        total = coords.shape[0]
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            jagged = fvdb.JaggedTensor.from_data_and_jidx(
                coords[start:end].contiguous(),
                levels[start:end].contiguous(),
                batch_size=self._fvdb_batch_size(),
            )
            getattr(grid, op)(jagged)

    def _grid_mutate_from_mask(self, mask: torch.Tensor, op: str) -> None:
        """Utility that encodes anchors selected by mask and applies the requested mutation."""
        if mask is None or mask.numel() == 0 or mask.dtype != torch.bool:
            return
        coords, levels = self._fvdb_coordinate_jagged()
        selected = mask.to(device=coords.device)
        if selected.numel() != coords.shape[0]:
            raise ValueError("Mask length does not match anchor count for fVDB mutation.")
        chosen_coords = coords[selected]
        chosen_levels = levels[selected]
        self._grid_chunked_mutation(chosen_coords, chosen_levels, op=op)

    def _grid_enable_new_voxels(self, anchor: torch.Tensor, level: torch.Tensor) -> None:
        """Enable ijk coordinates corresponding to the supplied anchors."""
        if anchor.numel() == 0 or self._fvdb_grid is None:
            return
        coords = self._quantize_anchor_positions(anchor.detach(), level.detach())
        levels = level.view(-1).to(dtype=torch.int16, device=coords.device)
        self._grid_chunked_mutation(coords, levels, op="enable_ijk")

    def _serialize_fvdb_grid(self) -> Optional[dict]:
        """Return a CPU-friendly snapshot of the current fVDB grid."""
        if self._fvdb_grid is None:
            return None
        ijk = self._fvdb_grid.ijk
        return {
            "ijk_jdata": ijk.jdata.detach().to(device="cpu"),
            "ijk_jidx": ijk.jidx.detach().to(device="cpu"),
            "batch_size": int(getattr(ijk, "batch_size", 1)),
            "voxel_sizes": self._fvdb_grid.voxel_sizes.detach().to(device="cpu"),
            "origins": self._fvdb_grid.origins.detach().to(device="cpu"),
        }

    def _hydrate_fvdb_from_serialized(self, payload: Optional[dict]) -> None:
        """Restore the cached grid from serialized metadata when available."""
        if not payload:
            self._mark_fvdb_dirty()
            return
        device = self._anchor.device if self._anchor.numel() > 0 else torch.device("cuda")
        jagged = fvdb.JaggedTensor.from_data_and_jidx(
            payload["ijk_jdata"].to(device=device),
            payload["ijk_jidx"].to(device=device),
            batch_size=int(payload.get("batch_size", 1)),
        )
        self._fvdb_grid = fvdb.sparse_grid_from_ijk(
            jagged,
            pad_min=[0, 0, 0],
            pad_max=[0, 0, 0],
            voxel_sizes=payload["voxel_sizes"].to(device=device, dtype=torch.float64),
            origins=payload["origins"].to(device=device, dtype=torch.float64),
            mutable=True,
        )
        self._fvdb_dirty = False
        self._bind_attributes_to_grid()
    
    def set_coarse_interval(self, opt):
        self.coarse_intervals = []
        num_level = self.levels - 1 - self.init_level
        if num_level > 0:
            q = 1/opt.coarse_factor
            a1 = opt.coarse_iter*(1-q)/(1-q**num_level)
            temp_interval = 0
            for i in range(num_level):
                interval = a1 * q ** i + temp_interval
                temp_interval = interval
                self.coarse_intervals.append(interval)

    def set_level(self, points, cameras, scales):
        all_dist = torch.tensor([]).cuda()
        self.cam_infos = torch.empty(0, 4).float().cuda()
        for scale in scales:
            for cam in cameras[scale]:
                cam_center = cam.camera_center
                cam_info = torch.tensor([cam_center[0], cam_center[1], cam_center[2], scale]).float().cuda()
                self.cam_infos = torch.cat((self.cam_infos, cam_info.unsqueeze(dim=0)), dim=0)
                dist = torch.sqrt(torch.sum((points - cam_center)**2, dim=1))
                dist_max = torch.quantile(dist, self.dist_ratio)
                dist_min = torch.quantile(dist, 1 - self.dist_ratio)
                new_dist = torch.tensor([dist_min, dist_max]).float().cuda()
                new_dist = new_dist * scale
                all_dist = torch.cat((all_dist, new_dist), dim=0)
        dist_max = torch.quantile(all_dist, self.dist_ratio)
        dist_min = torch.quantile(all_dist, 1 - self.dist_ratio)
        self.standard_dist = dist_max
        if self.levels == -1:
            self.levels = torch.round(torch.log2(dist_max/dist_min)/math.log2(self.fork)).int().item() + 1
        if self.init_level == -1:
            self.init_level = int(self.levels/2)
            
    def octree_sample(self, data):
        torch.cuda.synchronize(); t0 = time.time()
        self.positions = torch.empty(0, 3).float().cuda()
        self._level = torch.empty(0).int().cuda() 
        for cur_level in range(self.levels):
            cur_size = self.voxel_size/(float(self.fork) ** cur_level)
            new_positions = torch.unique(torch.round((data - self.init_pos) / cur_size), dim=0) * cur_size + self.init_pos
            new_positions += self.padding * cur_size
            new_level = torch.ones(new_positions.shape[0], dtype=torch.int, device="cuda") * cur_level
            self.positions = torch.concat((self.positions, new_positions), dim=0)
            self._level = torch.concat((self._level, new_level), dim=0)
        torch.cuda.synchronize(); t1 = time.time()
        time_diff = t1 - t0
        print(f"Building octree time: {int(time_diff // 60)} min {time_diff % 60} sec")

    def create_from_pcd(self, pcd, spatial_lr_scale, logger, *args):
        points = torch.tensor(pcd.points).float().cuda()
        self.set_level(points, *args)
        self.spatial_lr_scale = spatial_lr_scale
        box_min = torch.min(points)*self.extend
        box_max = torch.max(points)*self.extend
        box_d = box_max - box_min
        if self.base_layer < 0:
            default_voxel_size = 0.02
            self.base_layer = torch.round(torch.log2(box_d/default_voxel_size)).int().item()-(self.levels//2)+1
        self.voxel_size = box_d/(float(self.fork) ** self.base_layer)
        self.init_pos = torch.tensor([box_min, box_min, box_min]).float().cuda()
        self.octree_sample(points)

        if self.visible_threshold < 0:
            self.visible_threshold = 0.0
            self.positions, self._level, self.visible_threshold, _ = self.weed_out(self.positions, self._level)
        self.positions, self._level, _, _ = self.weed_out(self.positions, self._level)
        
        logger.info(f'Branches of Tree: {self.fork}')
        logger.info(f'Base Layer of Tree: {self.base_layer}')
        logger.info(f'Visible Threshold: {self.visible_threshold}')
        logger.info(f'Appearance Embedding Dimension: {self.appearance_dim}')
        logger.info(f'LOD Levels: {self.levels}')
        logger.info(f'Initial Levels: {self.init_level}')
        logger.info(f'Initial Voxel Number: {self.positions.shape[0]}')
        logger.info(f'Min Voxel Size: {self.voxel_size/(2.0 ** (self.levels - 1))}')
        logger.info(f'Max Voxel Size: {self.voxel_size}')

        fused_point_cloud = self.positions
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()
        
        dist2 = (knn(fused_point_cloud, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._level = self._level.unsqueeze(dim=1)
        self._extra_level = torch.zeros(self._anchor.shape[0], dtype=torch.float, device="cuda")
        self._anchor_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device="cuda")
        self._sync_grid_attributes(["anchor_mask"])
        self._sync_grid_attributes(
            [
                "anchor",
                "offset",
                "anchor_feat",
                "scaling",
                "rotation",
                "level",
                "extra_level",
                "anchor_mask",
            ]
        )
        self._mark_fvdb_dirty()
        self._refresh_fvdb_cache()

    def weed_out(self, anchor_positions, anchor_levels):
        visible_count = torch.zeros(anchor_positions.shape[0], dtype=torch.int, device="cuda")
        for cam in self.cam_infos:
            cam_center, scale = cam[:3], cam[3]
            dist = torch.sqrt(torch.sum((anchor_positions - cam_center)**2, dim=1)) * scale
            pred_level = torch.log2(self.standard_dist/dist)/math.log2(self.fork)   
            int_level = self.map_to_int_level(pred_level, self.levels - 1)
            visible_count += (anchor_levels <= int_level).int()
        visible_count = visible_count/len(self.cam_infos)
        weed_mask = (visible_count > self.visible_threshold)
        mean_visible = torch.mean(visible_count)
        return anchor_positions[weed_mask], anchor_levels[weed_mask], mean_visible, weed_mask

    def set_anchor_mask(self, cam_center, iteration, resolution_scale):
        dist = torch.sqrt(torch.sum((self.get_anchor - cam_center)**2, dim=1)) * resolution_scale
        pred_level = torch.log2(self.standard_dist/dist)/math.log2(self.fork) + self._extra_level
        
        if self.progressive:
            coarse_index = np.searchsorted(self.coarse_intervals, iteration) + 1 + self.init_level
        else:
            coarse_index = self.levels

        int_level = self.map_to_int_level(pred_level, coarse_index - 1)
        self._anchor_mask = (self._level.squeeze(dim=1) <= int_level)
        self._sync_grid_attributes(["anchor_mask"])

    def compute_pose_local_state(self, viewpoint_camera, iteration) -> PoseLocalState:
        """Compute the original LoD math without changing current-pose model fields."""
        dist = torch.sqrt(torch.sum((self.get_anchor - viewpoint_camera.camera_center)**2, dim=1)) * viewpoint_camera.resolution_scale
        pred_level = torch.log2(self.standard_dist/dist)/math.log2(self.fork) + self._extra_level
        if self.progressive:
            coarse_index = np.searchsorted(self.coarse_intervals, iteration) + 1 + self.init_level
        else:
            coarse_index = self.levels
        cur_level = coarse_index - 1
        prog_ratio = transition_mask = None
        if self.dist2level == "floor":
            int_level = torch.clamp(torch.floor(pred_level).int(), min=0, max=cur_level)
        elif self.dist2level == "round":
            int_level = torch.clamp(torch.round(pred_level).int(), min=0, max=cur_level)
        elif self.dist2level == "ceil":
            int_level = torch.clamp(torch.ceil(pred_level).int(), min=0, max=cur_level)
        elif self.dist2level == "progressive":
            pred_level = torch.clamp(pred_level+1.0, min=0.9999, max=cur_level + 0.9999)
            int_level = torch.floor(pred_level).int()
            prog_ratio = torch.frac(pred_level).unsqueeze(dim=1)
            transition_mask = self._level.squeeze(dim=1) == int_level
        else:
            raise ValueError(f"Unknown dist2level: {self.dist2level}")
        return PoseLocalState(self._level.squeeze(dim=1) <= int_level, prog_ratio, transition_mask)

    def set_anchor_mask_perlevel(self, cam_center, resolution_scale, cur_level):
        dist = torch.sqrt(torch.sum((self.get_anchor - cam_center)**2, dim=1)) * resolution_scale
        pred_level = torch.log2(self.standard_dist/dist)/math.log2(self.fork) + self._extra_level
        int_level = self.map_to_int_level(pred_level, cur_level)
        self._anchor_mask = (self._level.squeeze(dim=1) <= int_level)
        self._sync_grid_attributes(["anchor_mask"])

    def training_setup(self, training_args):

        self._sync_grid_attributes(["anchor", "offset", "anchor_feat", "scaling", "rotation"])
        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        if self._grid_attributes_enabled():
            self._grid_attributes.configure_optimizer(
                "anchor", lr=training_args.position_lr_init * self.spatial_lr_scale
            )
            self._grid_attributes.configure_optimizer(
                "offset", lr=training_args.offset_lr_init * self.spatial_lr_scale
            )
            self._grid_attributes.configure_optimizer("anchor_feat", lr=training_args.feature_lr)
            self._grid_attributes.configure_optimizer("scaling", lr=training_args.scaling_lr)
            self._grid_attributes.configure_optimizer("rotation", lr=training_args.rotation_lr)
            grid_groups = list(self._grid_attributes.optimizer_param_groups())
        else:
            grid_groups = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            ]

        l = grid_groups + [
            {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
            {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
            {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
        ]
        if self.appearance_dim > 0:
            l.append({'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"})
        if self.use_feat_bank:
            l.append({'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        
        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)
        
        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)
        
        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                        lr_final=training_args.mlp_featurebank_lr_final,
                                                        lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                        max_steps=training_args.mlp_featurebank_lr_max_steps)
        if self.appearance_dim > 0:
            self.appearance_scheduler_args = get_expon_lr_func(lr_init=training_args.appearance_lr_init,
                                                        lr_final=training_args.appearance_lr_final,
                                                        lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                        max_steps=training_args.appearance_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.appearance_dim > 0 and param_group["name"] == "embedding_appearance":
                lr = self.appearance_scheduler_args(iteration)
                param_group['lr'] = lr
            
    def construct_list_of_attributes(self):
        l = []
        l.append('x')
        l.append('y')
        l.append('z')
        l.append('level')
        l.append('extra_level')
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path, iteration):
        mkdir_p(os.path.dirname(path))

        if self.progressive:
            coarse_index = np.searchsorted(self.coarse_intervals, iteration) + 1 + self.init_level
        else:
            coarse_index = self.levels

        level_mask = (self._level <= coarse_index-1).squeeze(-1)
        anchor = self._anchor[level_mask].detach().cpu().numpy()
        levels = self._level[level_mask].detach().cpu().numpy()
        extra_levels = self._extra_level.unsqueeze(dim=1)[level_mask].detach().cpu().numpy()
        anchor_feats = self._anchor_feat[level_mask].detach().cpu().numpy()
        offsets = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous()[level_mask].cpu().numpy()
        scales = self._scaling[level_mask].detach().cpu().numpy()
        rots = self._rotation[level_mask].detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, levels, extra_levels, offsets, anchor_feats, scales, rots), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        
        info = [
            'standard_dist {:.6f}'.format(self.standard_dist),
            'levels {:.6f}'.format(self.levels),
            'voxel_size {:.6f}'.format(self._scalar(self.voxel_size)),
        ]
        if hasattr(self, "base_layer"):
            info.append('base_layer {:.6f}'.format(float(self.base_layer)))
        if hasattr(self, "extend"):
            info.append('extend {:.6f}'.format(float(self.extend)))
        if hasattr(self, "init_pos"):
            init_components = self.init_pos.detach().to(device="cpu").view(-1).tolist()
            axis_labels = ("x", "y", "z")
            for axis, value in zip(axis_labels, init_components):
                info.append(f'init_pos_{axis} {value:.6f}')

        plydata = PlyData([el], obj_info=info)
        plydata.write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)
        infos = plydata.obj_info
        for info in infos:
            var_name = info.split(' ')[0]
            self.__dict__[var_name] = float(info.split(' ')[1])
    
        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        
        levels = np.asarray(plydata.elements[0]["level"])[... ,np.newaxis].astype(np.int16)
        extra_levels = np.asarray(plydata.elements[0]["extra_level"])[... ,np.newaxis].astype(np.float32)
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        
        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))
        
        anchor_tensor = torch.tensor(anchor, dtype=torch.float, device="cuda")

        axis_labels = ("x", "y", "z")
        init_components = tuple(self.__dict__.get(f"init_pos_{axis}") for axis in axis_labels)
        missing_axes = [axis for axis, component in zip(axis_labels, init_components) if component is None]
        if missing_axes:
            missing = ", ".join(missing_axes)
            raise RuntimeError(
                f"GaussianLoDModel: init_pos metadata missing from PLY (missing axes: {missing}). "
                "Please re-export the checkpoint with the current fvdb-native exporter so init_pos is embedded."
            )
        init_pos_tensor = torch.tensor(init_components, dtype=torch.float32, device="cuda")
        self.init_pos = init_pos_tensor
        self._ensure_voxel_size(anchor_tensor)

        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))
        self._level = torch.tensor(levels, dtype=torch.int, device="cuda")
        self._extra_level = torch.tensor(extra_levels, dtype=torch.float, device="cuda").squeeze(dim=1)
        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(anchor_tensor.requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))
        self._anchor_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device="cuda")
        self._sync_grid_attributes(
            [
                "anchor",
                "offset",
                "anchor_feat",
                "scaling",
                "rotation",
                "level",
                "extra_level",
                "anchor_mask",
            ]
        )
        self.levels = round(self.levels)
        if hasattr(self, "base_layer"):
            self.base_layer = int(round(self.base_layer))
        if self.init_level == -1:
            self.init_level = int(self.levels/2)
        self._mark_fvdb_dirty()
        self._refresh_fvdb_cache()
        self._loaded_ply_path = str(Path(path).expanduser().resolve())

    def prune_anchor(self,mask):
        valid_points_mask = ~mask

        self._ensure_fvdb_ready()
        self._grid_mutate_from_mask(mask.to(device=self._anchor.device, dtype=torch.bool), op="disable_ijk")

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._level = self._level[valid_points_mask]    
        self._extra_level = self._extra_level[valid_points_mask]
        self._sync_grid_attributes(
            [
                "anchor",
                "offset",
                "anchor_feat",
                "scaling",
                "rotation",
                "level",
                "extra_level",
            ]
        )
        self._mark_fvdb_dirty()
        self._refresh_fvdb_cache()

    def anchor_growing(self, iteration, grads, threshold, update_ratio, extra_ratio, extra_up, offset_mask, overlap):
        init_length = self.get_anchor.shape[0]
        grads[~offset_mask] = 0.0
        anchor_grads = torch.sum(grads.reshape(-1, self.n_offsets), dim=-1) / (torch.sum(offset_mask.reshape(-1, self.n_offsets), dim=-1) + 1e-6)
        for cur_level in range(self.levels):
            update_value = self.fork ** update_ratio
            level_mask = (self.get_level == cur_level).squeeze(dim=1)
            level_ds_mask = (self.get_level == cur_level + 1).squeeze(dim=1)
            if torch.sum(level_mask) == 0:
                continue
            cur_size = self.voxel_size / (float(self.fork) ** cur_level)
            ds_size = cur_size / self.fork
            # update threshold
            cur_threshold = threshold * (update_value ** cur_level)
            ds_threshold = cur_threshold * update_value
            extra_threshold = cur_threshold * extra_ratio
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold) & (grads < ds_threshold)
            candidate_ds_mask = (grads >= ds_threshold)
            candidate_extra_mask = (anchor_grads >= extra_threshold)

            length_inc = self.get_anchor.shape[0] - init_length
            if length_inc > 0 :
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc * self.n_offsets, dtype=torch.bool, device='cuda')], dim=0)
                candidate_ds_mask = torch.cat([candidate_ds_mask, torch.zeros(length_inc * self.n_offsets, dtype=torch.bool, device='cuda')], dim=0)
                candidate_extra_mask = torch.cat([candidate_extra_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)   
            
            repeated_mask = repeat(level_mask, 'n -> (n k)', k=self.n_offsets)
            candidate_mask = torch.logical_and(candidate_mask, repeated_mask)
            candidate_ds_mask = torch.logical_and(candidate_ds_mask, repeated_mask)
            candidate_extra_mask = torch.logical_and(candidate_extra_mask, level_mask)
            if ~self.progressive or iteration > self.coarse_intervals[-1]:
                self._extra_level += extra_up * candidate_extra_mask.float()    

            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:,:3].unsqueeze(dim=1)

            grid_coords = torch.round((self.get_anchor[level_mask]-self.init_pos)/cur_size - self.padding).int()
            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round((selected_xyz-self.init_pos)/cur_size - self.padding).int()
            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)
            if overlap:
                remove_duplicates = torch.ones(selected_grid_coords_unique.shape[0], dtype=torch.bool, device="cuda")
                candidate_anchor = selected_grid_coords_unique[remove_duplicates] * cur_size + self.init_pos + self.padding * cur_size
                new_level = torch.ones(candidate_anchor.shape[0], dtype=torch.int, device='cuda') * cur_level
                candidate_anchor, new_level, _, weed_mask = self.weed_out(candidate_anchor, new_level)
                remove_duplicates_clone = remove_duplicates.clone()
                remove_duplicates[remove_duplicates_clone] = weed_mask
            elif selected_grid_coords_unique.shape[0] > 0 and grid_coords.shape[0] > 0:
                remove_duplicates = self.get_remove_duplicates(grid_coords, selected_grid_coords_unique)
                remove_duplicates = ~remove_duplicates
                candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size + self.init_pos + self.padding * cur_size
                new_level = torch.ones(candidate_anchor.shape[0], dtype=torch.int, device='cuda') * cur_level
                candidate_anchor, new_level, _, weed_mask = self.weed_out(candidate_anchor, new_level)
                remove_duplicates_clone = remove_duplicates.clone()
                remove_duplicates[remove_duplicates_clone] = weed_mask
            else:
                candidate_anchor = torch.zeros([0, 3], dtype=torch.float, device='cuda')
                remove_duplicates = torch.zeros(selected_grid_coords_unique.shape[0], dtype=torch.bool, device='cuda')
                new_level = torch.zeros([0], dtype=torch.int, device='cuda')

            grid_coords_ds = torch.round((self.get_anchor[level_ds_mask]-self.init_pos)/ds_size-self.padding).int()
            selected_xyz_ds = all_xyz.view([-1, 3])[candidate_ds_mask]
            selected_grid_coords_ds = torch.round((selected_xyz_ds-self.init_pos)/ds_size-self.padding).int()
            selected_grid_coords_unique_ds, inverse_indices_ds = torch.unique(selected_grid_coords_ds, return_inverse=True, dim=0)
            if (~self.progressive or iteration > self.coarse_intervals[-1]) and cur_level < self.levels - 1:
                if overlap:
                    remove_duplicates_ds =  torch.ones(selected_grid_coords_unique_ds.shape[0], dtype=torch.bool, device="cuda")
                    candidate_anchor_ds = selected_grid_coords_unique_ds[remove_duplicates_ds]*ds_size+self.init_pos+self.padding*ds_size
                    new_level_ds = torch.ones(candidate_anchor_ds.shape[0], dtype=torch.int, device='cuda') * (cur_level + 1)
                    candidate_anchor_ds, new_level_ds, _, weed_ds_mask = self.weed_out(candidate_anchor_ds, new_level_ds)
                    remove_duplicates_ds_clone = remove_duplicates_ds.clone()
                    remove_duplicates_ds[remove_duplicates_ds_clone] = weed_ds_mask
                elif selected_grid_coords_unique_ds.shape[0] > 0 and grid_coords_ds.shape[0] > 0:
                    remove_duplicates_ds = self.get_remove_duplicates(grid_coords_ds, selected_grid_coords_unique_ds)
                    remove_duplicates_ds = ~remove_duplicates_ds
                    candidate_anchor_ds = selected_grid_coords_unique_ds[remove_duplicates_ds]*ds_size+self.init_pos+self.padding*ds_size
                    new_level_ds = torch.ones(candidate_anchor_ds.shape[0], dtype=torch.int, device='cuda') * (cur_level + 1)
                    candidate_anchor_ds, new_level_ds, _, weed_ds_mask = self.weed_out(candidate_anchor_ds, new_level_ds)
                    remove_duplicates_ds_clone = remove_duplicates_ds.clone()
                    remove_duplicates_ds[remove_duplicates_ds_clone] = weed_ds_mask
                else:
                    candidate_anchor_ds = torch.zeros([0, 3], dtype=torch.float, device='cuda')
                    remove_duplicates_ds = torch.zeros(selected_grid_coords_unique_ds.shape[0], dtype=torch.bool, device='cuda')
                    new_level_ds = torch.zeros([0], dtype=torch.int, device='cuda')
            else:
                candidate_anchor_ds = torch.zeros([0, 3], dtype=torch.float, device='cuda')
                remove_duplicates_ds = torch.zeros(selected_grid_coords_unique_ds.shape[0], dtype=torch.bool, device='cuda')
                new_level_ds = torch.zeros([0], dtype=torch.int, device='cuda')

            if candidate_anchor.shape[0] + candidate_anchor_ds.shape[0] > 0:
                
                new_anchor = torch.cat([candidate_anchor, candidate_anchor_ds], dim=0)
                new_level = torch.cat([new_level, new_level_ds]).unsqueeze(dim=1).float().cuda()
                
                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]
                new_feat_ds = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_ds_mask]
                new_feat_ds = scatter_max(new_feat_ds, inverse_indices_ds.unsqueeze(1).expand(-1, new_feat_ds.size(1)), dim=0)[0][remove_duplicates_ds]
                new_feat = torch.cat([new_feat, new_feat_ds], dim=0)
                
                new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*cur_size # *0.05
                new_scaling_ds = torch.ones_like(candidate_anchor_ds).repeat([1,2]).float().cuda()*ds_size # *0.05
                new_scaling = torch.cat([new_scaling, new_scaling_ds], dim=0)
                new_scaling = torch.log(new_scaling)
                
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], dtype=torch.float, device='cuda')
                new_rotation_ds = torch.zeros([candidate_anchor_ds.shape[0], 4], dtype=torch.float, device='cuda')
                new_rotation = torch.cat([new_rotation, new_rotation_ds], dim=0)
                new_rotation[:,0] = 1.0

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()
                new_offsets_ds = torch.zeros_like(candidate_anchor_ds).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()
                new_offsets = torch.cat([new_offsets, new_offsets_ds], dim=0)

                new_extra_level = torch.zeros(candidate_anchor.shape[0], dtype=torch.float, device='cuda')
                new_extra_level_ds = torch.zeros(candidate_anchor_ds.shape[0], dtype=torch.float, device='cuda')
                new_extra_level = torch.cat([new_extra_level, new_extra_level_ds])
                
                d = {
                    "anchor": new_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                }   

                self._append_optimizer_accumulators(int(new_anchor.shape[0]))
                torch.cuda.empty_cache()
                
                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._level = torch.cat([self._level, new_level], dim=0)
                self._extra_level = torch.cat([self._extra_level, new_extra_level], dim=0)
                self._sync_grid_attributes(
                    [
                        "anchor",
                        "offset",
                        "anchor_feat",
                        "scaling",
                        "rotation",
                        "level",
                        "extra_level",
                    ]
                )
                self._ensure_fvdb_ready()
                self._grid_enable_new_voxels(new_anchor, new_level)
                self._mark_fvdb_dirty()
                self._refresh_fvdb_cache()

    def run_densify(self, iteration, opt):
        # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom # [N*k, 1]
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > opt.update_interval * opt.success_threshold * 0.5).squeeze(dim=1)
        
        self.anchor_growing(iteration, grads_norm, opt.densify_grad_threshold, opt.update_ratio, opt.extra_ratio, opt.extra_up, offset_mask, opt.overlap)

        offset_total = self.offset_denom.shape[0]
        if offset_mask.numel() != offset_total:
            pad = offset_total - offset_mask.numel()
            if pad > 0:
                padding = torch.zeros(pad, dtype=torch.bool, device=offset_mask.device)
                offset_mask = torch.cat([offset_mask.view(-1), padding], dim=0)
            else:
                offset_mask = offset_mask.view(-1)[:offset_total]
        self._reset_offset_statistics(offset_mask)
        
        # prune anchors
        prune_mask = (self.opacity_accum < opt.min_opacity*self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > opt.update_interval * opt.success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask) # [N] 

        self._zero_anchor_statistics(anchors_mask)
        self._prune_optimizer_accumulators(prune_mask)

        if bool(prune_mask.any()):
            self.prune_anchor(prune_mask)

    def save_mlp_checkpoints(self, path):#split or unite
        mkdir_p(os.path.dirname(path))
        self.eval()
        opacity_mlp = torch.jit.trace(self.mlp_opacity, (torch.rand(1, self.feat_dim+self.view_dim).cuda()))
        opacity_mlp.save(os.path.join(path, 'opacity_mlp.pt'))
        cov_mlp = torch.jit.trace(self.mlp_cov, (torch.rand(1, self.feat_dim+self.view_dim).cuda()))
        cov_mlp.save(os.path.join(path, 'cov_mlp.pt'))
        color_mlp = torch.jit.trace(self.mlp_color, (torch.rand(1, self.feat_dim+self.view_dim+self.appearance_dim).cuda()))
        color_mlp.save(os.path.join(path, 'color_mlp.pt'))
        if self.use_feat_bank:
            feature_bank_mlp = torch.jit.trace(self.mlp_feature_bank, (torch.rand(1, self.view_dim).cuda()))
            feature_bank_mlp.save(os.path.join(path, 'feature_bank_mlp.pt'))
        if self.appearance_dim > 0:
            emd = torch.jit.trace(self.embedding_appearance, (torch.zeros((1,), dtype=torch.long).cuda()))
            emd.save(os.path.join(path, 'embedding_appearance.pt'))
        self.train()

    def load_mlp_checkpoints(self, path):
        self.mlp_opacity = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
        self.mlp_cov = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
        self.mlp_color = torch.jit.load(os.path.join(path, 'color_mlp.pt')).cuda()
        if self.use_feat_bank:
            self.mlp_feature_bank = torch.jit.load(os.path.join(path, 'feature_bank_mlp.pt')).cuda()
        if self.appearance_dim > 0:
            self.embedding_appearance = torch.jit.load(os.path.join(path, 'embedding_appearance.pt')).cuda()
    
    def generate_neural_gaussians(
        self,
        viewpoint_camera,
        visible_mask=None,
        ape_code=-1,
        *,
        build_descriptor: bool = True,
        pose_state: Optional[PoseLocalState] = None,
        return_bundle_metadata: bool = False,
    ):
        if self._anchor.numel() == 0:
            raise RuntimeError("fVDB integration requires anchors to be initialized before rendering.")

        pose_local = pose_state is not None or return_bundle_metadata
        if pose_state is not None:
            if pose_state.anchor_mask.shape != (self._anchor.shape[0],) or pose_state.anchor_mask.device != self._anchor.device:
                raise ValueError("Pose-local state must match the model anchor count and device.")
        if not pose_local or build_descriptor:
            self._ensure_fvdb_ready()

        indices, descriptor = self._normalize_visibility_input(visible_mask, strict=pose_local)
        if build_descriptor and descriptor is None:
            coords_table, level_table = self.fvdb_anchor_tables()
            descriptor = JaggedVisibilityDescriptor.from_indices(
                self.fvdb_grid,
                indices,
                total_voxels=self._anchor.shape[0],
                coords_table=coords_table,
                level_table=level_table,
                legacy_indices=indices,
                strict=pose_local,
            )
        if build_descriptor and not pose_local:
            self._last_visibility_descriptor = descriptor
        device = self._anchor.device

        sh_degree = getattr(self, "active_sh_degree", None)

        if indices.numel() == 0:
            empty_xyz = torch.empty(0, 3, device=device)
            empty_color = torch.empty(0, 3, device=device)
            empty_opacity = torch.empty(0, 1, device=device)
            empty_scaling = torch.empty(0, 3, device=device)
            empty_rot = torch.empty(0, 4, device=device)
            empty_mask = torch.zeros(0, dtype=torch.bool, device=device)
            empty_batch = NeuralGaussianBatch(
                descriptor=descriptor,
                anchor_indices=indices,
                xyz=empty_xyz,
                color=empty_color,
                opacity=empty_opacity,
                scaling=empty_scaling,
                rotation=empty_rot,
                selection_mask=empty_mask,
                sh_degree=sh_degree,
                attribute_handles={},
                bundle_metadata=(
                    BundleMetadata.from_selection(indices, self._level[indices].view(-1).long(), empty_mask, self.n_offsets)
                    if return_bundle_metadata else None
                ),
            )
            return empty_batch

        anchor = self._anchor[indices]
        feat = self._anchor_feat[indices]
        grid_offsets = self._offset[indices]
        grid_scaling = self.scaling_activation(self._scaling[indices])

        ob_view = anchor - viewpoint_camera.camera_center
        ob_dist = ob_view.norm(dim=1, keepdim=True)
        ob_view = ob_view / ob_dist

        if self.use_feat_bank:
            bank_weight = self.get_featurebank_mlp(ob_view).unsqueeze(dim=1)
            feat = feat.unsqueeze(dim=-1)
            feat = feat[:, ::4, :1].repeat([1, 4, 1]) * bank_weight[:, :, :1] + \
                feat[:, ::2, :1].repeat([1, 2, 1]) * bank_weight[:, :, 1:2] + \
                feat[:, ::1, :1] * bank_weight[:, :, 2:]
            feat = feat.squeeze(dim=-1)

        cat_local_view = torch.cat([feat, ob_view], dim=1)

        if self.appearance_dim > 0:
            # Legacy code compared ape_code as a scalar but indexed it as a
            # sequence. Accept either representation without changing -1/UID.
            appearance_code = ape_code
            if isinstance(appearance_code, (tuple, list)):
                if len(appearance_code) != 1:
                    raise ValueError("Appearance override must contain exactly one integer.")
                appearance_code = appearance_code[0]
            if isinstance(appearance_code, torch.Tensor):
                if appearance_code.numel() != 1 or appearance_code.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                    raise TypeError("Appearance override must contain exactly one integer.")
                appearance_code = appearance_code.item()
            if isinstance(appearance_code, bool) or not isinstance(appearance_code, Integral):
                raise TypeError("Appearance override must be an integer or a single-integer sequence.")
            if appearance_code < 0:
                camera_indices = torch.ones_like(cat_local_view[:, 0], dtype=torch.long, device=device) * viewpoint_camera.uid
            else:
                camera_indices = torch.ones_like(cat_local_view[:, 0], dtype=torch.long, device=device) * int(appearance_code)
            appearance = self.get_appearance(camera_indices)

        neural_opacity = self.get_opacity_mlp(cat_local_view)

        if self.dist2level == "progressive":
            prog_ratio_attr = pose_state.prog_ratio if pose_state is not None else getattr(self, "_prog_ratio", None)
            transition_mask_attr = pose_state.transition_mask if pose_state is not None else getattr(self, "transition_mask", None)
            if prog_ratio_attr is None or transition_mask_attr is None:
                raise RuntimeError("Progressive LOD metadata must be initialized before rendering.")
            prog = prog_ratio_attr[indices].clone()
            transition_mask = transition_mask_attr[indices].bool()
            prog[~transition_mask] = 1.0
            neural_opacity = neural_opacity * prog

        neural_opacity = neural_opacity.reshape([-1, 1])
        mask = (neural_opacity > 0.0).view(-1)
        opacity = neural_opacity[mask]

        if self.appearance_dim > 0:
            color = self.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = self.get_color_mlp(cat_local_view)
        color = color.reshape([anchor.shape[0] * self.n_offsets, 3])

        scale_rot = self.get_cov_mlp(cat_local_view)
        scale_rot = scale_rot.reshape([anchor.shape[0] * self.n_offsets, 7])

        # Restore baseline tensor layout before masking so that downstream math is identical.
        grid_offsets = grid_offsets.view(anchor.shape[0], self.n_offsets, 3).reshape(-1, 3)

        concatenated = torch.cat([grid_scaling, anchor], dim=-1)
        concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=self.n_offsets)
        concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, grid_offsets], dim=-1)
        masked = concatenated_all[mask]
        scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)

        scaling = scaling_repeat[:, 3:] * torch.sigmoid(scale_rot[:, :3])
        rot = self.rotation_activation(scale_rot[:, 3:7])
        offsets = offsets * scaling_repeat[:, :3]
        xyz = repeat_anchor + offsets

        attribute_handles: Dict[str, GridAttributeHandle] = {}
        for name in ("anchor", "offset", "anchor_feat", "scaling", "rotation", "level", "extra_level"):
            handle = self._grid_attribute_handle(name)
            if handle is not None:
                attribute_handles[name] = handle

        batch = NeuralGaussianBatch(
            descriptor=descriptor,
            anchor_indices=indices,
            xyz=xyz,
            color=color,
            opacity=opacity,
            scaling=scaling,
            rotation=rot,
            selection_mask=mask.to(dtype=torch.bool, device=device),
            sh_degree=sh_degree,
            attribute_handles=attribute_handles,
            bundle_metadata=(
                BundleMetadata.from_selection(indices, self._level[indices].view(-1).long(), mask, self.n_offsets)
                if return_bundle_metadata else None
            ),
        )
        return batch
