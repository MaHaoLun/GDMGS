"""
Utilities for building and transporting fVDB-backed visibility descriptors.

The runtime cache expects jagged descriptors that mirror the GridBatch layout
so we can avoid re-materialising flat anchor ids on every frame.  This module
centralises the bookkeeping so both the renderer and the precompute pipeline
share the exact same conversion logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Dict, List, Optional
import warnings

import torch

try:
    import fvdb  # type: ignore
except ImportError:  # pragma: no cover - fallback for documentation builds
    fvdb = None


_MASK_LENGTH_WARNED = False
_OOB_INDEX_WARNED = False


def validate_anchor_ids(
    indices: torch.Tensor,
    anchor_count: int,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Validate an explicit selection without dropping, flattening or reordering IDs."""
    if not isinstance(indices, torch.Tensor):
        raise TypeError("Anchor IDs must be a tensor.")
    if indices.ndim != 1:
        raise ValueError("Anchor IDs must be a one-dimensional tensor.")
    if indices.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("Anchor IDs must have an integer dtype (not bool or floating point).")
    if int(anchor_count) < 0:
        raise ValueError("Anchor count must be nonnegative.")
    indices = indices.to(device=device or indices.device, dtype=torch.long).clone()
    if indices.numel():
        if bool((indices < 0).any()) or bool((indices >= anchor_count).any()):
            raise ValueError(f"Anchor IDs must be in [0, {anchor_count}).")
        if torch.unique(indices).numel() != indices.numel():
            raise ValueError("Anchor IDs must be unique.")
    return indices


def _infer_batch_size(grid: "fvdb.GridBatch", fallback: int = 1, level_data: Optional[torch.Tensor] = None) -> int:
    """Best-effort batch size reconstruction for JaggedTensor builders."""
    grid_count = getattr(grid, "grid_count", None)
    if isinstance(grid_count, Integral) and not isinstance(grid_count, bool) and grid_count > 0:
        return int(grid_count)
    attr = getattr(grid, "batch_size", None)
    if isinstance(attr, int) and attr > 0:
        return attr
    if level_data is not None and level_data.numel() > 0:
        return int(level_data.max().item()) + 1
    jagged = getattr(grid, "ijk", None)
    if jagged is not None:
        jidx = getattr(jagged, "jidx", None)
        if isinstance(jidx, torch.Tensor) and jidx.numel() > 0:
            return int(jidx.max().item()) + 1
    return max(int(attr or fallback), 1)


@dataclass
class JaggedVisibilityDescriptor:
    """Container that packages visibility as a jagged slice plus cached indices."""

    grid: "fvdb.GridBatch"
    jagged: "fvdb.JaggedTensor"
    _anchor_indices: Optional[torch.Tensor] = None

    @classmethod
    def from_mask(
        cls,
        grid: "fvdb.GridBatch",
        mask: torch.Tensor,
        *,
        legacy_indices: Optional[torch.Tensor] = None,
        coords_table: Optional[torch.Tensor] = None,
        level_table: Optional[torch.Tensor] = None,
        strict: bool = False,
    ) -> "JaggedVisibilityDescriptor":
        jagged = getattr(grid, "ijk", None)
        if jagged is None:
            raise RuntimeError("GridBatch is missing ijk coordinates; cannot materialize visibility.")
        total = int(jagged.jdata.shape[0])
        if strict:
            expected = int(coords_table.shape[0]) if coords_table is not None else total
            if mask.dtype != torch.bool or mask.ndim != 1 or mask.numel() != expected:
                raise ValueError(f"Visibility mask must be boolean with shape [{expected}].")
            indices = torch.nonzero(mask, as_tuple=False).flatten()
            return cls.from_indices(
                grid, indices, coords_table=coords_table, level_table=level_table,
                legacy_indices=legacy_indices, strict=True,
            )
        if mask.numel() != total:
            if coords_table is None or level_table is None:
                raise ValueError(
                    f"JaggedVisibilityDescriptor: mask has {mask.numel()} entries but grid tracks {total} voxels."
                )
            indices = torch.nonzero(mask.to(device=coords_table.device, dtype=torch.bool), as_tuple=False).flatten()
            return cls.from_indices(
                grid,
                indices,
                coords_table=coords_table,
                level_table=level_table,
                legacy_indices=indices,
            )
        mask = mask.to(device=grid.ijk.jdata.device, dtype=torch.bool)
        jagged_slice = jagged.r_masked_select(mask)  # type: ignore[attr-defined]
        descriptor = cls(grid, jagged_slice)
        if legacy_indices is not None:
            descriptor._anchor_indices = legacy_indices.to(
                dtype=torch.long, device=grid.ijk.jdata.device, non_blocking=True
            )
        return descriptor

    @classmethod
    def from_indices(
        cls,
        grid: "fvdb.GridBatch",
        indices: torch.Tensor,
        *,
        total_voxels: Optional[int] = None,
        coords_table: Optional[torch.Tensor] = None,
        level_table: Optional[torch.Tensor] = None,
        legacy_indices: Optional[torch.Tensor] = None,
        strict: bool = False,
    ) -> "JaggedVisibilityDescriptor":
        jagged = getattr(grid, "ijk", None)
        if jagged is None:
            raise RuntimeError("GridBatch is missing ijk coordinates; cannot materialize visibility.")
        if strict:
            if (coords_table is None) != (level_table is None):
                raise ValueError("Coordinate and level tables must be supplied together.")
            anchor_total = int(coords_table.shape[0]) if coords_table is not None else int(jagged.jdata.shape[0])
            indices = validate_anchor_ids(indices, anchor_total)
            if total_voxels is not None and int(total_voxels) != anchor_total:
                raise ValueError("total_voxels must match the anchor table length.")
            if coords_table is not None:
                if coords_table.ndim != 2 or coords_table.shape[1] != 3:
                    raise ValueError("Coordinate table must have shape [N, 3].")
                if level_table.ndim != 1 or level_table.shape[0] != anchor_total:
                    raise ValueError("Level table must have shape [N].")
            if legacy_indices is not None:
                legacy_indices = validate_anchor_ids(legacy_indices, anchor_total, device=indices.device)
                if not torch.equal(legacy_indices, indices):
                    raise ValueError("Legacy anchor IDs must match the explicit selection in order.")
        global _MASK_LENGTH_WARNED
        global _OOB_INDEX_WARNED
        if coords_table is not None and level_table is not None:
            anchor_total = int(coords_table.shape[0])
            valid_indices = indices.to(device=coords_table.device, dtype=torch.long)
            legacy_tensor = (
                legacy_indices.to(device=coords_table.device, dtype=torch.long)
                if legacy_indices is not None
                else valid_indices
            )
            if valid_indices.numel() > 0:
                max_index = int(valid_indices.max().item())
                if max_index >= anchor_total:
                    selector = valid_indices < anchor_total
                    dropped = int((~selector).sum().item())
                    if dropped == valid_indices.numel():
                        warnings.warn(
                            (
                                "JaggedVisibilityDescriptor: discarded all visibility indices because they "
                                f"exceeded the anchor table length ({anchor_total})."
                            ),
                            RuntimeWarning,
                        )
                        valid_indices = valid_indices[:0]
                    else:
                        if not _OOB_INDEX_WARNED:
                            warnings.warn(
                                (
                                    "JaggedVisibilityDescriptor: visibility indices exceed the anchor table length; "
                                    f"dropping {dropped} / {indices.numel()} entries."
                                ),
                                RuntimeWarning,
                            )
                            _OOB_INDEX_WARNED = True
                        valid_indices = valid_indices[selector]
                        legacy_tensor = legacy_tensor[selector]
            if fvdb is None:
                raise RuntimeError("fvdb is not available; cannot hydrate jagged descriptor from indices.")
            if valid_indices.numel() == 0:
                coords = coords_table[:0].contiguous()
                level_ids = level_table[:0].contiguous()
            else:
                coords = coords_table.index_select(0, valid_indices).contiguous()
                level_ids = level_table.index_select(0, valid_indices).contiguous()
            batch_size = _infer_batch_size(grid, level_data=level_ids)
            jagged_slice = fvdb.JaggedTensor.from_data_and_jidx(coords, level_ids, batch_size=int(batch_size))
            descriptor = cls(grid, jagged_slice)
            if legacy_tensor.numel() > 0:
                descriptor._anchor_indices = legacy_tensor.to(dtype=torch.long, device=coords.device, non_blocking=True)
            return descriptor

        grid_voxels = int(jagged.jdata.shape[0])
        requested_total = int(total_voxels) if total_voxels is not None else grid_voxels

        if requested_total != grid_voxels and not _MASK_LENGTH_WARNED:
            warnings.warn(
                (
                    "JaggedVisibilityDescriptor: total_voxels does not match GridBatch entries "
                    f"({requested_total} vs {grid_voxels}); using the grid size instead."
                ),
                RuntimeWarning,
            )
            _MASK_LENGTH_WARNED = True
        valid_indices = indices
        legacy_tensor = (
            legacy_indices.to(device=jagged.jdata.device, dtype=torch.long)
            if legacy_indices is not None
            else valid_indices.to(device=jagged.jdata.device, dtype=torch.long)
        )
        if valid_indices.numel() > 0:
            max_index = int(valid_indices.max().item())
            if max_index >= grid_voxels:
                selector = valid_indices < grid_voxels
                dropped = int((~selector).sum().item())
                if dropped == valid_indices.numel():
                    warnings.warn(
                        (
                            "JaggedVisibilityDescriptor: discarded all visibility indices because they "
                            f"exceeded the GridBatch length ({grid_voxels})."
                        ),
                        RuntimeWarning,
                    )
                    valid_indices = valid_indices[:0]
                else:
                    if not _OOB_INDEX_WARNED:
                        warnings.warn(
                            (
                                "JaggedVisibilityDescriptor: visibility indices exceed the GridBatch length; "
                                f"dropping {dropped} / {indices.numel()} entries."
                            ),
                            RuntimeWarning,
                        )
                        _OOB_INDEX_WARNED = True
                    valid_indices = valid_indices[selector]
                    legacy_tensor = legacy_tensor[selector]
        if fvdb is None:
            raise RuntimeError("fvdb is not available; cannot hydrate jagged descriptor from indices.")
        if valid_indices.numel() == 0:
            coords = jagged.jdata[:0].clone()
            level_ids = jagged.jidx[:0].clone()
            batch_size = _infer_batch_size(grid, level_data=jagged.jidx)
        else:
            device = jagged.jdata.device
            gather_idx = valid_indices.to(device=device, dtype=torch.long)
            coords = jagged.jdata.index_select(0, gather_idx)
            level_ids = jagged.jidx.index_select(0, gather_idx)
            batch_size = _infer_batch_size(grid, level_data=level_ids)
        jagged_slice = fvdb.JaggedTensor.from_data_and_jidx(
            coords.contiguous(), level_ids.contiguous(), batch_size=int(batch_size)
        )
        descriptor = cls(grid, jagged_slice)
        if legacy_tensor.numel() > 0:
            descriptor._anchor_indices = legacy_tensor.to(dtype=torch.long, device=coords.device, non_blocking=True)
        return descriptor

    @property
    def level_ids(self) -> torch.Tensor:
        """Return per-entry level identifiers as a 1-D tensor."""
        return self.jagged.jidx.to(dtype=torch.long)

    @property
    def ijk(self) -> torch.Tensor:
        """Return integer xyz coordinates for each visible voxel."""
        return self.jagged.jdata

    def numel(self) -> int:
        return int(self.jagged.jdata.shape[0])

    def anchor_indices(self) -> torch.Tensor:
        """Materialise legacy anchor ids once, using GridBatch metadata."""
        if self._anchor_indices is None:
            idx = self.grid.ijk_to_index(self.jagged)
            self._anchor_indices = idx.jdata.to(dtype=torch.long)
        return self._anchor_indices

    def to_serializable_tensors(self) -> Dict[str, torch.Tensor]:
        """Return tensors that can be persisted inside torch.save payloads."""
        ijk = self.ijk.detach().to(dtype=torch.int32, device="cpu", copy=False).contiguous()
        jidx = self.level_ids.detach().to(dtype=torch.int16, device="cpu", copy=False).contiguous()
        payload: Dict[str, torch.Tensor] = {
            "ijk_jdata": ijk.clone(),
            "ijk_jidx": jidx.clone(),
        }
        return payload

    @classmethod
    def from_serialized(
        cls,
        grid: "fvdb.GridBatch",
        *,
        ijk_jdata: torch.Tensor,
        ijk_jidx: torch.Tensor,
        batch_size: int = 1,
        legacy_indices: Optional[torch.Tensor] = None,
    ) -> "JaggedVisibilityDescriptor":
        """Hydrate a descriptor from serialized tensors."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size <= 0:
            raise ValueError("Serialized descriptor batch_size must be a positive integer.")
        integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if not isinstance(ijk_jdata, torch.Tensor) or ijk_jdata.ndim != 2 or ijk_jdata.shape[1] != 3:
            raise ValueError("Serialized ijk_jdata must have shape [N, 3].")
        if ijk_jdata.dtype not in integer_dtypes:
            raise TypeError("Serialized ijk_jdata must have an integer dtype.")
        if not isinstance(ijk_jidx, torch.Tensor) or ijk_jidx.ndim != 1 or ijk_jidx.shape[0] != ijk_jdata.shape[0]:
            raise ValueError("Serialized ijk_jidx must have shape [N] matching ijk_jdata.")
        if ijk_jidx.dtype not in integer_dtypes:
            raise TypeError("Serialized ijk_jidx must have an integer dtype.")
        level_ids = ijk_jidx.to(dtype=torch.long)
        if level_ids.numel() and (bool((level_ids < 0).any()) or bool((level_ids >= batch_size).any())):
            raise ValueError("Serialized ijk_jidx contains a level outside [0, batch_size).")
        if fvdb is None:
            raise RuntimeError("fvdb is not available; cannot hydrate jagged descriptor.")
        jagged = fvdb.JaggedTensor.from_data_and_jidx(
            ijk_jdata.to(device=grid.ijk.jdata.device),
            ijk_jidx.to(device=grid.ijk.jidx.device),
            batch_size=int(batch_size),
        )
        descriptor = cls(grid, jagged)
        if legacy_indices is not None:
            descriptor._anchor_indices = legacy_indices.to(dtype=torch.long, device=grid.ijk.jdata.device, non_blocking=True)
        return descriptor

    def level_histogram(self) -> Dict[int, int]:
        """Return counts per level id."""
        level_ids = self.level_ids
        if level_ids.numel() == 0:
            return {}
        level_cpu = level_ids.to(device="cpu", dtype=torch.long)
        max_level = int(level_cpu.max().item())
        counts = torch.bincount(level_cpu, minlength=max_level + 1)
        return {level: int(counts[level].item()) for level in range(max_level + 1) if counts[level].item() > 0}

    def summarize(self, sample_voxels: int = 16) -> Dict[str, object]:
        """Build a JSON-serialisable summary for logs/metrics."""
        histogram = self.level_histogram()
        total = sum(histogram.values())
        samples: List[Dict[str, object]] = []
        if sample_voxels > 0 and total > 0:
            coords = self.ijk[:sample_voxels].to(device="cpu", dtype=torch.int32)
            levels = self.level_ids[:sample_voxels].to(device="cpu", dtype=torch.long)
            for level, coord in zip(levels, coords):
                samples.append(
                    {
                        "level": int(level.item()),
                        "ijk": [int(coord[0].item()), int(coord[1].item()), int(coord[2].item())],
                    }
                )
        return {
            "total": int(total),
            "levels": {str(k): v for k, v in histogram.items()},
            "samples": samples,
        }
