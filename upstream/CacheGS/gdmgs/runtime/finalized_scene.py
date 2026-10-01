"""An in-memory binding to exact loaded PLY rows and decoder parameters.

The caller owns a dedicated evaluation model. This binding never freezes or
detaches the shared training model. It detects supported tensor/parameter
updates and requires a new session after model replacement or training.
"""

from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from types import MappingProxyType

import torch

from utils.fvdb_visibility import validate_anchor_ids
from .camera_key import integer_field


_SESSION_NUMBERS = count(1)
_ROW_FIELDS = ("_anchor", "_level", "_extra_level", "_offset", "_anchor_feat", "_scaling", "_rotation")
_MODULE_FIELDS = ("mlp_opacity", "mlp_cov", "mlp_color", "mlp_feature_bank", "embedding_appearance")
_SETTINGS = ("n_offsets", "feat_dim", "view_dim", "appearance_dim", "use_feat_bank", "dist2level",
             "progressive", "levels", "init_level", "fork", "padding", "standard_dist", "voxel_size",
             "coarse_intervals", "active_sh_degree")


def _setting_value(value):
    if isinstance(value, torch.Tensor):
        return (str(value.dtype), tuple(value.shape), tuple(value.detach().cpu().reshape(-1).tolist()))
    if isinstance(value, (list, tuple)):
        return tuple(_setting_value(v) for v in value)
    if hasattr(value, "tolist"):
        return _setting_value(value.tolist())
    return value


def _guard(tensor):
    try:
        version = tensor._version
    except RuntimeError as exc:
        raise ValueError("Finalize a normally loaded model, not tensors created inside inference_mode.") from exc
    return (id(tensor), tensor.data_ptr(), version, tuple(tensor.shape), str(tensor.dtype), str(tensor.device))


@dataclass(frozen=True)
class FinalizedScene:
    checkpoint_path: str
    iteration: int
    token: str
    anchor_count: int
    _rows: object = field(repr=False)
    _guards: object = field(repr=False)
    _modules: object = field(repr=False)
    _settings: object = field(repr=False)
    _grid_id: int = field(repr=False)
    _renderer_id: int = field(repr=False)
    _loaded_ply_path: object = field(repr=False)

    @classmethod
    def from_model(cls, model, checkpoint_path, iteration):
        iteration = integer_field(iteration, "Checkpoint iteration")
        if iteration < 0:
            raise ValueError("Resolve the checkpoint iteration before creating an inference session.")
        source = str(Path(checkpoint_path).expanduser().resolve())
        loaded_ply = getattr(model, "_loaded_ply_path", None)
        expected_ply = str((Path(source) / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply").resolve())
        if loaded_ply is not None and str(Path(loaded_ply).resolve()) != expected_ply:
            raise ValueError("Declared checkpoint/iteration does not match the model's loaded PLY path.")
        model._ensure_fvdb_ready()
        n = int(model.get_anchor.shape[0])
        if n <= 0:
            raise ValueError("Cannot finalize an uninitialized model.")
        rows = {name: getattr(model, name) for name in _ROW_FIELDS}
        for name, value in rows.items():
            if not isinstance(value, torch.Tensor) or value.ndim == 0 or value.shape[0] != n:
                raise ValueError(f"Finalized field {name} is not aligned with the PLY rows.")
            if name == "_scaling":
                # Checkpoints store logarithmic scales: -inf encodes exactly zero.
                raw_valid = torch.isfinite(value) | torch.isneginf(value)
                activated = model.scaling_activation(value) if hasattr(model, "scaling_activation") else torch.exp(value)
                valid = bool(raw_valid.all()) and bool(torch.isfinite(activated).all()) and bool((activated >= 0).all())
            else:
                valid = bool(torch.isfinite(value).all())
            if value.device != model.get_anchor.device or not valid:
                raise ValueError(f"Finalized field {name} has invalid values or device.")
        if tuple(rows["_anchor"].shape) != (n, 3):
            raise ValueError("Anchor positions must be N x 3.")
        levels = rows["_level"]
        if bool((levels < 0).any()) or not torch.equal(levels, levels.round()):
            raise ValueError("Anchor levels must be nonnegative integers.")
        guards = {name: _guard(value) for name, value in rows.items()}
        for name in ("init_pos", "_fvdb_coords", "_fvdb_levels"):
            value = getattr(model, name, None)
            if isinstance(value, torch.Tensor):
                guards[name] = _guard(value)
        modules = {}
        for name in _MODULE_FIELDS:
            module = getattr(model, name, None)
            if module is None:
                continue
            if module.training:
                raise ValueError("Create a dedicated evaluation model before finalizing a scene.")
            for key, value in list(module.named_parameters()) + list(module.named_buffers()):
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(f"Finalized decoder {name}.{key} contains non-finite values.")
            params = {key: _guard(value) for key, value in module.named_parameters()}
            buffers = {key: _guard(value) for key, value in module.named_buffers()}
            modules[name] = (id(module), params, buffers)
        settings = {name: _setting_value(getattr(model, name, None)) for name in _SETTINGS}
        return cls(source, iteration,
                   f"scene-session-{next(_SESSION_NUMBERS)}", n, MappingProxyType(rows),
                   MappingProxyType(guards), MappingProxyType(modules), MappingProxyType(settings),
                   id(model.fvdb_grid), id(getattr(model, "fvdb_renderer", None)), loaded_ply)

    @property
    def anchor_ids(self):
        return torch.arange(self.anchor_count, dtype=torch.int64, device=self._rows["_anchor"].device)

    def assert_current(self, model):
        if id(getattr(model, "fvdb_renderer", None)) != self._renderer_id:
            raise RuntimeError("Finalized rasterizer changed; create a new session.")
        if getattr(model, "_loaded_ply_path", None) != self._loaded_ply_path:
            raise RuntimeError("Loaded PLY provenance changed; create a new session.")
        for name, guard in self._guards.items():
            value = getattr(model, name, None)
            if not isinstance(value, torch.Tensor) or _guard(value) != guard:
                raise RuntimeError(f"Finalized scene changed at {name}; create a new session.")
        if id(getattr(model, "_fvdb_grid", None)) != self._grid_id or getattr(model, "_fvdb_dirty", False):
            raise RuntimeError("Finalized fVDB binding changed; create a new session.")
        for name, (identity, params, buffers) in self._modules.items():
            module = getattr(model, name, None)
            if module is None or id(module) != identity or module.training:
                raise RuntimeError(f"Finalized decoder {name} changed or entered training mode.")
            if {key: _guard(value) for key, value in module.named_parameters()} != params:
                raise RuntimeError(f"Finalized decoder {name} parameters changed.")
            if {key: _guard(value) for key, value in module.named_buffers()} != buffers:
                raise RuntimeError(f"Finalized decoder {name} buffers changed.")
        for name, value in self._settings.items():
            if _setting_value(getattr(model, name, None)) != value:
                raise RuntimeError(f"Finalized setting {name} changed; create a new session.")

    def validate_ids(self, ids):
        return validate_anchor_ids(ids, self.anchor_count, device=self._rows["_anchor"].device)
