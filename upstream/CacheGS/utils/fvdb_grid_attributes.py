"""
Helpers that mirror the VDBTensor/GridAttribute layering used in XCube.

The AnyGS training stack still manipulates torch tensors directly, but Phase 1
expects every optimizer-facing payload to have a resident fvdb.JaggedTensor
view so future densify/prune work can mutate data in place.  This module keeps
the bookkeeping in one place: each attribute registers its tensor, optional
optimizer metadata, and (once a GridBatch is available) a jagged view that
shares storage with the underlying tensor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Tuple

import torch

try:  # pragma: no cover - fvdb is only available inside the training env
    import fvdb  # type: ignore
except ImportError:  # pragma: no cover
    fvdb = None


def _dtype_to_str(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _str_to_dtype(name: str) -> torch.dtype:
    return getattr(torch, name)


@dataclass
class GridAttributeSpec:
    """Describes a per-anchor attribute stored alongside the GridBatch."""

    name: str
    shape: Optional[Tuple[int, ...]] = None
    dtype: torch.dtype = torch.float32
    requires_grad: bool = True
    description: str = ""


@dataclass
class GridAttributeState:
    """Serializable snapshot used inside checkpoints."""

    spec: GridAttributeSpec
    optimizer: Dict[str, Any] = field(default_factory=dict)
    tensor: Optional[torch.Tensor] = None


class GridAttributeHandle:
    """Tracks the tensor + jagged view for a single attribute."""

    def __init__(self, spec: GridAttributeSpec):
        self.spec = spec
        self.tensor: Optional[torch.Tensor] = None
        self.optimizer: Dict[str, Any] = {}
        self._jagged: Optional["fvdb.JaggedTensor"] = None

    @property
    def jagged(self) -> Optional["fvdb.JaggedTensor"]:
        return self._jagged

    def attach_tensor(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor
        if self.spec.shape is None:
            self.spec.shape = tuple(tensor.shape[1:]) or (1,)

    def bind_grid(self, grid: Optional["fvdb.GridBatch"]) -> None:
        if fvdb is None or grid is None or self.tensor is None:
            self._jagged = None
            return
        if not hasattr(grid, "jagged_like"):
            self._jagged = None
            return
        grid_voxels = getattr(grid, "total_voxels", None)
        if grid_voxels is None:
            ijk = getattr(grid, "ijk", None)
            if ijk is not None and hasattr(ijk, "jdata"):
                grid_voxels = int(ijk.jdata.shape[0])
        if grid_voxels is None:
            self._jagged = None
            return
        if self.tensor.shape[0] != grid_voxels:
            # Mismatched counts; skip binding until the grid is refreshed.
            self._jagged = None
            return
        self._jagged = grid.jagged_like(self.tensor)

    def optimizer_group(self) -> Optional[Dict[str, Any]]:
        if self.tensor is None or not self.spec.requires_grad:
            return None
        group = {"params": [self.tensor], "name": self.spec.name}
        group.update(self.optimizer)
        return group

    def state_dict(self) -> GridAttributeState:
        payload = self.tensor.detach().to(device="cpu").clone() if self.tensor is not None else None
        return GridAttributeState(
            spec=self.spec,
            optimizer=dict(self.optimizer),
            tensor=payload,
        )

    def load_state(self, state: GridAttributeState) -> None:
        self.optimizer.update(state.optimizer)
        if self.tensor is None or state.tensor is None:
            return
        if self.tensor.shape != state.tensor.shape:
            return
        self.tensor.data.copy_(state.tensor.to(device=self.tensor.device, dtype=self.tensor.dtype))


class GridAttributeStore:
    """Lightweight registry that keeps fvdb-aware handles for per-anchor tensors."""

    def __init__(self):
        self._grid: Optional["fvdb.GridBatch"] = None
        self._handles: Dict[str, GridAttributeHandle] = {}

    def bind_grid(self, grid: Optional["fvdb.GridBatch"]) -> None:
        self._grid = grid
        for handle in self._handles.values():
            handle.bind_grid(grid)

    def register_tensor(
        self,
        name: str,
        tensor: Optional[torch.Tensor],
        *,
        spec: Optional[GridAttributeSpec] = None,
    ) -> None:
        if tensor is None:
            return
        handle = self._handles.get(name)
        if handle is None:
            handle = GridAttributeHandle(spec or GridAttributeSpec(name=name))
            self._handles[name] = handle
        elif spec is not None:
            handle.spec = spec
        handle.attach_tensor(tensor)
        handle.bind_grid(self._grid)

    def configure_optimizer(self, name: str, **hparams: Any) -> None:
        handle = self._handles.get(name)
        if handle is None:
            handle = GridAttributeHandle(GridAttributeSpec(name=name))
            self._handles[name] = handle
        handle.optimizer.update(hparams)

    def optimizer_param_groups(self) -> Iterable[Dict[str, Any]]:
        for handle in self._handles.values():
            group = handle.optimizer_group()
            if group is not None:
                yield group

    def handle(self, name: str) -> Optional[GridAttributeHandle]:
        """Return the registered handle, if present."""
        return self._handles.get(name)

    def update_tensor(self, name: str, tensor: torch.Tensor) -> None:
        if name not in self._handles:
            self.register_tensor(name, tensor)
            return
        self._handles[name].attach_tensor(tensor)
        self._handles[name].bind_grid(self._grid)

    def state_dict(self) -> Dict[str, Dict[str, Any]]:
        state: Dict[str, Dict[str, Any]] = {}
        for name, handle in self._handles.items():
            payload = handle.state_dict()
            state[name] = {
                "spec": {
                    "shape": payload.spec.shape,
                    "dtype": _dtype_to_str(payload.spec.dtype),
                    "requires_grad": payload.spec.requires_grad,
                    "description": payload.spec.description,
                },
                "optimizer": payload.optimizer,
                "tensor": payload.tensor,
            }
        return state

    def load_state_dict(self, state: Optional[Dict[str, Dict[str, Any]]]) -> None:
        if not state:
            return
        for name, payload in state.items():
            spec_dict = payload.get("spec", {})
            spec = GridAttributeSpec(
                name=name,
                shape=tuple(spec_dict.get("shape") or ()) or None,
                dtype=_str_to_dtype(spec_dict.get("dtype", "float32")),
                requires_grad=bool(spec_dict.get("requires_grad", True)),
                description=spec_dict.get("description", ""),
            )
            handle = self._handles.get(name)
            if handle is None:
                handle = GridAttributeHandle(spec)
                self._handles[name] = handle
            else:
                handle.spec = spec
            tensor = payload.get("tensor")
            if tensor is not None:
                if handle.tensor is None:
                    handle.attach_tensor(tensor.to(device=tensor.device, dtype=spec.dtype))
                elif handle.tensor.shape == tensor.shape:
                    handle.tensor.data.copy_(tensor.to(device=handle.tensor.device, dtype=handle.tensor.dtype))
            handle.optimizer.update(payload.get("optimizer", {}))
            handle.bind_grid(self._grid)
