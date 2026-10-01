"""Decoder tensors, optional bundle ownership, and legacy tuple adaptation.

Jagged descriptors and grid attribute handles remain available to existing
visibility and cache consumers without altering the seven-item dense tuple.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional, Tuple

import torch

from utils.fvdb_visibility import JaggedVisibilityDescriptor

try:  # pragma: no cover - optional dependency during documentation builds
    from utils.fvdb_grid_attributes import GridAttributeHandle
except ImportError:  # pragma: no cover
    GridAttributeHandle = object  # type: ignore[misc,assignment]


@dataclass
class BundleMetadata:
    """Ownership of decoded rows, including requests with no surviving offsets.

    Rows are grouped in request order. IDs identify checkpoint anchor rows;
    ``row_offset_slots`` identifies the decoder offset before opacity filtering.
    """

    request_anchor_ids: torch.Tensor
    request_level_ids: torch.Tensor
    row_owner_ids: torch.Tensor
    row_owner_levels: torch.Tensor
    row_offset_slots: torch.Tensor
    counts: torch.Tensor
    offsets: torch.Tensor

    def __post_init__(self) -> None:
        self.validate()

    def validate(self, row_count: Optional[int] = None) -> None:
        names = (
            "request_anchor_ids", "request_level_ids", "row_owner_ids",
            "row_owner_levels", "row_offset_slots", "counts", "offsets",
        )
        device = None
        for name in names:
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.dtype != torch.long:
                raise ValueError(f"BundleMetadata.{name} must be a rank-one int64 tensor")
            if device is None:
                device = value.device
            elif value.device != device:
                raise ValueError("BundleMetadata tensors must share a device")
            if bool((value < 0).any()):
                raise ValueError(f"BundleMetadata.{name} must be nonnegative")
        requests = self.request_anchor_ids.numel()
        rows = self.row_owner_ids.numel()
        if self.request_level_ids.numel() != requests or self.counts.numel() != requests:
            raise ValueError("BundleMetadata request IDs, levels and counts must have equal lengths")
        if self.offsets.numel() != requests + 1:
            raise ValueError("BundleMetadata.offsets must have request_count + 1 entries")
        expected_offsets = torch.cat((self.counts.new_zeros(1), self.counts.cumsum(0)))
        if not torch.equal(self.offsets, expected_offsets):
            raise ValueError("BundleMetadata.offsets must be the exclusive prefix sum of counts")
        if int(self.offsets[-1]) != rows:
            raise ValueError("BundleMetadata counts must sum to the decoded row count")
        if self.row_owner_levels.numel() != rows or self.row_offset_slots.numel() != rows:
            raise ValueError("BundleMetadata row ownership arrays must have equal lengths")
        if row_count is not None and rows != row_count:
            raise ValueError("BundleMetadata row count does not match Gaussian tensors")
        if not torch.equal(self.row_owner_ids, self.request_anchor_ids.repeat_interleave(self.counts)):
            raise ValueError("BundleMetadata row owners must follow request order and counts")
        if not torch.equal(self.row_owner_levels, self.request_level_ids.repeat_interleave(self.counts)):
            raise ValueError("BundleMetadata row owner levels must follow request order and counts")
        # Filtering preserves increasing offset slots within each request.
        request_rows = torch.arange(requests, device=device).repeat_interleave(self.counts)
        if rows > 1:
            same_request = request_rows[1:] == request_rows[:-1]
            if bool((same_request & (self.row_offset_slots[1:] <= self.row_offset_slots[:-1])).any()):
                raise ValueError("BundleMetadata offset slots must increase within each request")

    @classmethod
    def from_selection(
        cls,
        request_anchor_ids: torch.Tensor,
        request_level_ids: torch.Tensor,
        selection_mask: torch.Tensor,
        n_offsets: int,
    ) -> "BundleMetadata":
        """Apply the decoder's exact opacity mask to expanded owner/slot IDs."""
        if not isinstance(n_offsets, int) or isinstance(n_offsets, bool) or n_offsets <= 0:
            raise ValueError("n_offsets must be a positive integer")
        for name, value in (("request_anchor_ids", request_anchor_ids), ("request_level_ids", request_level_ids)):
            if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.dtype != torch.long:
                raise ValueError(f"{name} must be a rank-one int64 tensor")
        if request_level_ids.shape != request_anchor_ids.shape or request_level_ids.device != request_anchor_ids.device:
            raise ValueError("request anchor IDs and levels must share shape and device")
        if not isinstance(selection_mask, torch.Tensor) or selection_mask.dtype != torch.bool or selection_mask.ndim != 1:
            raise ValueError("selection_mask must be a rank-one bool tensor")
        if selection_mask.device != request_anchor_ids.device:
            raise ValueError("selection_mask and request IDs must share a device")
        requests = request_anchor_ids.numel()
        if selection_mask.numel() != requests * n_offsets:
            raise ValueError("selection_mask must contain request_count * n_offsets entries")
        counts = selection_mask.reshape(requests, n_offsets).sum(dim=1, dtype=torch.long)
        slots = torch.arange(n_offsets, device=request_anchor_ids.device).repeat(requests)
        return cls(
            request_anchor_ids=request_anchor_ids,
            request_level_ids=request_level_ids,
            row_owner_ids=request_anchor_ids.repeat_interleave(n_offsets)[selection_mask],
            row_owner_levels=request_level_ids.repeat_interleave(n_offsets)[selection_mask],
            row_offset_slots=slots[selection_mask],
            counts=counts,
            offsets=torch.cat((counts.new_zeros(1), counts.cumsum(0))),
        )


@dataclass
class NeuralGaussianBatch:
    """Container that pairs jagged visibility with decoded Gaussian tensors."""

    descriptor: Optional[JaggedVisibilityDescriptor]
    anchor_indices: torch.Tensor
    xyz: torch.Tensor
    color: torch.Tensor
    opacity: torch.Tensor
    scaling: torch.Tensor
    rotation: torch.Tensor
    selection_mask: torch.Tensor
    sh_degree: Optional[int]
    attribute_handles: Dict[str, GridAttributeHandle] = field(default_factory=dict)
    bundle_metadata: Optional[BundleMetadata] = None

    def __post_init__(self) -> None:
        """Apply mixed precision storage for expensive buffers."""
        if isinstance(self.scaling, torch.Tensor) and self.scaling.dtype != torch.float16:
            self.scaling = self.scaling.to(dtype=torch.float16)
        if isinstance(self.rotation, torch.Tensor) and self.rotation.dtype != torch.float16:
            self.rotation = self.rotation.to(dtype=torch.float16)
        if self.bundle_metadata is not None:
            self.bundle_metadata.validate(row_count=self.xyz.shape[0])

    def _dense_payload(self) -> Tuple[torch.Tensor, ...]:
        scaling = (
            self.scaling.to(dtype=torch.float32) if isinstance(self.scaling, torch.Tensor) else self.scaling
        )
        rotation = (
            self.rotation.to(dtype=torch.float32) if isinstance(self.rotation, torch.Tensor) else self.rotation
        )
        return (
            self.xyz,
            self.color,
            self.opacity,
            scaling,
            rotation,
            self.sh_degree,
            self.selection_mask,
        )

    def as_tuple(self) -> Tuple[torch.Tensor, ...]:
        """Return the legacy SoA tuple expected by existing renderers."""
        return self._dense_payload()

    def materialize(self, *, jagged: bool = False) -> Tuple[torch.Tensor, ...]:
        """Return decoded tensors and optionally jagged metadata."""

        payload = self._dense_payload()
        if not jagged:
            return payload
        return payload + (self._build_jagged_metadata(),)

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Allow tuple-style unpacking for backwards compatibility."""
        yield from self.as_tuple()

    def jagged_handle(self, name: str) -> Optional[GridAttributeHandle]:
        """Expose the underlying GridAttributeHandle, if registered."""
        return self.attribute_handles.get(name)

    def jagged_payload(self) -> Dict[str, Optional[torch.Tensor]]:
        """Return fvdb jagged tensors for registered attributes when available."""
        payload: Dict[str, Optional[torch.Tensor]] = {}
        for name, handle in self.attribute_handles.items():
            jagged = getattr(handle, "jagged", None)
            payload[name] = jagged if jagged is not None else None
        return payload

    def is_empty(self) -> bool:
        """True when no Gaussian survived visibility/opacity filtering."""
        return bool(self.xyz is None or self.xyz.numel() == 0)

    def _build_jagged_metadata(self) -> Dict[str, Optional[torch.Tensor]]:
        """Return jagged-aligned metadata for cache/shading stages."""

        metadata: Dict[str, Optional[torch.Tensor]] = {
            "anchor_indices": None,
            "level_ids": None,
        }
        if self.descriptor is not None:
            try:
                metadata["anchor_indices"] = self.descriptor.anchor_indices().detach().clone()
            except Exception:
                metadata["anchor_indices"] = None
            try:
                metadata["level_ids"] = self.descriptor.level_ids.detach().clone()
            except Exception:
                metadata["level_ids"] = None
        return metadata


def ensure_gaussian_batch(
    pc,
    payload,
    *,
    anchor_reference: Optional[torch.Tensor] = None,
    descriptor: Optional[JaggedVisibilityDescriptor] = None,
) -> NeuralGaussianBatch:
    """Coerce legacy tuples into NeuralGaussianBatch for downstream consumers."""
    if isinstance(payload, NeuralGaussianBatch):
        return payload
    xyz, color, opacity, scaling, rot, sh_degree, selection_mask = payload
    if selection_mask is None:
        selection_mask = torch.ones(xyz.shape[0], dtype=torch.bool, device=xyz.device)
    descriptor = descriptor or getattr(pc, "last_visibility_descriptor", None)
    anchor_device = pc.get_anchor.device if hasattr(pc, "get_anchor") else xyz.device
    if anchor_reference is not None:
        anchor_indices = anchor_reference.to(device=anchor_device, dtype=torch.long)
    elif descriptor is not None:
        try:
            anchor_indices = descriptor.anchor_indices().to(device=anchor_device, dtype=torch.long)
        except Exception:
            anchor_indices = torch.empty(0, dtype=torch.long, device=anchor_device)
    else:
        anchor_indices = torch.empty(0, dtype=torch.long, device=anchor_device)
    return NeuralGaussianBatch(
        descriptor=descriptor,
        anchor_indices=anchor_indices,
        xyz=xyz,
        color=color,
        opacity=opacity,
        scaling=scaling,
        rotation=rot,
        selection_mask=selection_mask.to(dtype=torch.bool, device=selection_mask.device),
        sh_degree=sh_degree,
        attribute_handles={},
    )
