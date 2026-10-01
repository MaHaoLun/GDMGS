"""Strict ProxyGS decoder-to-raster contract for the Step 3 gsplat backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


def validate_ordered_anchor_ids(
    anchor_ids: torch.Tensor,
    *,
    anchor_count: int,
    device: torch.device,
) -> torch.Tensor:
    """Validate explicit IDs without sorting, deduplicating, or capping them."""
    if not isinstance(anchor_ids, torch.Tensor):
        raise TypeError("anchor_ids must be a torch.Tensor")
    if anchor_ids.dtype != torch.long or anchor_ids.ndim != 1:
        raise ValueError("anchor_ids must be a rank-one int64 tensor")
    if anchor_ids.device != device:
        raise ValueError("anchor_ids must be on the model device")
    if anchor_ids.numel() == 0:
        return anchor_ids.contiguous()
    if bool((anchor_ids < 0).any()) or bool((anchor_ids >= anchor_count).any()):
        raise ValueError("anchor_ids contain an out-of-range value")
    if torch.unique(anchor_ids).numel() != anchor_ids.numel():
        raise ValueError("anchor_ids must not contain duplicates")
    return anchor_ids.contiguous()


@dataclass(frozen=True)
class BundleMetadata:
    """Ownership of every decoded Gaussian row by ordered anchor request."""

    request_anchor_ids: torch.Tensor
    row_owner_ids: torch.Tensor
    row_offset_slots: torch.Tensor
    counts: torch.Tensor
    offsets: torch.Tensor
    request_level_ids: Optional[torch.Tensor] = None
    row_owner_levels: Optional[torch.Tensor] = None

    def validate(self, *, row_count: Optional[int] = None) -> None:
        values = {
            "request_anchor_ids": self.request_anchor_ids,
            "row_owner_ids": self.row_owner_ids,
            "row_offset_slots": self.row_offset_slots,
            "counts": self.counts,
            "offsets": self.offsets,
        }
        if (self.request_level_ids is None) != (self.row_owner_levels is None):
            raise ValueError(
                "BundleMetadata request and row level IDs must either both be present or both be absent"
            )
        if self.request_level_ids is not None:
            values["request_level_ids"] = self.request_level_ids
            values["row_owner_levels"] = self.row_owner_levels
        device = None
        for name, value in values.items():
            if not isinstance(value, torch.Tensor) or value.dtype != torch.long or value.ndim != 1:
                raise ValueError(f"BundleMetadata.{name} must be a rank-one int64 tensor")
            if device is None:
                device = value.device
            elif value.device != device:
                raise ValueError("BundleMetadata tensors must share a device")
            if bool((value < 0).any()):
                raise ValueError(f"BundleMetadata.{name} must be nonnegative")

        request_count = self.request_anchor_ids.numel()
        if self.counts.numel() != request_count:
            raise ValueError("BundleMetadata counts must match request count")
        if self.request_level_ids is not None and self.request_level_ids.numel() != request_count:
            raise ValueError("BundleMetadata request levels must match request count")
        if self.offsets.numel() != request_count + 1:
            raise ValueError("BundleMetadata offsets must have request_count + 1 entries")
        expected_offsets = torch.cat((self.counts.new_zeros(1), self.counts.cumsum(0)))
        if not torch.equal(self.offsets, expected_offsets):
            raise ValueError("BundleMetadata offsets must be the exclusive prefix sum of counts")
        rows = self.row_owner_ids.numel()
        if self.row_offset_slots.numel() != rows:
            raise ValueError("BundleMetadata row ownership arrays must have equal lengths")
        if int(self.offsets[-1]) != rows:
            raise ValueError("BundleMetadata counts must sum to the decoded row count")
        if row_count is not None and rows != row_count:
            raise ValueError("BundleMetadata row count does not match Gaussian tensors")
        if not torch.equal(self.row_owner_ids, self.request_anchor_ids.repeat_interleave(self.counts)):
            raise ValueError("BundleMetadata row owners must preserve request order")
        if self.request_level_ids is not None:
            if self.row_owner_levels.numel() != rows:
                raise ValueError("BundleMetadata row owner levels must match decoded rows")
            if not torch.equal(
                self.row_owner_levels,
                self.request_level_ids.repeat_interleave(self.counts),
            ):
                raise ValueError("BundleMetadata row owner levels must preserve request order")


@dataclass(frozen=True)
class NeuralGaussianBatch:
    """Minimal batch API consumed by the unchanged GDM-GS raster core."""

    anchor_indices: torch.Tensor
    xyz: torch.Tensor
    color: torch.Tensor
    opacity: torch.Tensor
    scaling: torch.Tensor
    rotation: torch.Tensor
    selection_mask: torch.Tensor
    sh_degree: Optional[int]
    bundle_metadata: Optional[BundleMetadata] = None

    def validate_contract(self) -> None:
        if self.anchor_indices.dtype != torch.long or self.anchor_indices.ndim != 1:
            raise ValueError("anchor_indices must be a rank-one int64 tensor")
        if self.selection_mask.dtype != torch.bool or self.selection_mask.ndim != 1:
            raise ValueError("selection_mask must be a rank-one bool tensor")
        if self.bundle_metadata is None:
            raise ValueError("bundle_metadata is required for the Step 3 handoff")
        self.bundle_metadata.validate(row_count=self.xyz.shape[0])
        if not torch.equal(self.bundle_metadata.request_anchor_ids, self.anchor_indices):
            raise ValueError("bundle request IDs do not match anchor_indices")
        for name in ("xyz", "color", "opacity", "scaling", "rotation"):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                raise ValueError(f"{name} must be a floating-point tensor")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} contains non-finite values")

    def materialize(self) -> Tuple[torch.Tensor, ...]:
        return (
            self.xyz,
            self.color,
            self.opacity,
            self.scaling,
            self.rotation,
            self.sh_degree,
            self.selection_mask,
        )

    def tensor_identity(self) -> Dict[str, Dict[str, object]]:
        record: Dict[str, Dict[str, object]] = {}
        for name in ("xyz", "color", "opacity", "scaling", "rotation", "selection_mask"):
            value = getattr(self, name)
            numeric = value if value.is_floating_point() else value.to(torch.float32)
            record[name] = {
                "object_id": id(value),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "contiguous": value.is_contiguous(),
                "finite": bool(torch.isfinite(numeric).all()),
                "min": float(numeric.min()) if numeric.numel() else None,
                "max": float(numeric.max()) if numeric.numel() else None,
            }
        return record


def batch_from_proxygs_decode(
    *,
    anchor_ids: torch.Tensor,
    decoded: Tuple[torch.Tensor, ...],
    n_offsets: int,
    request_level_ids: Optional[torch.Tensor] = None,
) -> NeuralGaussianBatch:
    """Bind one ProxyGS decode result to ordered request/row ownership."""
    if len(decoded) != 6:
        raise ValueError("ProxyGS inference decode must return six tensors")
    xyz, color, opacity, scaling, rotation, selection_mask = decoded
    if selection_mask.dtype != torch.bool or selection_mask.ndim != 1:
        raise ValueError("ProxyGS opacity selection mask must be rank-one bool")
    if request_level_ids is not None:
        if (
            request_level_ids.dtype != torch.long
            or request_level_ids.ndim != 1
            or request_level_ids.shape != anchor_ids.shape
            or request_level_ids.device != anchor_ids.device
        ):
            raise ValueError(
                "request_level_ids must be rank-one int64 and match anchor_ids shape/device"
            )
        if bool((request_level_ids < 0).any()):
            raise ValueError("request_level_ids must be nonnegative")
    expected = anchor_ids.numel() * n_offsets
    if selection_mask.numel() != expected:
        raise ValueError(
            f"selection mask contains {selection_mask.numel()} entries; expected {expected}"
        )
    counts = selection_mask.reshape(anchor_ids.numel(), n_offsets).sum(dim=1, dtype=torch.long)
    slots = torch.arange(n_offsets, device=anchor_ids.device).repeat(anchor_ids.numel())
    metadata = BundleMetadata(
        request_anchor_ids=anchor_ids,
        row_owner_ids=anchor_ids.repeat_interleave(n_offsets)[selection_mask],
        row_offset_slots=slots[selection_mask],
        counts=counts,
        offsets=torch.cat((counts.new_zeros(1), counts.cumsum(0))),
        request_level_ids=request_level_ids,
        row_owner_levels=(
            request_level_ids.repeat_interleave(n_offsets)[selection_mask]
            if request_level_ids is not None
            else None
        ),
    )
    batch = NeuralGaussianBatch(
        anchor_indices=anchor_ids,
        xyz=xyz,
        color=color,
        opacity=opacity,
        scaling=scaling,
        rotation=rotation,
        selection_mask=selection_mask,
        sh_degree=None,
        bundle_metadata=metadata,
    )
    batch.validate_contract()
    return batch
