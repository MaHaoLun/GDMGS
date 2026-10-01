"""GPU-resident full per-anchor bundle cache for Step 7A.

The current generation is immutable during resolution.  Every live directory
entry owns one complete, nonempty, variable-length Gaussian bundle keyed by
``(level_id, anchor_id)``.  This module deliberately contains no cross-pose
eligibility, replacement, residency, prefetch, or asynchronous scheduling
policy; those belong to Steps 7B and 7C.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
import time
from typing import Callable, Dict, Optional, Tuple

import torch

from batch import BundleMetadata, NeuralGaussianBatch


_PAYLOAD_FIELDS = ("xyz", "color", "opacity", "scaling", "rotation")


@dataclass(frozen=True)
class CacheIdentity:
    """Lifecycle identity whose change requires a complete cache reset."""

    scene: str
    model: str
    backend: str
    anchor_table: str
    trace: str

    def validate(self) -> None:
        for name in ("scene", "model", "backend", "anchor_table", "trace"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"CacheIdentity.{name} must be a nonempty string")


@dataclass(frozen=True)
class CacheGeneration:
    """A sealed cache directory and its complete row segments."""

    identity: CacheIdentity
    generation_id: int
    capacity_rows: int
    anchor_count: int
    n_offsets: int
    packed_keys: torch.Tensor
    level_ids: torch.Tensor
    anchor_ids: torch.Tensor
    row_offsets: torch.Tensor
    row_lengths: torch.Tensor
    xyz: torch.Tensor
    color: torch.Tensor
    opacity: torch.Tensor
    scaling: torch.Tensor
    rotation: torch.Tensor
    row_offset_slots: torch.Tensor
    source_camera_ids: Tuple[Optional[str], ...]

    @property
    def descriptor_count(self) -> int:
        return int(self.anchor_ids.numel())

    @property
    def occupied_rows(self) -> int:
        return int(self.xyz.shape[0])

    def memory_bytes(self) -> Dict[str, int]:
        values = {
            "directory": (
                self.packed_keys,
                self.level_ids,
                self.anchor_ids,
                self.row_offsets,
                self.row_lengths,
            ),
            "payload": tuple(getattr(self, name) for name in _PAYLOAD_FIELDS),
            "row_metadata": (self.row_offset_slots,),
        }
        result = {
            name: sum(value.numel() * value.element_size() for value in tensors)
            for name, tensors in values.items()
        }
        result["total"] = sum(result.values())
        return result

    def validate(self) -> None:
        self.identity.validate()
        if type(self.generation_id) is not int or self.generation_id < 0:
            raise ValueError("generation_id must be a nonnegative integer")
        if type(self.capacity_rows) is not int or self.capacity_rows < self.n_offsets:
            raise ValueError("capacity_rows must fit at least one complete maximum-size bundle")
        if type(self.anchor_count) is not int or self.anchor_count <= 0:
            raise ValueError("anchor_count must be positive")
        if type(self.n_offsets) is not int or self.n_offsets <= 0:
            raise ValueError("n_offsets must be positive")

        directory = {
            "packed_keys": self.packed_keys,
            "level_ids": self.level_ids,
            "anchor_ids": self.anchor_ids,
            "row_offsets": self.row_offsets,
            "row_lengths": self.row_lengths,
        }
        device = None
        descriptors = None
        for name, value in directory.items():
            if not isinstance(value, torch.Tensor) or value.dtype != torch.long or value.ndim != 1:
                raise ValueError(f"CacheGeneration.{name} must be rank-one int64")
            if device is None:
                device = value.device
            elif value.device != device:
                raise ValueError("cache directory tensors must share a device")
            if descriptors is None:
                descriptors = value.numel()
            elif value.numel() != descriptors:
                raise ValueError("cache directory tensors must have equal lengths")
        assert descriptors is not None
        if len(self.source_camera_ids) != descriptors:
            raise ValueError("source_camera_ids must match descriptor count")
        if descriptors:
            if bool((self.level_ids < 0).any()):
                raise ValueError("cache level IDs must be nonnegative")
            if bool((self.anchor_ids < 0).any()) or bool((self.anchor_ids >= self.anchor_count).any()):
                raise ValueError("cache anchor IDs must be in range")
            if bool((self.row_lengths <= 0).any()) or bool((self.row_lengths > self.n_offsets).any()):
                raise ValueError("resident bundle lengths must be in [1, n_offsets]")
            expected_offsets = torch.cat(
                (self.row_lengths.new_zeros(1), self.row_lengths.cumsum(0))
            )[:-1]
            if not torch.equal(self.row_offsets, expected_offsets):
                raise ValueError("resident bundle row segments must be contiguous and packed")
            if not bool((self.packed_keys[1:] > self.packed_keys[:-1]).all()):
                raise ValueError("cache directory keys must be sorted and unique")
            expected_keys = self.level_ids * self.anchor_count + self.anchor_ids
            if not torch.equal(self.packed_keys, expected_keys):
                raise ValueError("packed cache keys do not encode (level_id, anchor_id)")

        rows = int(self.row_lengths.sum()) if descriptors else 0
        if rows > self.capacity_rows:
            raise ValueError("cache generation exceeds row capacity")
        for name in _PAYLOAD_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                raise ValueError(f"cache payload {name} must be floating point")
            if value.device != device or value.shape[0] != rows:
                raise ValueError(f"cache payload {name} must share directory device and row count")
            if value.dtype != torch.float32:
                raise ValueError(f"Step 7A reference payload {name} must be float32")
            if not value.is_contiguous() or not bool(torch.isfinite(value).all()):
                raise ValueError(f"cache payload {name} must be finite and contiguous")
        if self.row_offset_slots.dtype != torch.long or self.row_offset_slots.ndim != 1:
            raise ValueError("row_offset_slots must be rank-one int64")
        if self.row_offset_slots.device != device or self.row_offset_slots.numel() != rows:
            raise ValueError("row_offset_slots must share payload device and row count")
        if rows and (
            bool((self.row_offset_slots < 0).any())
            or bool((self.row_offset_slots >= self.n_offsets).any())
        ):
            raise ValueError("row_offset_slots must be in [0, n_offsets)")


@dataclass(frozen=True)
class CacheResolution:
    """Sealed current-frame output plus exact hit/miss accounting."""

    batch: NeuralGaussianBatch
    fresh_batch: NeuralGaussianBatch
    hit_mask: torch.Tensor
    miss_mask: torch.Tensor
    generation_id: int
    hit_rows: int
    fresh_rows: int
    timings_ms: Dict[str, float]

    def validate(self) -> None:
        self.batch.validate_contract()
        self.fresh_batch.validate_contract()
        for name, value in (("hit_mask", self.hit_mask), ("miss_mask", self.miss_mask)):
            if value.dtype != torch.bool or value.ndim != 1:
                raise ValueError(f"{name} must be rank-one bool")
        if not torch.equal(self.miss_mask, ~self.hit_mask):
            raise ValueError("hit and miss masks must be exact complements")
        if self.hit_mask.numel() != self.batch.anchor_indices.numel():
            raise ValueError("hit/miss masks must match request count")
        if self.hit_rows + self.fresh_rows != self.batch.xyz.shape[0]:
            raise ValueError("hit/fresh row accounting must match sealed output")
        if any(value < 0.0 for value in self.timings_ms.values()):
            raise ValueError("cache resolution timings must be nonnegative")


DecodeMisses = Callable[[torch.Tensor, torch.Tensor], NeuralGaussianBatch]


def _segment_rows(starts: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Expand packed ``(start, length)`` segments without a Python loop."""
    if starts.dtype != torch.long or lengths.dtype != torch.long:
        raise ValueError("segment starts and lengths must be int64")
    total = int(lengths.sum()) if lengths.numel() else 0
    if total == 0:
        return starts.new_empty(0)
    prefix = lengths.cumsum(0)
    local = torch.arange(total, dtype=torch.long, device=starts.device)
    local -= (prefix - lengths).repeat_interleave(lengths)
    return starts.repeat_interleave(lengths) + local


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class FullBundleCacheCore:
    """Step 7A cache substrate with immutable current-frame generations."""

    def __init__(
        self,
        *,
        identity: CacheIdentity,
        anchor_levels: torch.Tensor,
        capacity_rows: int,
        n_offsets: int,
    ) -> None:
        identity.validate()
        if (
            not isinstance(anchor_levels, torch.Tensor)
            or anchor_levels.ndim != 1
            or anchor_levels.dtype != torch.long
            or anchor_levels.numel() == 0
        ):
            raise ValueError("anchor_levels must be a nonempty rank-one int64 tensor")
        if bool((anchor_levels < 0).any()):
            raise ValueError("anchor_levels must be nonnegative")
        if type(n_offsets) is not int or n_offsets <= 0:
            raise ValueError("n_offsets must be positive")
        if type(capacity_rows) is not int or capacity_rows < n_offsets:
            raise ValueError("capacity_rows must be at least n_offsets")
        self.identity = identity
        self.anchor_levels = anchor_levels.contiguous()
        self.capacity_rows = capacity_rows
        self.n_offsets = n_offsets
        self._state_lock = Lock()
        self._active_resolutions = 0
        self._generation = self._empty_generation(generation_id=0)

    @property
    def generation(self) -> CacheGeneration:
        return self._generation

    @property
    def device(self) -> torch.device:
        return self.anchor_levels.device

    @property
    def anchor_count(self) -> int:
        return int(self.anchor_levels.numel())

    def _empty_generation(self, *, generation_id: int) -> CacheGeneration:
        device = self.anchor_levels.device
        empty_long = torch.empty(0, dtype=torch.long, device=device)
        generation = CacheGeneration(
            identity=self.identity,
            generation_id=generation_id,
            capacity_rows=self.capacity_rows,
            anchor_count=self.anchor_count,
            n_offsets=self.n_offsets,
            packed_keys=empty_long,
            level_ids=empty_long.clone(),
            anchor_ids=empty_long.clone(),
            row_offsets=empty_long.clone(),
            row_lengths=empty_long.clone(),
            xyz=torch.empty(0, 3, dtype=torch.float32, device=device),
            color=torch.empty(0, 3, dtype=torch.float32, device=device),
            opacity=torch.empty(0, 1, dtype=torch.float32, device=device),
            scaling=torch.empty(0, 3, dtype=torch.float32, device=device),
            rotation=torch.empty(0, 4, dtype=torch.float32, device=device),
            row_offset_slots=empty_long.clone(),
            source_camera_ids=(),
        )
        generation.validate()
        return generation

    def _validate_requests(
        self,
        anchor_ids: torch.Tensor,
        level_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for name, value in (("anchor_ids", anchor_ids), ("level_ids", level_ids)):
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != torch.long
                or value.ndim != 1
                or value.device != self.device
            ):
                raise ValueError(f"{name} must be rank-one int64 on the cache device")
        if anchor_ids.shape != level_ids.shape:
            raise ValueError("anchor_ids and level_ids must have equal shape")
        if anchor_ids.numel():
            if bool((anchor_ids < 0).any()) or bool((anchor_ids >= self.anchor_count).any()):
                raise ValueError("anchor_ids contain an out-of-range value")
            if torch.unique(anchor_ids).numel() != anchor_ids.numel():
                raise ValueError("anchor_ids must not contain duplicates")
            expected_levels = self.anchor_levels.index_select(0, anchor_ids)
            if not torch.equal(expected_levels, level_ids):
                raise ValueError("request level IDs do not match the frozen anchor table")
        return anchor_ids.contiguous(), level_ids.contiguous()

    def _packed_keys(self, anchor_ids: torch.Tensor, level_ids: torch.Tensor) -> torch.Tensor:
        return level_ids * self.anchor_count + anchor_ids

    def build_generation(
        self,
        batch: NeuralGaussianBatch,
        *,
        request_level_ids: torch.Tensor,
        source_camera_id: Optional[str],
        generation_id: Optional[int] = None,
    ) -> CacheGeneration:
        """Build, validate, and seal one generation without publishing it."""
        batch.validate_contract()
        anchor_ids, level_ids = self._validate_requests(
            batch.anchor_indices,
            request_level_ids,
        )
        metadata = batch.bundle_metadata
        assert metadata is not None
        if not torch.equal(metadata.request_anchor_ids, anchor_ids):
            raise ValueError("batch metadata request IDs differ from the generation requests")
        if metadata.request_level_ids is not None and not torch.equal(
            metadata.request_level_ids, level_ids
        ):
            raise ValueError("batch metadata request levels differ from the generation requests")
        if bool((metadata.counts > self.n_offsets).any()):
            raise ValueError("decoded bundle exceeds n_offsets")
        for name in _PAYLOAD_FIELDS:
            value = getattr(batch, name)
            if value.dtype != torch.float32 or value.device != self.device:
                raise ValueError(f"Step 7A cache payload {name} must be float32 on cache device")

        nonempty_requests = torch.nonzero(metadata.counts > 0, as_tuple=False).flatten()
        keys = self._packed_keys(anchor_ids, level_ids)
        live_keys = keys.index_select(0, nonempty_requests)
        if live_keys.numel() != torch.unique(live_keys).numel():
            raise ValueError("one generation cannot contain duplicate live cache keys")
        sorted_order = torch.argsort(live_keys, stable=True)
        request_order = nonempty_requests.index_select(0, sorted_order)
        live_keys = live_keys.index_select(0, sorted_order)
        live_anchor_ids = anchor_ids.index_select(0, request_order)
        live_level_ids = level_ids.index_select(0, request_order)
        live_lengths = metadata.counts.index_select(0, request_order)
        old_starts = metadata.offsets[:-1].index_select(0, request_order)
        source_rows = _segment_rows(old_starts, live_lengths)
        live_offsets = torch.cat(
            (live_lengths.new_zeros(1), live_lengths.cumsum(0))
        )[:-1]
        total_rows = int(live_lengths.sum()) if live_lengths.numel() else 0
        if total_rows > self.capacity_rows:
            raise ValueError(
                f"complete bundles require {total_rows} rows; capacity is {self.capacity_rows}"
            )
        if generation_id is None:
            generation_id = self._generation.generation_id + 1
        source_ids = tuple(source_camera_id for _ in range(int(live_keys.numel())))
        generation = CacheGeneration(
            identity=self.identity,
            generation_id=generation_id,
            capacity_rows=self.capacity_rows,
            anchor_count=self.anchor_count,
            n_offsets=self.n_offsets,
            packed_keys=live_keys.contiguous(),
            level_ids=live_level_ids.contiguous(),
            anchor_ids=live_anchor_ids.contiguous(),
            row_offsets=live_offsets.contiguous(),
            row_lengths=live_lengths.contiguous(),
            xyz=batch.xyz.index_select(0, source_rows).contiguous(),
            color=batch.color.index_select(0, source_rows).contiguous(),
            opacity=batch.opacity.index_select(0, source_rows).contiguous(),
            scaling=batch.scaling.index_select(0, source_rows).contiguous(),
            rotation=batch.rotation.index_select(0, source_rows).contiguous(),
            row_offset_slots=metadata.row_offset_slots.index_select(0, source_rows).contiguous(),
            source_camera_ids=source_ids,
        )
        generation.validate()
        return generation

    def publish_generation(self, generation: CacheGeneration) -> None:
        """Atomically publish a fully validated generation between frames."""
        generation.validate()
        if generation.identity != self.identity:
            raise ValueError("cache generation lifecycle identity differs from this cache")
        if (
            generation.capacity_rows != self.capacity_rows
            or generation.anchor_count != self.anchor_count
            or generation.n_offsets != self.n_offsets
            or generation.packed_keys.device != self.device
        ):
            raise ValueError("cache generation configuration differs from this cache")
        with self._state_lock:
            if self._active_resolutions:
                raise RuntimeError("cannot publish a cache generation during current-frame resolution")
            if generation.generation_id <= self._generation.generation_id:
                raise ValueError("cache generation IDs must increase monotonically")
            self._generation = generation

    def reset(self) -> None:
        self.publish_generation(
            self._empty_generation(generation_id=self._generation.generation_id + 1)
        )

    def _lookup(
        self,
        generation: CacheGeneration,
        packed_keys: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        positions = torch.full_like(packed_keys, -1)
        if not packed_keys.numel() or not generation.packed_keys.numel():
            return positions, positions >= 0
        insertion = torch.searchsorted(generation.packed_keys, packed_keys)
        valid = insertion < generation.packed_keys.numel()
        safe = insertion.clamp(max=generation.packed_keys.numel() - 1)
        hit_mask = valid & (generation.packed_keys.index_select(0, safe) == packed_keys)
        positions[hit_mask] = safe[hit_mask]
        return positions, hit_mask

    def _empty_fresh_batch(
        self,
        anchor_ids: torch.Tensor,
        level_ids: torch.Tensor,
    ) -> NeuralGaussianBatch:
        counts = torch.zeros_like(anchor_ids)
        metadata = BundleMetadata(
            request_anchor_ids=anchor_ids,
            row_owner_ids=anchor_ids.new_empty(0),
            row_offset_slots=anchor_ids.new_empty(0),
            counts=counts,
            offsets=torch.cat((counts.new_zeros(1), counts.cumsum(0))),
            request_level_ids=level_ids,
            row_owner_levels=level_ids.new_empty(0),
        )
        return NeuralGaussianBatch(
            anchor_indices=anchor_ids,
            xyz=torch.empty(0, 3, dtype=torch.float32, device=self.device),
            color=torch.empty(0, 3, dtype=torch.float32, device=self.device),
            opacity=torch.empty(0, 1, dtype=torch.float32, device=self.device),
            scaling=torch.empty(0, 3, dtype=torch.float32, device=self.device),
            rotation=torch.empty(0, 4, dtype=torch.float32, device=self.device),
            selection_mask=torch.empty(0, dtype=torch.bool, device=self.device),
            sh_degree=None,
            bundle_metadata=metadata,
        )

    def resolve(
        self,
        *,
        anchor_ids: torch.Tensor,
        level_ids: torch.Tensor,
        decode_misses: DecodeMisses,
        profile: bool = False,
    ) -> CacheResolution:
        """Resolve one frame against a read-only generation and seal its rows."""
        anchor_ids, level_ids = self._validate_requests(anchor_ids, level_ids)
        if not callable(decode_misses):
            raise TypeError("decode_misses must be callable")
        with self._state_lock:
            self._active_resolutions += 1
            generation = self._generation
        try:
            timings: Dict[str, float] = {}
            if profile:
                _synchronize(self.device)
                total_start = time.perf_counter()
                phase_start = total_start
            packed_keys = self._packed_keys(anchor_ids, level_ids)
            if profile:
                _synchronize(self.device)
                timings["key_build"] = (time.perf_counter() - phase_start) * 1000.0
                phase_start = time.perf_counter()
            positions, hit_mask = self._lookup(generation, packed_keys)
            miss_mask = ~hit_mask
            if profile:
                _synchronize(self.device)
                timings["lookup"] = (time.perf_counter() - phase_start) * 1000.0
                phase_start = time.perf_counter()
            miss_ids = anchor_ids[miss_mask]
            miss_levels = level_ids[miss_mask]
            fresh_batch = (
                decode_misses(miss_ids, miss_levels)
                if miss_ids.numel()
                else self._empty_fresh_batch(miss_ids, miss_levels)
            )
            fresh_batch.validate_contract()
            fresh_metadata = fresh_batch.bundle_metadata
            assert fresh_metadata is not None
            if not torch.equal(fresh_batch.anchor_indices, miss_ids):
                raise ValueError("miss decoder changed the ordered miss request IDs")
            if not torch.equal(fresh_metadata.request_anchor_ids, miss_ids):
                raise ValueError("miss decoder metadata changed the ordered miss request IDs")
            if fresh_metadata.request_level_ids is None or not torch.equal(
                fresh_metadata.request_level_ids, miss_levels
            ):
                raise ValueError("miss decoder must preserve request level IDs")
            if bool((fresh_metadata.counts > self.n_offsets).any()):
                raise ValueError("miss decoder produced a bundle larger than n_offsets")
            if profile:
                _synchronize(self.device)
                timings["miss_decode_and_regroup"] = (
                    time.perf_counter() - phase_start
                ) * 1000.0
                phase_start = time.perf_counter()

            counts = torch.zeros_like(anchor_ids)
            hit_positions = positions[hit_mask]
            if hit_positions.numel():
                counts[hit_mask] = generation.row_lengths.index_select(0, hit_positions)
            if miss_ids.numel():
                counts[miss_mask] = fresh_metadata.counts
            offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
            row_request_ids = torch.arange(
                anchor_ids.numel(), dtype=torch.long, device=self.device
            ).repeat_interleave(counts)
            row_hit_mask = hit_mask.index_select(0, row_request_ids)
            total_rows = int(offsets[-1])
            if profile:
                _synchronize(self.device)
                timings["prefix_sum"] = (time.perf_counter() - phase_start) * 1000.0
                phase_start = time.perf_counter()

            cache_rows = _segment_rows(
                generation.row_offsets.index_select(0, hit_positions),
                generation.row_lengths.index_select(0, hit_positions),
            )
            hit_output_rows = torch.nonzero(row_hit_mask, as_tuple=False).flatten()
            miss_output_rows = torch.nonzero(~row_hit_mask, as_tuple=False).flatten()
            if miss_output_rows.numel() != fresh_batch.xyz.shape[0]:
                raise ValueError("fresh row count does not match mixed assembly positions")

            assembled: Dict[str, torch.Tensor] = {}
            for name in _PAYLOAD_FIELDS:
                cached = getattr(generation, name)
                fresh = getattr(fresh_batch, name)
                if fresh.dtype != torch.float32 or fresh.device != self.device:
                    raise ValueError(f"fresh payload {name} must be float32 on cache device")
                output = torch.empty(
                    (total_rows, *cached.shape[1:]),
                    dtype=torch.float32,
                    device=self.device,
                )
                if hit_output_rows.numel():
                    output.index_copy_(0, hit_output_rows, cached.index_select(0, cache_rows))
                if miss_output_rows.numel():
                    output.index_copy_(0, miss_output_rows, fresh)
                assembled[name] = output.contiguous()

            row_offset_slots = torch.empty(total_rows, dtype=torch.long, device=self.device)
            if hit_output_rows.numel():
                row_offset_slots.index_copy_(
                    0,
                    hit_output_rows,
                    generation.row_offset_slots.index_select(0, cache_rows),
                )
            if miss_output_rows.numel():
                row_offset_slots.index_copy_(
                    0,
                    miss_output_rows,
                    fresh_metadata.row_offset_slots,
                )
            row_owner_ids = anchor_ids.repeat_interleave(counts)
            row_owner_levels = level_ids.repeat_interleave(counts)
            metadata = BundleMetadata(
                request_anchor_ids=anchor_ids,
                row_owner_ids=row_owner_ids,
                row_offset_slots=row_offset_slots,
                counts=counts,
                offsets=offsets,
                request_level_ids=level_ids,
                row_owner_levels=row_owner_levels,
            )
            sealed = NeuralGaussianBatch(
                anchor_indices=anchor_ids,
                xyz=assembled["xyz"],
                color=assembled["color"],
                opacity=assembled["opacity"],
                scaling=assembled["scaling"],
                rotation=assembled["rotation"],
                selection_mask=torch.ones(total_rows, dtype=torch.bool, device=self.device),
                sh_degree=fresh_batch.sh_degree,
                bundle_metadata=metadata,
            )
            if profile:
                _synchronize(self.device)
                timings["gather_and_sealed_assembly"] = (
                    time.perf_counter() - phase_start
                ) * 1000.0
                timings["total"] = (time.perf_counter() - total_start) * 1000.0
            resolution = CacheResolution(
                batch=sealed,
                fresh_batch=fresh_batch,
                hit_mask=hit_mask,
                miss_mask=miss_mask,
                generation_id=generation.generation_id,
                hit_rows=int(counts[hit_mask].sum()) if hit_mask.numel() else 0,
                fresh_rows=int(counts[miss_mask].sum()) if miss_mask.numel() else 0,
                timings_ms=timings,
            )
            resolution.validate()
            return resolution
        finally:
            with self._state_lock:
                self._active_resolutions -= 1
