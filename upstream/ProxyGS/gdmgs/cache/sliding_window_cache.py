"""Serial, bounded two-generation cache for sliding camera-frame experiments.

The selected anchor IDs are always produced for the target frame. Cached
materialization is immutable and retains its source frame; hits never renew
its age. An age-two hit is allowed only in the separately selected ablation.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from gaussian_renderer.raster_batch import BundleMetadata, NeuralGaussianBatch
from .temporal_bundle_cache_v3 import (
    FIELDS, TemporalBundleCache, TemporalGeneration, take_prefix, take_requests,
)


@dataclass(frozen=True)
class SlidingMode:
    name: str
    max_age: int
    eager_next: bool


MODES = {
    "slide_lazy_age1": SlidingMode("slide_lazy_age1", 1, False),
    "slide_eager_age1": SlidingMode("slide_eager_age1", 1, True),
    "slide_lazy_age2": SlidingMode("slide_lazy_age2", 2, False),
}


def _combine_request_blocks(blocks, request_ids, request_levels, empty_batch):
    """Return complete bundles in the target frame's exact selected-ID order."""
    if not blocks:
        return empty_batch()
    if len(blocks) == 1 and torch.equal(blocks[0][1].anchor_indices, request_ids):
        return blocks[0][1]
    batches = [batch for _, batch in blocks]
    joined_ids = torch.cat([batch.anchor_indices for batch in batches])
    counts = torch.cat([batch.bundle_metadata.counts for batch in batches])
    joined = NeuralGaussianBatch(
        anchor_indices=joined_ids,
        **{name: torch.cat([getattr(batch, name) for batch in batches]) for name in FIELDS},
        selection_mask=torch.cat([batch.selection_mask for batch in batches]),
        sh_degree=batches[0].sh_degree,
        bundle_metadata=BundleMetadata(
            request_anchor_ids=joined_ids,
            request_level_ids=torch.cat([batch.bundle_metadata.request_level_ids for batch in batches]),
            counts=counts,
            offsets=torch.cat((counts.new_zeros(1), counts.cumsum(0))),
            row_owner_ids=torch.cat([batch.bundle_metadata.row_owner_ids for batch in batches]),
            row_owner_levels=torch.cat([batch.bundle_metadata.row_owner_levels for batch in batches]),
            row_offset_slots=torch.cat([batch.bundle_metadata.row_offset_slots for batch in batches]),
        ),
    )
    positions = torch.empty_like(request_ids)
    offset = 0
    for request_positions, batch in blocks:
        positions[request_positions] = torch.arange(
            batch.anchor_indices.numel(), device=request_ids.device) + offset
        offset += batch.anchor_indices.numel()
    if offset != request_ids.numel():
        raise RuntimeError("sliding cache did not cover every current request")
    return take_requests(joined, positions, request_ids, request_levels)


class SlidingWindowCache:
    """Hold at most two recent complete-bundle generations under one row cap."""

    def __init__(self, *, identity, anchor_levels, capacity_rows, n_offsets, mode, audit=False):
        if mode not in MODES:
            raise ValueError("unknown sliding mode")
        self.mode = MODES[mode]
        self.codec = TemporalBundleCache(
            identity=identity, anchor_levels=anchor_levels,
            capacity_rows=capacity_rows, n_offsets=n_offsets, audit=audit,
            max_age=1,
        )
        self.identity = identity
        self.levels = self.codec.levels
        self.capacity_rows = capacity_rows
        self.audit = audit
        self.generations: list[TemporalGeneration] = []  # oldest -> newest
        self.last_frame = -1

    def reset(self):
        self.generations.clear()
        self.last_frame = -1

    def _check_ids(self, ids):
        self.codec._check_ids(ids)
        if ids.numel() > 1 and bool((ids[1:] <= ids[:-1]).any()):
            raise ValueError("sliding requests must be sorted unique IDs")

    def _admit(self, frame, batch):
        """Publish one immutable generation, evicting only complete bundles."""
        evicted_rows = evicted_generations = 0
        if batch is not None and batch.anchor_indices.numel():
            if batch.xyz.shape[0] > self.capacity_rows:
                counts = batch.bundle_metadata.counts
                fits = (counts.cumsum(0) <= self.capacity_rows) | (counts == 0)
                kept = torch.nonzero(fits, as_tuple=False).flatten()
                evicted_rows += int(batch.xyz.shape[0] - counts[kept].sum())
                batch = take_requests(batch, kept, batch.anchor_indices[kept],
                                      batch.bundle_metadata.request_level_ids[kept])
            self.generations.append(TemporalGeneration(frame, batch))
        while len(self.generations) > 2 or sum(g.rows for g in self.generations) > self.capacity_rows:
            old = self.generations.pop(0)
            evicted_rows += old.rows
            evicted_generations += 1
        return evicted_rows, evicted_generations

    def resolve(self, *, frame_id, anchor_ids, decode, next_ids=None):
        if frame_id <= self.last_frame:
            raise ValueError("sliding frame IDs must be strictly increasing")
        self._check_ids(anchor_ids)
        if self.mode.eager_next:
            if next_ids is not None:
                self._check_ids(next_ids)
        elif next_ids is not None:
            raise ValueError("lazy sliding mode does not consume lookahead")

        expired = [g for g in self.generations if frame_id - g.source_frame > self.mode.max_age]
        self.generations = [g for g in self.generations
                            if frame_id - g.source_frame <= self.mode.max_age]
        ids = anchor_ids
        levels = self.levels[ids]
        claimed = torch.zeros(ids.numel(), dtype=torch.bool, device=ids.device)
        blocks = []
        stats = dict(selected_anchors=ids.numel(), decoder_calls=0, decoded_anchors=0,
                     hit_anchors=0, hit_rows=0, hit_age1_anchors=0, hit_age2_anchors=0,
                     hit_age1_rows=0, hit_age2_rows=0, empty_hits=0,
                     source_age_max=0, miss_anchors=0, prefetch_anchors=0,
                     expired_generations=len(expired),
                     expired_rows=sum(g.rows for g in expired))

        for generation in reversed(self.generations):
            keys = generation.batch.anchor_indices
            if not keys.numel() or not ids.numel():
                continue
            positions = torch.searchsorted(keys, ids)
            safe = positions.clamp(max=keys.numel() - 1)
            take = (~claimed) & (positions < keys.numel()) & (keys[safe] == ids)
            if not bool(take.any()):
                continue
            request_positions = torch.nonzero(take, as_tuple=False).flatten()
            selected_ids = ids[request_positions]
            batch = take_requests(generation.batch, positions[request_positions],
                                  selected_ids, levels[request_positions])
            blocks.append((request_positions, batch))
            claimed |= take
            age = frame_id - generation.source_frame
            rows = batch.xyz.shape[0]
            count = selected_ids.numel()
            stats["hit_anchors"] += count
            stats["hit_rows"] += rows
            stats[f"hit_age{age}_anchors"] += count
            stats[f"hit_age{age}_rows"] += rows
            stats["empty_hits"] += int((batch.bundle_metadata.counts == 0).sum())
            stats["source_age_max"] = max(stats["source_age_max"], age)

        miss_positions = torch.nonzero(~claimed, as_tuple=False).flatten()
        miss_ids = ids[miss_positions]
        stats["miss_anchors"] = miss_ids.numel()
        current_fresh = None
        publication = None
        if self.mode.eager_next and next_ids is not None:
            # Current misses form a contiguous decoder prefix. The supplied
            # future demand is produced by the live GPU query, never an oracle.
            union = torch.unique(torch.cat((miss_ids, next_ids)), sorted=True)
            extra = union[~torch.isin(union, miss_ids)]
            ordered = torch.cat((miss_ids, extra))
            decoded = self.codec._decode(ordered, decode, stats)
            current_fresh = take_prefix(decoded, miss_ids.numel())
            positions = torch.searchsorted(ordered, next_ids)
            # `ordered` is current-first, not globally sorted. Build a sorted
            # ID-to-decoder-position map without moving Gaussian rows.
            sorted_ids, sorted_to_decoded = torch.sort(ordered)
            positions = sorted_to_decoded[torch.searchsorted(sorted_ids, next_ids)]
            publication = take_requests(decoded, positions, next_ids, self.levels[next_ids])
            stats["prefetch_anchors"] = union.numel() - miss_ids.numel()
        elif miss_ids.numel():
            current_fresh = self.codec._decode(miss_ids, decode, stats)
            publication = current_fresh if not self.mode.eager_next else None
        if current_fresh is not None and miss_ids.numel():
            blocks.append((miss_positions, current_fresh))
        result = _combine_request_blocks(
            blocks, ids, levels,
            lambda: self.codec._decode(ids, decode, stats),
        )
        if self.audit:
            result.validate_contract()
        evicted_rows, evicted_generations = self._admit(frame_id, publication)
        self.last_frame = frame_id
        stats.update(
            output_rows=result.xyz.shape[0],
            empty_output_requests=int((result.bundle_metadata.counts == 0).sum()),
            resident_rows=sum(g.rows for g in self.generations),
            resident_descriptors=sum(g.batch.anchor_indices.numel() for g in self.generations),
            resident_bytes=sum(g.memory_bytes() for g in self.generations),
            resident_generations=len(self.generations),
            evicted_rows=evicted_rows,
            evicted_generations=evicted_generations,
            capacity_rows=self.capacity_rows,
        )
        return result, stats
