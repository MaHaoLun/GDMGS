"""Sliding directory with exactly one planned decoder refresh per frame pair.

At an even frame, the previous pair's next-demand batch may be consulted for
current requests. With max_age=1 it is ineligible (source age two), making
the mode exactly equivalent to fixed pair2. With max_age=2 it may save some
current decoding, while the following odd frame consumes the new generation.
"""
from __future__ import annotations

import torch

from .sliding_window_cache import _combine_request_blocks
from .temporal_bundle_cache_v3 import (
    TemporalBundleCache, TemporalGeneration, take_prefix, take_requests,
)


class SlidingPairParityCache:
    def __init__(self, *, identity, anchor_levels, capacity_rows, n_offsets,
                 max_age, audit=False):
        if max_age not in (1, 2):
            raise ValueError("pair-parity max_age must be 1 or 2")
        self.max_age = max_age
        self.codec = TemporalBundleCache(
            identity=identity, anchor_levels=anchor_levels,
            capacity_rows=capacity_rows, n_offsets=n_offsets, audit=audit,
            max_age=1,
        )
        self.identity = identity
        self.levels = self.codec.levels
        self.capacity_rows = capacity_rows
        self.audit = audit
        self.generation = None
        self.last_frame = -1

    def _check_ids(self, ids):
        self.codec._check_ids(ids)
        if ids.numel() > 1 and bool((ids[1:] <= ids[:-1]).any()):
            raise ValueError("pair-parity IDs must be sorted unique")

    def refresh(self, *, frame_id, anchor_ids, next_ids, decode):
        if frame_id % 2 or (self.last_frame >= 0 and frame_id != self.last_frame + 1):
            raise ValueError("refresh must occur at the next even frame")
        self._check_ids(anchor_ids)
        self._check_ids(next_ids)
        ids, levels = anchor_ids, self.levels[anchor_ids]
        blocks = []
        hits = torch.zeros(ids.numel(), dtype=torch.bool, device=ids.device)
        stats = dict(decoder_calls=0, decoded_anchors=0, selected_anchors=ids.numel(),
                     hit_anchors=0, hit_rows=0, hit_age2_anchors=0,
                     hit_age2_rows=0, source_age_max=0, miss_anchors=0,
                     prefetch_anchors=0, evicted_anchors=0)
        old = self.generation
        if old is not None and 1 <= frame_id - old.source_frame <= self.max_age:
            keys = old.batch.anchor_indices
            if keys.numel() and ids.numel():
                positions = torch.searchsorted(keys, ids)
                safe = positions.clamp(max=keys.numel() - 1)
                hits = (positions < keys.numel()) & (keys[safe] == ids)
                if bool(hits.any()):
                    request_positions = torch.nonzero(hits, as_tuple=False).flatten()
                    batch = take_requests(old.batch, positions[request_positions],
                                          ids[request_positions], levels[request_positions])
                    blocks.append((request_positions, batch))
                    stats["hit_anchors"] = ids[request_positions].numel()
                    stats["hit_rows"] = batch.xyz.shape[0]
                    stats["hit_age2_anchors"] = stats["hit_anchors"]
                    stats["hit_age2_rows"] = stats["hit_rows"]
                    stats["source_age_max"] = frame_id - old.source_frame

        miss_positions = torch.nonzero(~hits, as_tuple=False).flatten()
        miss_ids = ids[miss_positions]
        stats["miss_anchors"] = miss_ids.numel()
        union = torch.unique(torch.cat((miss_ids, next_ids)), sorted=True)
        extra = union[~torch.isin(union, miss_ids)]
        ordered = torch.cat((miss_ids, extra))
        decoded = self.codec._decode(ordered, decode, stats)
        current_fresh = take_prefix(decoded, miss_ids.numel())
        if miss_ids.numel():
            blocks.append((miss_positions, current_fresh))
        result = _combine_request_blocks(
            blocks, ids, levels,
            lambda: self.codec._decode(ids, decode, stats),
        )

        sorted_ids, sorted_to_decoded = torch.sort(ordered)
        positions = sorted_to_decoded[torch.searchsorted(sorted_ids, next_ids)]
        next_batch = take_requests(decoded, positions, next_ids, self.levels[next_ids])
        counts = next_batch.bundle_metadata.counts
        fits = (counts.cumsum(0) <= self.capacity_rows) | (counts == 0)
        if not bool(fits.all()):
            kept = torch.nonzero(fits, as_tuple=False).flatten()
            next_batch = take_requests(next_batch, kept, next_ids[kept],
                                       self.levels[next_ids[kept]])
        stats["evicted_anchors"] = int((~fits).sum())
        stats["prefetch_anchors"] = union.numel() - miss_ids.numel()
        self.generation = TemporalGeneration(frame_id, next_batch)
        self.last_frame = frame_id
        if self.audit:
            result.validate_contract()
            next_batch.validate_contract()
        stats.update(output_rows=result.xyz.shape[0],
                     resident_rows=self.generation.rows,
                     resident_bytes=self.generation.memory_bytes(),
                     resident_descriptors=next_batch.anchor_indices.numel(),
                     source_frame=frame_id,
                     capacity_rows=self.capacity_rows)
        return result, stats

    def consume(self, *, frame_id, anchor_ids, decode):
        if frame_id % 2 != 1 or frame_id != self.last_frame + 1:
            raise ValueError("consume must follow its even refresh")
        self._check_ids(anchor_ids)
        old = self.generation
        if old is None or old.source_frame != frame_id - 1:
            raise RuntimeError("the exact preceding generation is missing")
        stats = dict(decoder_calls=0, decoded_anchors=0,
                     selected_anchors=anchor_ids.numel(), hit_anchors=0,
                     hit_rows=0, hit_age2_anchors=0, hit_age2_rows=0,
                     source_age_max=1, miss_anchors=0, prefetch_anchors=0,
                     evicted_anchors=0)
        if torch.equal(old.batch.anchor_indices, anchor_ids):
            result = old.batch
            stats["hit_anchors"] = anchor_ids.numel()
            stats["hit_rows"] = old.rows
        else:
            # A capacity miss is always repaired at the current pose and
            # counted as an extra decoder call, failing formal call parity.
            keys = old.batch.anchor_indices
            positions = torch.searchsorted(keys, anchor_ids)
            safe = positions.clamp(max=max(keys.numel() - 1, 0))
            hits = ((positions < keys.numel()) & (keys[safe] == anchor_ids)
                    if keys.numel() else torch.zeros_like(anchor_ids, dtype=torch.bool))
            blocks = []
            if bool(hits.any()):
                hit_positions = torch.nonzero(hits, as_tuple=False).flatten()
                cached = take_requests(old.batch, positions[hit_positions],
                                       anchor_ids[hit_positions], self.levels[anchor_ids[hit_positions]])
                blocks.append((hit_positions, cached))
                stats["hit_anchors"] = hit_positions.numel()
                stats["hit_rows"] = cached.xyz.shape[0]
            miss_positions = torch.nonzero(~hits, as_tuple=False).flatten()
            if miss_positions.numel():
                fresh = self.codec._decode(anchor_ids[miss_positions], decode, stats)
                blocks.append((miss_positions, fresh))
            result = _combine_request_blocks(
                blocks, anchor_ids, self.levels[anchor_ids],
                lambda: self.codec._decode(anchor_ids, decode, stats),
            )
            stats["miss_anchors"] = miss_positions.numel()
        self.last_frame = frame_id
        if self.audit:
            result.validate_contract()
        stats.update(output_rows=result.xyz.shape[0],
                     resident_rows=old.rows,
                     resident_bytes=old.memory_bytes(),
                     resident_descriptors=old.batch.anchor_indices.numel(),
                     source_frame=old.source_frame,
                     capacity_rows=self.capacity_rows)
        return result, stats
