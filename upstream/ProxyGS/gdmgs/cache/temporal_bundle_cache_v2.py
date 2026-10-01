"""ADR 0015 ablations: bounded-age complete bundles with explicit next demand.

The synchronous frame transaction owns publication. The old Step 7A/7B
implementation and experiment are intentionally unchanged.
"""
from dataclasses import dataclass
from threading import Lock
from typing import Callable, Optional

import torch

from gaussian_renderer.raster_batch import BundleMetadata, NeuralGaussianBatch
from .full_bundle_cache import CacheIdentity, _segment_rows

FIELDS = ("xyz", "color", "opacity", "scaling", "rotation")


def take_requests(batch, positions, ids, levels):
    """Gather complete request segments, including explicit empty results."""
    meta = batch.bundle_metadata
    counts = meta.counts[positions]
    rows = _segment_rows(meta.offsets[:-1][positions], counts)
    return NeuralGaussianBatch(
        anchor_indices=ids,
        **{name: getattr(batch, name)[rows] for name in FIELDS},
        selection_mask=torch.ones(rows.numel(), dtype=torch.bool, device=ids.device),
        sh_degree=batch.sh_degree,
        bundle_metadata=BundleMetadata(
            request_anchor_ids=ids, request_level_ids=levels,
            counts=counts, offsets=torch.cat((counts.new_zeros(1), counts.cumsum(0))),
            row_owner_ids=ids.repeat_interleave(counts),
            row_owner_levels=levels.repeat_interleave(counts),
            row_offset_slots=meta.row_offset_slots[rows],
        ),
    )


@dataclass(frozen=True)
class TemporalGeneration:
    source_frame: int
    batch: NeuralGaussianBatch

    @property
    def rows(self):
        return self.batch.xyz.shape[0]

    def memory_bytes(self):
        meta = self.batch.bundle_metadata
        tensors = [getattr(self.batch, n) for n in FIELDS]
        tensors += [self.batch.selection_mask, meta.request_anchor_ids,
                    meta.request_level_ids, meta.counts, meta.offsets,
                    meta.row_owner_ids, meta.row_owner_levels, meta.row_offset_slots]
        return sum(t.numel() * t.element_size() for t in tensors)


class TemporalBundleCache:
    """Bounded next-demand residency; source age is never extended by a hit.

    Request IDs refer to an immutable anchor table with exactly one LoD per
    row, making anchor ID lookup equivalent to lookup by (LoD, anchor ID).
    Instantiate a new cache when its CacheIdentity changes. Calls are serial
    transactions; a generation is replaced only after output is assembled.
    """
    def __init__(self, *, identity: CacheIdentity, anchor_levels, capacity_rows,
                 n_offsets=10, audit=False, max_age=1, fast_handoff=False,
                 union_arena=False, payload_half=False):
        identity.validate()
        if capacity_rows < n_offsets or n_offsets < 1:
            raise ValueError("row capacity must fit a complete maximum-size bundle")
        if anchor_levels.dtype != torch.long or anchor_levels.ndim != 1:
            raise ValueError("anchor levels must be rank-one int64")
        if not anchor_levels.numel() or bool((anchor_levels < 0).any()):
            raise ValueError("anchor table must be nonempty with nonnegative levels")
        self.identity = identity
        self.levels = anchor_levels.clone()
        self.capacity_rows = capacity_rows
        self.n_offsets = n_offsets
        if max_age not in (1, 2, 3):
            raise ValueError("max_age must be one, two, or three")
        if union_arena and payload_half:
            raise ValueError("arena and fp16 are separate ablations")
        self.max_age = max_age
        self.fast_handoff = fast_handoff
        self.union_arena = union_arena
        self.payload_half = payload_half
        self.audit = audit
        self.generation: Optional[TemporalGeneration] = None
        self.last_frame = -1
        self._lock = Lock()

    def reset(self):
        with self._lock:
            self.generation = None
            self.last_frame = -1

    def _check_ids(self, ids):
        if ids.dtype != torch.long or ids.ndim != 1 or ids.device != self.levels.device:
            raise ValueError("requests must be rank-one int64 on the cache device")
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= self.levels.numel()):
            raise ValueError("request ID outside immutable anchor table")
        if torch.unique(ids).numel() != ids.numel():
            raise ValueError("duplicate request IDs")

    def _decode(self, ids, decode, stats):
        stats["decoder_calls"] += int(ids.numel() > 0)
        stats["decoded_anchors"] += ids.numel()
        # Empty calls are forbidden: build an empty contract from a typed template.
        if not ids.numel():
            from .full_bundle_cache import FullBundleCacheCore
            core = FullBundleCacheCore(identity=self.identity, anchor_levels=self.levels,
                                       capacity_rows=self.capacity_rows, n_offsets=self.n_offsets)
            return core._empty_fresh_batch(ids, self.levels[ids])
        batch = decode(ids, self.levels[ids])
        meta = batch.bundle_metadata
        if meta is None or not torch.equal(batch.anchor_indices, ids):
            raise ValueError("decoder changed request identity/order")
        if meta.request_level_ids is None or not torch.equal(meta.request_level_ids, self.levels[ids]):
            raise ValueError("decoder changed LoD identity")
        if bool((meta.counts > self.n_offsets).any()):
            raise ValueError("decoder exceeded maximum complete bundle length")
        if self.audit:
            batch.validate_contract()
        return batch

    def resolve(self, *, frame_id: int, anchor_ids, decode: Callable,
                next_ids=None, reuse_allowed=True):
        with self._lock:
            return self._resolve(frame_id, anchor_ids, decode, next_ids, reuse_allowed)

    def _resolve(self, frame_id, ids, decode, next_ids, reuse_allowed):
        if frame_id <= self.last_frame:
            raise ValueError("frame IDs must increase; reset before a new trace")
        self._check_ids(ids)
        if next_ids is not None:
            self._check_ids(next_ids)
        stats = dict(decoder_calls=0, decoded_anchors=0, selected_anchors=ids.numel(),
                     hit_anchors=0, empty_hits=0, prefetch_anchors=0,
                     evicted_anchors=0, resident_rows=0, resident_descriptors=0,
                     resident_bytes=0, source_age=0, source_frame=-1, hit_rows=0)
        old = self.generation
        # Prefetch is a refresh transaction, never derived from stale hit output.
        if next_ids is not None:
            union, inverse = torch.unique(torch.cat((ids, next_ids)), sorted=True,
                                          return_inverse=True)
            fresh = self._decode(union, decode, stats)
            stats["prefetch_anchors"] = union.numel() - ids.numel()
            result = take_requests(fresh, inverse[:ids.numel()], ids, self.levels[ids])
            next_sorted = torch.sort(next_ids).values
            positions = torch.searchsorted(union, next_sorted)
            counts = fresh.bundle_metadata.counts[positions]
            # Whole-bundle prefix admission. Empty records do not consume rows,
            # but are bounded by the immutable anchor universe (no unbounded map).
            fits = (counts.cumsum(0) <= self.capacity_rows) | (counts == 0)
            kept = next_sorted[fits]
            if self.union_arena and fresh.xyz.shape[0] <= self.capacity_rows:
                # Keep a bounded immutable union arena; avoid a full compaction
                # copy. All physically retained rows count against capacity.
                retained = fresh
            else:
                retained = take_requests(fresh, positions[fits], kept, self.levels[kept])
            if self.payload_half:
                from dataclasses import replace
                retained = replace(retained, **{name: getattr(retained, name).half() for name in FIELDS})
            pending = TemporalGeneration(frame_id, retained)
            stats["evicted_anchors"] = int((~fits).sum())
        elif old is not None and 1 <= frame_id - old.source_frame <= self.max_age and reuse_allowed:
            stored = old.batch
            keys = stored.anchor_indices
            exact_request_match = self.fast_handoff and torch.equal(keys, ids)
            positions = (torch.arange(ids.numel(), device=ids.device) if exact_request_match
                         else torch.searchsorted(keys, ids))
            if keys.numel():
                safe = positions.clamp(max=keys.numel()-1)
                hits = (positions < keys.numel()) & (keys[safe] == ids)
            else:
                hits = torch.zeros_like(ids, dtype=torch.bool)
            stats["hit_anchors"] = int(hits.sum())
            stats["source_age"] = frame_id - old.source_frame
            stats["source_frame"] = old.source_frame
            stats["hit_rows"] = int(stored.bundle_metadata.counts[positions[hits]].sum())
            stats["empty_hits"] = int((stored.bundle_metadata.counts[positions[hits]] == 0).sum())
            if exact_request_match:
                # Ownership transfer at final use: the cache drops this immutable
                # generation after the frame. No payload gather or metadata rebuild.
                # With a longer lifetime, the caller must treat output as read-only.
                result = stored
            elif bool(hits.all()):
                result = take_requests(stored, positions, ids, self.levels[ids])
            elif not bool(hits.any()):
                result = self._decode(ids, decode, stats)
            else:
                cached = take_requests(stored, positions[hits], ids[hits], self.levels[ids[hits]])
                fresh = self._decode(ids[~hits], decode, stats)
                # Concatenate complete blocks, then restore exact request order.
                blocks = (cached, fresh)
                block_ids = torch.cat([b.anchor_indices for b in blocks])
                counts = torch.cat([b.bundle_metadata.counts for b in blocks])
                joined = NeuralGaussianBatch(
                    anchor_indices=block_ids,
                    **{n: torch.cat([getattr(b, n) for b in blocks]) for n in FIELDS},
                    selection_mask=torch.ones(int(counts.sum()), device=ids.device, dtype=torch.bool),
                    sh_degree=fresh.sh_degree,
                    bundle_metadata=BundleMetadata(
                        request_anchor_ids=block_ids, request_level_ids=self.levels[block_ids],
                        counts=counts, offsets=torch.cat((counts.new_zeros(1), counts.cumsum(0))),
                        row_owner_ids=torch.cat([b.bundle_metadata.row_owner_ids for b in blocks]),
                        row_owner_levels=torch.cat([b.bundle_metadata.row_owner_levels for b in blocks]),
                        row_offset_slots=torch.cat([b.bundle_metadata.row_offset_slots for b in blocks]),
                    ),
                )
                order = torch.empty_like(ids)
                order[hits] = torch.arange(cached.anchor_indices.numel(), device=ids.device)
                order[~hits] = torch.arange(fresh.anchor_indices.numel(), device=ids.device) + cached.anchor_indices.numel()
                result = take_requests(joined, order, ids, self.levels[ids])
            pending = old if frame_id - old.source_frame < self.max_age else None
        else:
            result = self._decode(ids, decode, stats)
            pending = None
        if self.audit:
            result.validate_contract()
            if pending is not None:
                pending.batch.validate_contract()
        # No reference to a partially constructed generation is ever published.
        self.generation = pending
        self.last_frame = frame_id
        if pending is not None:
            stats.update(resident_rows=pending.rows,
                         resident_descriptors=pending.batch.anchor_indices.numel(),
                         resident_bytes=pending.memory_bytes())
        stats["output_rows"] = result.xyz.shape[0]
        stats["empty_output_requests"] = int((result.bundle_metadata.counts == 0).sum())
        stats["resident_empty_descriptors"] = (
            int((pending.batch.bundle_metadata.counts == 0).sum()) if pending else 0)
        stats["capacity_rows"] = self.capacity_rows
        stats["lookahead_requests"] = next_ids.numel() if next_ids is not None else 0
        stats["miss_anchors"] = ids.numel() - stats["hit_anchors"]
        return result, stats
