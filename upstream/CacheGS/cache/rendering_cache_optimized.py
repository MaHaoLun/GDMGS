"""
Lightweight cache for Gaussian rendering.

This implementation follows the paper's cache-centric pipeline: reuse decoded
Gaussians across consecutive frames, adjust cache depth with a simple guiding
function, and avoid heavyweight lookups that can slow FPS. The cache operates
on jagged visibility descriptors and relies on GridBatch-derived anchor indices
for hit detection.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple, Union

import torch

from utils.fvdb_visibility import JaggedVisibilityDescriptor


def _empty_gaussian_payload(device: torch.device) -> Dict[str, Optional[torch.Tensor]]:
    return {
        "xyz": None,
        "color": None,
        "opacity": None,
        "scaling": None,
        "rotation": None,
        "original_indices": None,
        "metadata": None,
    }


class DynamicCacheDepthScheduler:
    """Minimal dynamic depth scheduler used to balance quality and speed."""

    def __init__(
        self,
        initial_depth: int = 8,
        max_depth: int = 16,
        min_depth: int = 2,
        depth_decay_factor: float = 0.9,
        depth_growth_factor: float = 1.05,
    ) -> None:
        self.current_depth = int(initial_depth)
        self.max_depth = int(max_depth)
        self.min_depth = int(min_depth)
        self.depth_decay_factor = float(depth_decay_factor)
        self.depth_growth_factor = float(depth_growth_factor)

    def guiding_function(self, update_rate: float) -> int:
        """Update depth based on how many anchors are new in the current frame."""
        if update_rate >= 0.8:
            self.current_depth = max(self.min_depth, int(self.current_depth * self.depth_decay_factor))
        elif update_rate <= 0.2:
            self.current_depth = min(self.max_depth, int(self.current_depth * self.depth_growth_factor))
        return self.current_depth

    def should_flush_cache(self, consecutive_high_reuse_frames: int) -> bool:
        return consecutive_high_reuse_frames > (self.current_depth * 2)


class RenderingCache:
    """Cache decoded Gaussian parameters for reuse between frames."""

    def __init__(
        self,
        max_cache_size: int = 100000,
        device: torch.device = torch.device("cuda"),
        enable_dynamic_scheduling: bool = True,
        cache_depth_config: Optional[Dict] = None,
        block_size: int = 8,  # kept for API compatibility; unused in the simplified cache
    ) -> None:
        self.device = torch.device(device)
        self.capacity = int(max_cache_size)
        self.enable_dynamic_scheduling = bool(enable_dynamic_scheduling)
        self._block_size = int(block_size)

        cfg = cache_depth_config or {}
        self.depth_scheduler = DynamicCacheDepthScheduler(
            initial_depth=int(cfg.get("initial_depth", 8)),
            max_depth=int(cfg.get("max_depth", 16)),
            min_depth=int(cfg.get("min_depth", 2)),
            depth_decay_factor=float(cfg.get("depth_decay_factor", 0.9)),
            depth_growth_factor=float(cfg.get("depth_growth_factor", 1.05)),
        )
        self._warmup_depth: Optional[int] = int(cfg.get("warmup_depth", -1))
        if self._warmup_depth is not None and self._warmup_depth < 0:
            self._warmup_depth = None
        self._warmup_frames: int = int(cfg.get("warmup_frames", 0))

        self.stats: Dict[str, Union[int, float]] = {
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_evictions": 0,
            "total_queries": 0,
            "consecutive_high_reuse_frames": 0,
            "last_duplicate_rate": 0.0,
        }
        self.current_frame: int = 0
        self._per_level_stats: Dict[int, Dict[str, int]] = {}
        self._metadata_enabled: bool = False
        self._inserts_since_sort: int = 0
        self._sort_threshold: int = max(self.capacity // 32, 8192)

        self._init_storage()

    # -- storage helpers -------------------------------------------------
    def _init_storage(self) -> None:
        cap = self.capacity
        self._metadata_enabled = False
        self._cache_keys = torch.full((cap,), -1, dtype=torch.long, device=self.device)
        self._levels = torch.full((cap,), -1, dtype=torch.long, device=self.device)
        self._occupied = torch.zeros(cap, dtype=torch.bool, device=self.device)
        self._timestamps = torch.zeros(cap, dtype=torch.long, device=self.device)
        self._metadata: Optional[list] = None

        self._xyz = torch.zeros((cap, 3), dtype=torch.float32, device=self.device)
        self._color = torch.zeros((cap, 3), dtype=torch.float32, device=self.device)
        self._opacity = torch.zeros((cap, 1), dtype=torch.float32, device=self.device)
        self._scaling = torch.zeros((cap, 3), dtype=torch.float32, device=self.device)
        self._rotation = torch.zeros((cap, 4), dtype=torch.float32, device=self.device)

        self._keys_dirty = True
        self._sorted_keys = torch.empty(0, dtype=torch.long, device=self.device)
        self._sorted_positions = torch.empty(0, dtype=torch.long, device=self.device)

        self._prev_cache_keys = torch.empty(0, dtype=torch.long, device=self.device)
        self._inserts_since_sort = 0

    def _ensure_sorted_index(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not bool(self._occupied.any()):
            self._sorted_keys = torch.empty(0, dtype=torch.long, device=self.device)
            self._sorted_positions = torch.empty(0, dtype=torch.long, device=self.device)
            self._keys_dirty = False
            return self._sorted_keys, self._sorted_positions

        if not self._keys_dirty and self._sorted_keys.numel() > 0:
            return self._sorted_keys, self._sorted_positions
        if self._keys_dirty and self._sorted_keys.numel() > 0 and self._inserts_since_sort < self._sort_threshold:
            return self._sorted_keys, self._sorted_positions

        active = torch.nonzero(self._occupied, as_tuple=False).flatten()
        if active.numel() == 0:
            self._sorted_keys = torch.empty(0, dtype=torch.long, device=self.device)
            self._sorted_positions = torch.empty(0, dtype=torch.long, device=self.device)
            self._keys_dirty = False
            return self._sorted_keys, self._sorted_positions

        keys = self._cache_keys.index_select(0, active)
        sorted_keys, order = torch.sort(keys)
        self._sorted_keys = sorted_keys
        self._sorted_positions = active.index_select(0, order)
        self._keys_dirty = False
        self._inserts_since_sort = 0
        return self._sorted_keys, self._sorted_positions

    # -- lookup ----------------------------------------------------------
    def _lookup_positions(self, cache_keys: torch.Tensor) -> torch.Tensor:
        sorted_keys, sorted_pos = self._ensure_sorted_index()
        if sorted_keys.numel() == 0:
            return torch.full_like(cache_keys, -1, dtype=torch.long, device=self.device)

        idx = torch.searchsorted(sorted_keys, cache_keys)
        idx = torch.clamp(idx, 0, sorted_keys.numel() - 1)
        candidate_keys = sorted_keys.index_select(0, idx)
        match = candidate_keys == cache_keys

        positions = torch.full(cache_keys.shape, -1, dtype=torch.long, device=self.device)
        if not match.any():
            return positions

        candidate_pos = sorted_pos.index_select(0, idx)
        positions[match] = candidate_pos[match]
        return positions

    def _duplicate_rate(self, cache_keys: torch.Tensor) -> Tuple[float, torch.Tensor]:
        if cache_keys.numel() == 0:
            return 0.0, torch.zeros(0, dtype=torch.bool, device=self.device)

        if self._prev_cache_keys.numel() == 0:
            new_mask = torch.ones_like(cache_keys, dtype=torch.bool, device=self.device)
            return 0.0, new_mask

        prev = self._prev_cache_keys
        combined = torch.cat([prev, cache_keys.to(device=self.device, dtype=torch.long, non_blocking=True)], dim=0)
        unique, inverse = torch.unique(combined, sorted=True, return_inverse=True)
        prev_inv = inverse[: prev.numel()]
        curr_inv = inverse[prev.numel() :]
        lookup = torch.zeros(unique.numel(), dtype=torch.bool, device=self.device)
        lookup[prev_inv] = True
        duplicate_mask = lookup[curr_inv]

        new_mask = ~duplicate_mask
        duplicate_rate = duplicate_mask.float().mean().item() if duplicate_mask.numel() > 0 else 0.0
        return duplicate_rate, new_mask

    def _update_level_stats(self, level_ids: torch.Tensor, hit_mask: torch.Tensor) -> None:
        if level_ids.numel() == 0 or hit_mask.numel() == 0:
            return
        level_cpu = level_ids.to(device="cpu", dtype=torch.long)
        hit_cpu = hit_mask.to(device="cpu", dtype=torch.bool)
        miss_cpu = (~hit_cpu).to(device="cpu", dtype=torch.bool)
        max_level = int(level_cpu.max().item())
        if max_level < 0:
            return
        minlength = max_level + 1
        hits = torch.bincount(level_cpu[hit_cpu], minlength=minlength) if hit_cpu.any() else torch.zeros(
            minlength, dtype=torch.long
        )
        misses = torch.bincount(level_cpu[miss_cpu], minlength=minlength) if miss_cpu.any() else torch.zeros(
            minlength, dtype=torch.long
        )
        for level in range(minlength):
            h = int(hits[level].item())
            m = int(misses[level].item())
            if h == 0 and m == 0:
                continue
            entry = self._per_level_stats.setdefault(level, {"hits": 0, "misses": 0})
            entry["hits"] += h
            entry["misses"] += m

    def _summarize_level_stats(self, top_k: int = 3) -> Dict[int, Dict[str, float]]:
        ranked = sorted(
            self._per_level_stats.items(),
            key=lambda kv: kv[1]["hits"] + kv[1]["misses"],
            reverse=True,
        )
        summary: Dict[int, Dict[str, float]] = {}
        for level, stats in ranked[:top_k]:
            total = stats["hits"] + stats["misses"]
            if total <= 0:
                continue
            summary[level] = {
                "reuse": round(stats["hits"] / total, 3),
                "samples": float(total),
            }
        return summary

    # -- public API ------------------------------------------------------
    def process_frame(
        self,
        visibility: Union[torch.Tensor, JaggedVisibilityDescriptor],
        total_anchors: Optional[int] = None,  # kept for API compatibility
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Optional[torch.Tensor]], bool]:
        if not isinstance(visibility, JaggedVisibilityDescriptor):
            raise TypeError("RenderingCache.process_frame requires a JaggedVisibilityDescriptor visibility payload.")

        self.current_frame += 1
        if (
            self._warmup_depth is not None
            and self._warmup_frames > 0
            and self.current_frame <= self._warmup_frames
        ):
            self.depth_scheduler.current_depth = int(self._warmup_depth)

        cache_keys = visibility.anchor_indices().to(device=self.device, dtype=torch.long, non_blocking=True).view(-1)
        try:
            level_ids = visibility.level_ids.to(device=self.device, dtype=torch.long, non_blocking=True)
        except Exception:
            level_ids = None

        if cache_keys.numel() == 0:
            empty = torch.zeros(0, dtype=torch.bool, device=self.device)
            return empty, empty, _empty_gaussian_payload(self.device), False

        positions = self._lookup_positions(cache_keys)
        hit_mask = positions >= 0
        miss_mask = ~hit_mask

        # Touch timestamps to support simple LRU eviction.
        if hit_mask.any():
            hit_pos = positions[hit_mask]
            self._timestamps[hit_pos] = self.current_frame

        cached_gaussians = self._gather_cached(hit_mask, positions, cache_keys)

        self.stats["cache_hits"] += int(hit_mask.sum().item())
        self.stats["cache_misses"] += int(miss_mask.sum().item())
        self.stats["total_queries"] += 1

        duplicate_rate, new_mask = self._duplicate_rate(cache_keys)
        self.stats["last_duplicate_rate"] = duplicate_rate

        new_fraction = new_mask.float().mean().item() if new_mask.numel() > 0 else 0.0
        reuse_ratio = 1.0 - new_fraction
        if reuse_ratio >= 0.8:
            self.stats["consecutive_high_reuse_frames"] += 1
        else:
            self.stats["consecutive_high_reuse_frames"] = 0

        should_flush = False
        if self.enable_dynamic_scheduling:
            self.depth_scheduler.guiding_function(update_rate=new_fraction)
            if self.depth_scheduler.should_flush_cache(self.stats["consecutive_high_reuse_frames"]):
                self.flush_cache()
                should_flush = True

        if level_ids is not None and level_ids.numel() == cache_keys.numel():
            self._update_level_stats(level_ids, hit_mask)

        self._prev_cache_keys = cache_keys.detach().clone()

        return hit_mask, miss_mask, cached_gaussians, should_flush

    def store_cache(
        self,
        cache_keys: torch.Tensor,
        xyz: torch.Tensor,
        color: torch.Tensor,
        opacity: torch.Tensor,
        scaling: torch.Tensor,
        rotation: torch.Tensor,
        *,
        level_ids: Optional[torch.Tensor] = None,
        metadata: Optional[list] = None,
    ) -> None:
        cache_keys = cache_keys.to(device=self.device, dtype=torch.long, non_blocking=True).view(-1)
        n = int(cache_keys.numel())
        if n == 0:
            return

        if n > self.capacity:
            logging.info(
                "RenderingCache.store_cache(): skipping batch size (%d) over capacity (%d); not storing.",
                n,
                self.capacity,
            )
            return

        if n > self.capacity:
            logging.warning(
                "RenderingCache: incoming batch size (%d) exceeds capacity (%d); truncating.",
                n,
                self.capacity,
            )
            cache_keys = cache_keys[-self.capacity :]
            n = int(cache_keys.numel())
            xyz = xyz[-n:]
            color = color[-n:]
            opacity = opacity[-n:]
            scaling = scaling[-n:]
            rotation = rotation[-n:]
            if level_ids is not None:
                level_ids = level_ids[-n:]
            if metadata is not None and len(metadata) >= n:
                metadata = metadata[-n:]

        free_slots = torch.nonzero(~self._occupied, as_tuple=False).flatten()
        need = n - int(free_slots.numel())
        if need > 0:
            self._evict(need)
            free_slots = torch.nonzero(~self._occupied, as_tuple=False).flatten()

        target = free_slots[:n]
        self._cache_keys[target] = cache_keys
        self._xyz[target] = xyz.to(device=self.device, dtype=torch.float32, non_blocking=True)
        self._color[target] = color.to(device=self.device, dtype=torch.float32, non_blocking=True)
        self._opacity[target] = opacity.to(device=self.device, dtype=torch.float32, non_blocking=True)
        self._scaling[target] = scaling.to(device=self.device, dtype=torch.float32, non_blocking=True)
        self._rotation[target] = rotation.to(device=self.device, dtype=torch.float32, non_blocking=True)
        self._timestamps[target] = self.current_frame
        self._occupied[target] = True

        if level_ids is not None:
            self._levels[target] = level_ids.to(device=self.device, dtype=torch.long, non_blocking=True)
        else:
            self._levels[target] = -1

        if metadata is not None:
            if not self._metadata_enabled or self._metadata is None:
                self._metadata = [None for _ in range(self.capacity)]
                self._metadata_enabled = True
            for slot, meta in zip(target.to(device="cpu", dtype=torch.long).tolist(), metadata):
                self._metadata[slot] = meta
        elif self._metadata_enabled and self._metadata is not None:
            for slot in target.to(device="cpu", dtype=torch.long).tolist():
                self._metadata[slot] = None

        self._inserts_since_sort += int(n)
        self._keys_dirty = True

    def flush_cache(self) -> None:
        self._cache_keys.fill_(-1)
        self._levels.fill_(-1)
        self._occupied.zero_()
        self._timestamps.zero_()
        self._metadata = None
        self._metadata_enabled = False
        self._keys_dirty = True
        self._sorted_keys = torch.empty(0, dtype=torch.long, device=self.device)
        self._sorted_positions = torch.empty(0, dtype=torch.long, device=self.device)
        self._per_level_stats.clear()
        self.stats["consecutive_high_reuse_frames"] = 0
        self._prev_cache_keys = torch.empty(0, dtype=torch.long, device=self.device)

    def _evict(self, count: int) -> None:
        if count <= 0:
            return
        occupied_idx = torch.nonzero(self._occupied, as_tuple=False).flatten()
        if occupied_idx.numel() == 0:
            return
        if count >= occupied_idx.numel():
            self.flush_cache()
            self.stats["cache_evictions"] += int(occupied_idx.numel())
            return
        ts = self._timestamps.index_select(0, occupied_idx)
        _, order = torch.topk(ts, k=count, largest=False)
        to_evict = occupied_idx.index_select(0, order)
        self._cache_keys[to_evict] = -1
        self._levels[to_evict] = -1
        self._occupied[to_evict] = False
        self._timestamps[to_evict] = 0
        if self._metadata_enabled and self._metadata is not None:
            for slot in to_evict.to(device="cpu", dtype=torch.long).tolist():
                self._metadata[slot] = None
        self._keys_dirty = True
        self._inserts_since_sort = 0
        self.stats["cache_evictions"] += int(to_evict.numel())

    def _gather_cached(
        self,
        hit_mask: torch.Tensor,
        positions: torch.Tensor,
        cache_keys: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if not hit_mask.any():
            return _empty_gaussian_payload(self.device)
        pos_gpu = positions[hit_mask].to(device=self.device, dtype=torch.long, non_blocking=True)
        cached_meta = None
        if self._metadata_enabled and self._metadata is not None:
            cached_meta = []
            for pos in pos_gpu.to(device="cpu", dtype=torch.long).tolist():
                cached_meta.append(self._metadata[pos] if 0 <= pos < len(self._metadata) else None)
        return {
            "xyz": self._xyz.index_select(0, pos_gpu),
            "color": self._color.index_select(0, pos_gpu),
            "opacity": self._opacity.index_select(0, pos_gpu),
            "scaling": self._scaling.index_select(0, pos_gpu),
            "rotation": self._rotation.index_select(0, pos_gpu),
            "original_indices": cache_keys[hit_mask],
            "metadata": cached_meta,
        }

    def get_cache_statistics(self) -> Dict[str, Union[int, float]]:
        total_anchor_events = self.stats["cache_hits"] + self.stats["cache_misses"]
        hit_rate = (self.stats["cache_hits"] / total_anchor_events) if total_anchor_events > 0 else 0.0
        cache_size = int(self._occupied.sum().item())
        return {
            **self.stats,
            "cache_hit_rate": hit_rate,
            "cache_size": cache_size,
            "current_cache_size": cache_size,
            "current_cache_depth": self.depth_scheduler.current_depth,
            "current_frame": self.current_frame,
            "block_size": self._block_size,
            "per_level_summary": self._summarize_level_stats(),
        }

    def reset_statistics(self) -> None:
        self.stats = {key: 0 for key in self.stats}
        logging.info("Cache statistics reset")


def create_rendering_cache(config: Dict) -> RenderingCache:
    return RenderingCache(
        max_cache_size=config.get("max_cache_size", 100000),
        device=torch.device(config.get("device", "cuda")),
        enable_dynamic_scheduling=config.get("enable_dynamic_scheduling", True),
        cache_depth_config=config.get("cache_depth_config", {}),
        block_size=config.get("block_size", config.get("cache_block_size", 8)),
    )
