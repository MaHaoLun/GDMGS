#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import logging
import math
import os
import time
from numbers import Integral
from typing import Optional, Tuple

import torch

from cache import rendering_cache_optimized as _cache_mod  # type: ignore
from gaussian_renderer.neural_gaussians import NeuralGaussianBatch, ensure_gaussian_batch as _ensure_gaussian_batch
from gaussian_renderer.fvdb_native_renderer import render_jagged_gaussians, validate_raster_inputs
from gaussian_renderer.visibility import (
    attach_descriptor_lookup,
    sample_visibility,
)
from utils.fvdb_visibility import JaggedVisibilityDescriptor


def _refresh_render_environment() -> None:
    """Read legacy environment settings at import or an explicit scene reset."""
    global _CACHE_ENABLE, _CACHE_LOG_STATS, _CACHE_LOG_INTERVAL, _CACHE_STORE_MAX_BATCH
    global _PROFILE_RENDER, _PROFILE_LOG_INTERVAL, _VIS_SUMMARY_SAMPLES
    global _PRECOMP_INDICES_PATH, _PRECOMP_LOOP, _PRECOMP_START_AT, _PRECOMP_STRICT
    global _KEYFRAME_ENABLE, _KEYFRAME_REUSE_THRESHOLD, _KEYFRAME_PROBE_N
    global _KEYFRAME_MIN_GAP, _KEYFRAME_POSE_THRESH

    # Cache integration (optional, env-driven)
    _CACHE_ENABLE = os.environ.get("CACHE_ENABLE", "0").lower() in {"1", "true", "yes"}
    _CACHE_LOG_STATS = os.environ.get("CACHE_LOG_STATS", "0").lower() in {"1", "true", "yes"}
    _CACHE_LOG_INTERVAL = int(os.environ.get("CACHE_LOG_INTERVAL", "30"))
    # Optional render-time profiler (disabled unless explicitly requested)
    _PROFILE_RENDER = os.environ.get("RENDER_PROFILE", "0").lower() in {"1", "true", "yes"}
    _PROFILE_LOG_INTERVAL = max(1, int(os.environ.get("RENDER_PROFILE_INTERVAL", "1")))
    # Visibility summary sampling
    _VIS_SUMMARY_SAMPLES = max(0, int(os.environ.get("VIS_SUMMARY_SAMPLES", "32")))
    # Offline precomputed indices integration
    _PRECOMP_INDICES_PATH = os.environ.get("PRECOMP_INDICES_PATH", "")
    _PRECOMP_LOOP = os.environ.get("PRECOMP_LOOP", "0").lower() in {"1", "true", "yes"}
    _PRECOMP_START_AT = int(os.environ.get("PRECOMP_START_AT", "0"))
    _PRECOMP_STRICT = os.environ.get("PRECOMP_STRICT", "1").lower() in {"1", "true", "yes"}
    # Online keyframe sampling (env-driven)
    _KEYFRAME_ENABLE = os.environ.get("CACHE_KEYFRAME_ENABLE", "0").lower() in {"1", "true", "yes"}
    _KEYFRAME_REUSE_THRESHOLD = float(os.environ.get("CACHE_KEYFRAME_REUSE", "0.8"))
    _KEYFRAME_PROBE_N = max(1, int(os.environ.get("CACHE_KEYFRAME_PROBE", "5")))
    _KEYFRAME_MIN_GAP = max(1, int(os.environ.get("CACHE_KEYFRAME_MIN_GAP", "2")))
    _KEYFRAME_POSE_THRESH = float(os.environ.get("CACHE_KEYFRAME_POSE_THRESH", "0.1"))

    # Optional cap on per-frame store batch; 0 disables.
    _CACHE_STORE_MAX_BATCH = int(os.environ.get("CACHE_STORE_MAX_BATCH", "0"))


_refresh_render_environment()
_PROFILE_FRAME = 0

_PRECOMP_CACHE_DEFAULTS = {
    "loaded": False,
    "indices": None,  # Flattened GPU tensor of anchor indices
    "ijk_data": None,  # Flattened jagged xyz for v2 payloads
    "ijk_jidx": None,  # Flattened jagged jidx for v2 payloads
    "offsets": None,  # Tensor for slicing per-frame ranges
    "num_frames": 0,
    "total_anchors": 0,
    "cursor": 0,
    "batch_size": 1,
    "version": 1,
}
_PRECOMP_CACHE = dict(_PRECOMP_CACHE_DEFAULTS)


# Reusable boolean mask buffers to avoid full zeroing each frame
_VISIBLE_MASK_STATE = {
    "buf": None,
    "last_idx": None,
    "size": 0,
    "device": None,
}

_RENDERING_CACHE: Optional[_cache_mod.RenderingCache] = None
_KEYFRAME_SAMPLER = None


def reset_render_context(*, refresh_environment: bool = True) -> None:
    """Release scene state and reread legacy settings only on explicit reset."""
    global _RENDERING_CACHE, _KEYFRAME_SAMPLER, _PROFILE_FRAME
    if refresh_environment:
        _refresh_render_environment()
    _PRECOMP_CACHE.clear()
    _PRECOMP_CACHE.update(_PRECOMP_CACHE_DEFAULTS)
    _VISIBLE_MASK_STATE.clear()
    _VISIBLE_MASK_STATE.update(buf=None, last_idx=None, size=0, device=None)
    _RENDERING_CACHE = None
    _KEYFRAME_SAMPLER = None
    _PROFILE_FRAME = 0


def rasterize_batch(
    viewpoint_camera,
    pc,
    gaussian_batch: NeuralGaussianBatch,
    bg_color: torch.Tensor,
    render_mode: str,
    *,
    visible_mask: torch.Tensor,
    visibility_summary: Optional[dict] = None,
    cache_stats: Optional[dict] = None,
    keyframe_decision: Optional[dict] = None,
) -> dict:
    """Rasterize an already-decoded batch without selection or cache side effects.

    This shared boundary preserves training tensors and gradients. Callers choose
    when to decode and whether to enter an inference/no-grad context.
    """
    if not isinstance(gaussian_batch, NeuralGaussianBatch):
        raise TypeError("rasterize_batch requires a NeuralGaussianBatch")
    validate_raster_inputs(
        gaussian_batch.xyz, gaussian_batch.color, gaussian_batch.opacity,
        gaussian_batch.scaling, gaussian_batch.rotation, gaussian_batch.sh_degree,
    )
    if visibility_summary is None:
        visibility_summary = {"total": int(gaussian_batch.anchor_indices.numel()), "levels": {}, "samples": []}
    return render_jagged_gaussians(
        viewpoint_camera=viewpoint_camera,
        gaussian_batch=gaussian_batch,
        bg_color=bg_color,
        render_mode=render_mode,
        visibility_summary=visibility_summary,
        cache_stats=cache_stats,
        keyframe_decision=keyframe_decision,
        visible_mask=visible_mask,
        renderer=getattr(pc, "fvdb_renderer", None),
    )


def _rotation_to_euler(rotation: torch.Tensor) -> Tuple[float, float, float]:
    """Convert a 3x3 rotation matrix to yaw/pitch/roll (radians)."""
    r00, r01, r02 = rotation[0, 0], rotation[0, 1], rotation[0, 2]
    r10, r11, r12 = rotation[1, 0], rotation[1, 1], rotation[1, 2]
    r20, r21, r22 = rotation[2, 0], rotation[2, 1], rotation[2, 2]
    sy = torch.sqrt(r00 * r00 + r10 * r10)
    singular = sy < 1e-6
    if not bool(singular):
        yaw = torch.atan2(r10, r00)
        pitch = torch.atan2(-r20, sy)
        roll = torch.atan2(r21, r22)
    else:
        yaw = torch.atan2(-r01, r11)
        pitch = torch.atan2(-r20, sy)
        roll = torch.tensor(0.0, device=rotation.device)
    return float(yaw), float(pitch), float(roll)


def _extract_pose(viewpoint_camera) -> Optional[dict]:
    """Extract translation and yaw/pitch/roll from a Camera."""
    try:
        view_inv = viewpoint_camera.world_view_transform.inverse()
        translation = view_inv[3, :3]
        rotation = view_inv[:3, :3]
        yaw, pitch, roll = _rotation_to_euler(rotation)
        return {
            "translation": translation,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
        }
    except Exception:
        return None


class KeyframeSampler:
    """Lightweight online keyframe/mapping selector driven by cache reuse."""

    def __init__(
        self,
        reuse_threshold: float,
        probe_every_n_frames: int,
        min_keyframe_gap: int,
        mapper_pose_threshold: float,
    ) -> None:
        self.reuse_threshold = reuse_threshold
        self.probe_every_n_frames = max(1, int(probe_every_n_frames))
        self.min_keyframe_gap = max(1, int(min_keyframe_gap))
        self.mapper_pose_threshold = mapper_pose_threshold
        self.last_keyframe_idx: int = -1
        self.last_mapper_idx: int = -1
        self.last_pose: Optional[dict] = None

    def _pose_delta(self, pose: Optional[dict]) -> Optional[float]:
        if pose is None or self.last_pose is None:
            return None
        try:
            delta_t = (pose["translation"] - self.last_pose["translation"]).norm().item()
        except Exception:
            delta_t = 0.0
        try:
            delta_rot = max(
                abs(pose["yaw"] - self.last_pose["yaw"]),
                abs(pose["pitch"] - self.last_pose["pitch"]),
                abs(pose["roll"] - self.last_pose["roll"]),
            )
        except Exception:
            delta_rot = 0.0
        return float(delta_t + delta_rot)

    def update(self, frame_idx: int, duplicate_rate: float, pose: Optional[dict]) -> dict:
        """Return a decision dictionary for the current frame."""
        pose_delta = self._pose_delta(pose)
        should_probe = (frame_idx % self.probe_every_n_frames) == 0
        eligible = (self.last_keyframe_idx < 0) or ((frame_idx - self.last_keyframe_idx) >= self.min_keyframe_gap)
        is_keyframe = False
        is_mapper = False
        reason = None

        if self.last_keyframe_idx < 0:
            is_keyframe = True
            is_mapper = True
            reason = "bootstrap"
        elif should_probe and eligible and duplicate_rate < self.reuse_threshold:
            is_keyframe = True
            reason = "reuse_drop"

        if pose_delta is not None and pose_delta > self.mapper_pose_threshold:
            is_mapper = True
            if eligible and not is_keyframe:
                is_keyframe = True
                reason = "pose_drift"

        if is_keyframe:
            self.last_keyframe_idx = frame_idx
            self.last_pose = pose
            if is_mapper:
                self.last_mapper_idx = frame_idx

        return {
            "is_keyframe": is_keyframe,
            "is_mapper": is_mapper or is_keyframe,
            "pose_delta": pose_delta if pose_delta is not None else 0.0,
            "probe": should_probe,
            "reason": reason or ("probe" if should_probe else "reuse"),
            "frame_idx": frame_idx,
        }


def _summarize_from_indices(pc, indices: torch.Tensor, sample_voxels: int) -> dict:
    summary = {
        "total": int(indices.numel()),
        "levels": {},
        "samples": [],
    }
    if indices.numel() == 0:
        return summary

    level_attr = getattr(pc, "_level", None)
    anchor_attr = pc.get_anchor if hasattr(pc, "get_anchor") else None

    try:
        if level_attr is not None:
            levels = level_attr[indices].to(device="cpu", dtype=torch.long)
            max_level = int(levels.max().item()) if levels.numel() > 0 else -1
            if max_level >= 0:
                counts = torch.bincount(levels, minlength=max_level + 1)
                summary["levels"] = {str(i): int(counts[i].item()) for i in range(max_level + 1) if counts[i].item() > 0}
    except Exception:
        summary["levels"] = {}

    if sample_voxels > 0 and anchor_attr is not None:
        try:
            sample_idx = indices[:sample_voxels]
            coords = anchor_attr[sample_idx].detach().to(device="cpu", dtype=torch.float32)
            levels_for_samples = None
            if level_attr is not None:
                levels_for_samples = level_attr[sample_idx].to(device="cpu", dtype=torch.long)
            samples = []
            for i in range(coords.shape[0]):
                entry = {
                    "ijk": [
                        float(coords[i, 0].item()),
                        float(coords[i, 1].item()),
                        float(coords[i, 2].item()),
                    ]
                }
                if levels_for_samples is not None:
                    entry["level"] = int(levels_for_samples[i].item())
                samples.append(entry)
            summary["samples"] = samples
        except Exception:
            summary["samples"] = []
    return summary


def _maybe_load_precomputed(
    num_anchors: Optional[int] = None,
    device: Optional[torch.device] = None,
    pc=None,
    iteration: int = -1,
    model_path: Optional[str] = None,
) -> None:
    """Load precomputed per-frame data once, if a path is provided."""
    if _PRECOMP_CACHE["loaded"]:
        return
    if not _PRECOMP_INDICES_PATH or not os.path.exists(_PRECOMP_INDICES_PATH):
        return
    if device is None:
        device = torch.device("cuda")
    try:
        payload = torch.load(_PRECOMP_INDICES_PATH)
        version = int(payload.get("version", 1))
        total_anchors = int(payload.get("total_anchors", 0))
        num_frames = int(payload.get("num_frames", 0))
        if num_frames <= 0:
            logging.warning("Precomputed payload is empty; ignoring PRECOMP_INDICES_PATH")
            return
        if version < 2:
            raise RuntimeError(
                f"Precomputed payload '{_PRECOMP_INDICES_PATH}' uses a legacy schema. "
                "Please rerun precompute_cache.py with a current fvdb-native checkpoint."
            )
        if num_anchors is not None and total_anchors > 0 and num_anchors != total_anchors:
            msg = (
                f"Precomputed total_anchors ({total_anchors}) != runtime anchors ({num_anchors}); "
                "precomputed indices will be ignored"
            )
            if _PRECOMP_STRICT:
                raise RuntimeError(msg)
            logging.warning(msg)
            return

        if version >= 2 and isinstance(payload.get("frames"), list):
            batch_size = payload.get("grid_meta", {}).get("batch_size", 1)
            if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size <= 0:
                raise ValueError("Precomputed grid_meta.batch_size must be a positive integer (not bool).")
            frames = payload["frames"]
            offsets = torch.zeros(num_frames + 1, dtype=torch.long, device=device)
            idx_chunks = []
            ijk_chunks = []
            jidx_chunks = []
            running = 0
            for i, frame in enumerate(frames):
                frame_indices = torch.as_tensor(frame.get("indices", []), dtype=torch.long, device=device)
                offsets[i] = running
                running += int(frame_indices.numel())
                idx_chunks.append(frame_indices)
                if "ijk_jdata" in frame and "ijk_jidx" in frame:
                    ijk_chunks.append(
                        torch.as_tensor(frame["ijk_jdata"], dtype=torch.int32, device=device)
                    )
                    jidx_chunks.append(
                        torch.as_tensor(frame["ijk_jidx"], dtype=torch.int16, device=device)
                    )
            offsets[num_frames] = running
            all_indices = torch.cat(idx_chunks, dim=0) if idx_chunks else torch.empty(0, dtype=torch.long, device=device)
            all_ijk = torch.cat(ijk_chunks, dim=0).to(device=device) if ijk_chunks else None
            all_jidx = torch.cat(jidx_chunks, dim=0).to(device=device) if jidx_chunks else None
            _PRECOMP_CACHE["indices"] = all_indices
            _PRECOMP_CACHE["ijk_data"] = all_ijk
            _PRECOMP_CACHE["ijk_jidx"] = all_jidx
            _PRECOMP_CACHE["batch_size"] = int(batch_size)
            _PRECOMP_CACHE["offsets"] = offsets
        else:
            indices_list = payload.get("indices")
            if not isinstance(indices_list, list):
                logging.warning("Precomputed indices file is invalid; ignoring PRECOMP_INDICES_PATH")
                return
            offsets = torch.zeros(num_frames + 1, dtype=torch.long, device=device)
            running = 0
            normalized = []
            for i, indices in enumerate(indices_list):
                tensor = torch.as_tensor(indices, dtype=torch.long, device=device)
                offsets[i] = running
                running += int(tensor.numel())
                normalized.append(tensor)
            offsets[num_frames] = running
            all_indices = torch.cat(normalized, dim=0) if normalized else torch.empty(0, dtype=torch.long, device=device)
            _PRECOMP_CACHE["indices"] = all_indices
            _PRECOMP_CACHE["ijk_data"] = None
            _PRECOMP_CACHE["ijk_jidx"] = None
            _PRECOMP_CACHE["offsets"] = offsets

        _PRECOMP_CACHE["num_frames"] = num_frames
        _PRECOMP_CACHE["total_anchors"] = total_anchors
        _PRECOMP_CACHE["cursor"] = max(0, min(_PRECOMP_START_AT, max(0, num_frames - 1)))
        _PRECOMP_CACHE["loaded"] = True
        _PRECOMP_CACHE["version"] = version
        logging.info(
            "Loaded precomputed payload (v%s): frames=%d total_anchors=%d total_indices=%d jagged=%s",
            version,
            num_frames,
            total_anchors,
            int(_PRECOMP_CACHE["indices"].numel() if _PRECOMP_CACHE["indices"] is not None else 0),
            "yes" if _PRECOMP_CACHE.get("ijk_data") is not None else "no",
        )
    except Exception as exc:
        logging.warning(f"Failed to load PRECOMP_INDICES_PATH='{_PRECOMP_INDICES_PATH}': {exc}")


def _get_precomputed_frame_payload(device: torch.device, pc=None) -> Optional[dict]:
    """Return precomputed data for the current frame and advance cursor."""
    if not _PRECOMP_CACHE["loaded"]:
        return None
    num_frames = int(_PRECOMP_CACHE["num_frames"])
    if num_frames <= 0:
        return None
    cursor = int(_PRECOMP_CACHE["cursor"])
    if cursor >= num_frames:
        if _PRECOMP_LOOP:
            cursor = 0
        else:
            cursor = num_frames - 1
    all_indices = _PRECOMP_CACHE["indices"]
    offsets = _PRECOMP_CACHE["offsets"]
    start = int(offsets[cursor].item())
    end = int(offsets[cursor + 1].item())
    frame_indices = all_indices[start:end].to(device=device, non_blocking=True)
    payload = {"indices": frame_indices}
    descriptor = None
    if pc is not None and _PRECOMP_CACHE.get("ijk_data") is not None and _PRECOMP_CACHE.get("ijk_jidx") is not None:
        try:
            import fvdb  # type: ignore

            frame_ijk = _PRECOMP_CACHE["ijk_data"][start:end].to(device=pc.get_anchor.device, non_blocking=True)
            frame_jidx = _PRECOMP_CACHE["ijk_jidx"][start:end].to(device=pc.get_anchor.device, non_blocking=True)
            descriptor = JaggedVisibilityDescriptor.from_serialized(
                pc.fvdb_grid,
                ijk_jdata=frame_ijk,
                ijk_jidx=frame_jidx,
                batch_size=int(_PRECOMP_CACHE.get("batch_size", 1)),
                legacy_indices=frame_indices,
            )
            attach_descriptor_lookup(pc, descriptor)
        except Exception as exc:  # pragma: no cover - defensive against stale payloads
            logging.debug("render(): failed to hydrate precomputed jagged descriptor: %s", exc)
    payload["descriptor"] = descriptor
    _PRECOMP_CACHE["cursor"] = cursor + 1
    return payload


def _get_or_update_mask(state: dict, total_anchors: int, device: torch.device, indices: torch.Tensor) -> torch.Tensor:
    """Return a reusable boolean mask by only toggling changed indices.

    This avoids full zeroing each frame and only writes previous True to False,
    and current indices to True.
    """
    buf = state.get("buf")
    if buf is None or state.get("size") != total_anchors or state.get("device") != device:
        # Initialize fresh buffer
        state["buf"] = torch.zeros(total_anchors, dtype=torch.bool, device=device)
        state["last_idx"] = torch.empty(0, dtype=torch.long, device=device)
        state["size"] = total_anchors
        state["device"] = device
        buf = state["buf"]
    # Clear previous indices
    last_idx = state.get("last_idx")
    if isinstance(last_idx, torch.Tensor) and last_idx.numel() > 0:
        buf.index_fill_(0, last_idx, False)
    # Set current indices
    if isinstance(indices, torch.Tensor) and indices.numel() > 0:
        buf.index_fill_(0, indices.to(device=device, dtype=torch.long), True)
    state["last_idx"] = indices
    return buf

def _get_rendering_cache() -> _cache_mod.RenderingCache:
    global _RENDERING_CACHE
    if _RENDERING_CACHE is None:
        cfg = {
            "max_cache_size": int(os.environ.get("CACHE_MAX_SIZE", 1000000)),
            "device": os.environ.get("CACHE_DEVICE", "cuda"),
            "enable_dynamic_scheduling": os.environ.get("CACHE_DYNAMIC", "1").lower() in {"1", "true", "yes"},
            "cache_depth_config": {
                "initial_depth": int(os.environ.get("CACHE_DEPTH_INIT", 10)),
                "max_depth": int(os.environ.get("CACHE_DEPTH_MAX", 16)),
                "min_depth": int(os.environ.get("CACHE_DEPTH_MIN", 8)),
                # Warmup: fix depth=10 for the first 10 frames to balance performance and quality
                "warmup_depth": int(os.environ.get("CACHE_WARMUP_DEPTH", 10)),
                "warmup_frames": int(os.environ.get("CACHE_WARMUP_FRAMES", 10)),
            },
        }
        # Optional: force a fixed reuse depth via env (e.g., CACHE_DEPTH_FIXED=10)
        try:
            _fixed_depth_str = os.environ.get("CACHE_DEPTH_FIXED", None)
            if _fixed_depth_str is not None and len(_fixed_depth_str) > 0:
                _fixed_depth = int(_fixed_depth_str)
                cfg["enable_dynamic_scheduling"] = False
                cfg["cache_depth_config"]["initial_depth"] = _fixed_depth
                cfg["cache_depth_config"]["max_depth"] = _fixed_depth
                cfg["cache_depth_config"]["min_depth"] = _fixed_depth
        except Exception:
            pass
        _RENDERING_CACHE = _cache_mod.create_rendering_cache(cfg)
    return _RENDERING_CACHE


def _get_keyframe_sampler() -> Optional[KeyframeSampler]:
    global _KEYFRAME_SAMPLER
    if not _KEYFRAME_ENABLE:
        return None
    if _KEYFRAME_SAMPLER is None:
        _KEYFRAME_SAMPLER = KeyframeSampler(
            reuse_threshold=_KEYFRAME_REUSE_THRESHOLD,
            probe_every_n_frames=_KEYFRAME_PROBE_N,
            min_keyframe_gap=_KEYFRAME_MIN_GAP,
            mapper_pose_threshold=_KEYFRAME_POSE_THRESH,
        )
    return _KEYFRAME_SAMPLER


def render(viewpoint_camera, pc, pipe, bg_color, iteration, render_mode, ape_code: int = -1, disable_cache: bool = True):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!

    Args:
        viewpoint_camera: Camera specifying view/projection.
        pc: Point cloud / model providing anchors and decoding.
        pipe: Unused pipeline/config placeholder.
        bg_color: Background color tensor on CUDA, shape [3].
        iteration: Current training/iteration step.
        render_mode: Rendering mode passed to rasterizer.
        ape_code: Optional APE code for decoding.
        disable_cache: If True, bypass cache and use baseline decoding path (equivalent to render ori.py).
    """
    pc.set_anchor_mask(viewpoint_camera.camera_center, iteration, viewpoint_camera.resolution_scale)
    total_anchors_int = int(pc.get_anchor.shape[0])

    cache_enabled = (not disable_cache) and _CACHE_ENABLE
    precompute_requested = bool(_PRECOMP_INDICES_PATH)

    coords_table: Optional[torch.Tensor] = None
    level_table: Optional[torch.Tensor] = None
    if hasattr(pc, "fvdb_anchor_tables"):
        try:
            coords_table, level_table = pc.fvdb_anchor_tables()
        except Exception:
            coords_table = None
            level_table = None

    pc_model_path = getattr(pc, "model_path", None)
    precomputed_payload = None
    if precompute_requested:
        _maybe_load_precomputed(
            num_anchors=total_anchors_int,
            device=pc.get_anchor.device,
            pc=pc,
            iteration=iteration,
            model_path=pc_model_path,
        )
        precomputed_payload = _get_precomputed_frame_payload(device=pc.get_anchor.device, pc=pc)

    visibility_descriptor: Optional[JaggedVisibilityDescriptor] = None
    frame_indices = precomputed_payload["indices"] if precomputed_payload is not None else None
    if precomputed_payload is not None:
        visibility_descriptor = precomputed_payload.get("descriptor")

    final_visible_anchors_ref: Optional[torch.Tensor] = None
    visible_mask_tensor: Optional[torch.Tensor] = None
    cache_stats: Optional[dict] = None
    keyframe_decision: Optional[dict] = None
    selection_mask = None
    sh_degree_to_use: Optional[int] = None

    if precomputed_payload is not None:
        if visibility_descriptor is None:
            raise RuntimeError(
                "render(): precomputed payload is missing jagged visibility metadata. "
                "Regenerate it with the current fvdb pipeline."
            )
        if frame_indices is None:
            raise RuntimeError("render(): precomputed payload is missing per-frame indices.")
        final_visible_anchors_ref = frame_indices
    else:
        sample = sample_visibility(
            viewpoint_camera,
            pc,
            pipe,
            bg_color,
            coords_table=coords_table,
            level_table=level_table,
            return_mask=not cache_enabled,
        )
        visibility_descriptor = sample.descriptor
        final_visible_anchors_ref = sample.indices
        if not cache_enabled:
            visible_mask_tensor = sample.mask

    if visibility_descriptor is None:
        raise RuntimeError(
            "render(): Failed to materialize a JaggedVisibilityDescriptor for the current frame. "
            "Ensure the checkpoint carries fvdb metadata or rerun precompute_cache.py."
        )

    profile_stats = {} if _PROFILE_RENDER else None

    def _profile_start():
        if profile_stats is None:
            return None
        torch.cuda.synchronize()
        return time.perf_counter()

    def _profile_end(token, label):
        if token is None:
            return
        torch.cuda.synchronize()
        profile_stats[label] = profile_stats.get(label, 0.0) + (time.perf_counter() - token) * 1000.0

    def _decode(visibility_payload, label, anchor_reference=None, ape_override=None):
        timer = _profile_start()
        result = pc.generate_neural_gaussians(
            viewpoint_camera,
            visibility_payload,
            ape_code if ape_override is None else ape_override,
        )
        _profile_end(timer, label)
        descriptor_override = visibility_payload if isinstance(visibility_payload, JaggedVisibilityDescriptor) else None
        return _ensure_gaussian_batch(
            pc,
            result,
            anchor_reference=anchor_reference,
            descriptor=descriptor_override,
        )

    def _maybe_log_profile(extra_stats=None):
        if profile_stats is None:
            return
        if extra_stats:
            profile_stats.update(extra_stats)
        global _PROFILE_FRAME
        frame_idx = _PROFILE_FRAME
        _PROFILE_FRAME += 1
        if frame_idx % _PROFILE_LOG_INTERVAL == 0:
            logging.info(
                "render_profile frame=%d iter=%d cache=%s stats=%s",
                frame_idx,
                iteration,
                "on" if cache_enabled else "off",
                profile_stats,
            )
        profile_stats.clear()

    def _resolve_visible_mask(
        indices: Optional[torch.Tensor], existing_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if existing_mask is not None:
            return existing_mask
        if indices is None or indices.numel() == 0:
            return torch.zeros(total_anchors_int, dtype=torch.bool, device=pc.get_anchor.device)
        return _get_or_update_mask(
            _VISIBLE_MASK_STATE,
            total_anchors_int,
            pc.get_anchor.device,
            indices,
        )

    descriptor_anchor_indices: Optional[torch.Tensor] = None
    descriptor_level_ids: Optional[torch.Tensor] = None

    if cache_enabled:
        cache = _get_rendering_cache()
        cache_device = getattr(cache, "device", pc.get_anchor.device)
        if final_visible_anchors_ref is None:
            raise RuntimeError("render(): cache path requires visible anchor indices.")
        try:
            descriptor_anchor_indices = visibility_descriptor.anchor_indices().to(
                device=cache_device, dtype=torch.long, non_blocking=True
            )
            descriptor_level_ids = visibility_descriptor.level_ids.to(
                device=cache_device, dtype=torch.long, non_blocking=True
            )
        except Exception as exc:  # pragma: no cover - descriptor hydration is best-effort
            logging.debug("render(): failed to materialize cache keys from descriptor: %s", exc)
            descriptor_anchor_indices = None
            descriptor_level_ids = None

        cache_payload = visibility_descriptor

        # Process through cache (pass total_anchors for lazy map init)
        cache_token = _profile_start()
        cache_hit_mask, cache_miss_mask, cached_gaussians, _ = cache.process_frame(
            cache_payload,
            total_anchors=pc.get_anchor.shape[0],
        )
        _profile_end(cache_token, "cache_process_ms")
        try:
            cache_stats = cache.get_cache_statistics()
        except Exception as exc:
            logging.debug("render(): failed to collect cache stats: %s", exc)
            cache_stats = None
        if cache_stats is not None and _KEYFRAME_ENABLE:
            sampler = _get_keyframe_sampler()
            if sampler is not None:
                try:
                    pose = _extract_pose(viewpoint_camera)
                    keyframe_decision = sampler.update(
                        cache_stats.get("current_frame", cache.current_frame),
                        cache_stats.get("last_duplicate_rate", 0.0),
                        pose,
                    )
                    cache_stats["keyframe"] = keyframe_decision
                except Exception as exc:
                    logging.debug("render(): keyframe sampler failed: %s", exc)
        if (
            cache_stats is not None
            and _CACHE_LOG_STATS
            and _CACHE_LOG_INTERVAL > 0
            and (cache.current_frame % _CACHE_LOG_INTERVAL == 0)
        ):
            try:
                logging.info(
                    f"Cache stats @frame {cache_stats.get('current_frame', cache.current_frame)}: "
                    f"hit_rate={cache_stats.get('cache_hit_rate', 0.0):.3f}, "
                    f"hits={cache_stats.get('cache_hits', 0)}, misses={cache_stats.get('cache_misses', 0)}, "
                    f"size={cache_stats.get('cache_size', cache_stats.get('current_cache_size', 'n/a'))}, "
                    f"depth={cache_stats.get('current_cache_depth', 'n/a')}, "
                    f"levels={cache_stats.get('per_level_summary', {})}"
                )
            except Exception:
                pass

        # Decode only misses
        miss_cache_keys: Optional[torch.Tensor] = None
        miss_level_ids: Optional[torch.Tensor] = None
        allow_store = True
        if cache_miss_mask.any():
            miss_anchor_indices = final_visible_anchors_ref[cache_miss_mask]
            if descriptor_anchor_indices is not None:
                try:
                    miss_cache_keys = descriptor_anchor_indices[cache_miss_mask]
                except Exception:
                    miss_cache_keys = descriptor_anchor_indices.index_select(
                        0, cache_miss_mask.nonzero(as_tuple=False).flatten()
                    )
            else:
                miss_cache_keys = miss_anchor_indices
            if descriptor_level_ids is not None:
                try:
                    miss_level_ids = descriptor_level_ids[cache_miss_mask]
                except Exception:
                    miss_level_ids = descriptor_level_ids.index_select(0, cache_miss_mask.nonzero(as_tuple=False).flatten())
            miss_batch = _decode(
                miss_anchor_indices,
                "decode_miss_ms",
                anchor_reference=miss_anchor_indices,
            )
            (
                new_xyz,
                new_color,
                new_opacity,
                new_scaling,
                new_rot,
                sh_payload,
                selection_payload,
                miss_metadata,
            ) = miss_batch.materialize(jagged=True)
            sh_degree_to_use = sh_payload
            new_selection_mask = selection_payload

            if miss_anchor_indices.numel() > 0:
                store_limit = _CACHE_STORE_MAX_BATCH if _CACHE_STORE_MAX_BATCH > 0 else getattr(cache, "capacity", None)
                if store_limit is not None and miss_anchor_indices.numel() > int(store_limit):
                    allow_store = False
                    logging.info(
                        "render(): skip cache.store_cache; miss batch %d exceeds store limit %d",
                        miss_anchor_indices.numel(),
                        store_limit,
                    )

                gaussian_count = int(new_xyz.shape[0]) if isinstance(new_xyz, torch.Tensor) else 0
                anchor_count = int(miss_anchor_indices.numel())

                n_offsets = getattr(pc, "n_offsets", None)
                if not (isinstance(n_offsets, int) and n_offsets > 0):
                    if anchor_count > 0 and gaussian_count > 0 and gaussian_count % anchor_count == 0:
                        n_offsets = gaussian_count // anchor_count
                    else:
                        n_offsets = 1

                def _repeat_to_gaussians(t: Optional[torch.Tensor], *, dim: int = 0) -> Optional[torch.Tensor]:
                    if t is None:
                        return None
                    expanded = t.repeat_interleave(n_offsets, dim=dim) if n_offsets > 1 else t
                    return expanded[:gaussian_count]

                cache_keys_per_gauss = _repeat_to_gaussians(miss_cache_keys)
                level_ids_per_gauss = _repeat_to_gaussians(miss_level_ids)

                mask_device = new_xyz.device if isinstance(new_xyz, torch.Tensor) else (
                    cache_keys_per_gauss.device if cache_keys_per_gauss is not None else miss_anchor_indices.device
                )
                if gaussian_count > 0:
                    if new_selection_mask is not None:
                        sel_mask = new_selection_mask.view(-1).to(device=mask_device, dtype=torch.bool)
                        if sel_mask.numel() != gaussian_count:
                            if sel_mask.numel() > 0 and gaussian_count % sel_mask.numel() == 0:
                                sel_mask = sel_mask.repeat_interleave(gaussian_count // sel_mask.numel())[:gaussian_count]
                            else:
                                logging.debug(
                                    "render(): selection mask size %s mismatches decoded gaussians %s; defaulting to full mask",
                                    sel_mask.numel(),
                                    gaussian_count,
                                )
                                sel_mask = torch.ones(gaussian_count, device=mask_device, dtype=torch.bool)
                    else:
                        sel_mask = torch.ones(gaussian_count, device=mask_device, dtype=torch.bool)
                else:
                    sel_mask = torch.zeros(0, device=mask_device, dtype=torch.bool)

                if isinstance(new_xyz, torch.Tensor):
                    new_xyz = new_xyz[sel_mask]
                    new_color = new_color[sel_mask]
                    new_opacity = new_opacity[sel_mask]
                    new_scaling = new_scaling[sel_mask]
                    new_rot = new_rot[sel_mask]

                def _apply_mask(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
                    if t is None:
                        return None
                    mask = sel_mask if sel_mask.device == t.device else sel_mask.to(device=t.device)
                    return t[:gaussian_count][mask]

                cache_keys_selected = _apply_mask(cache_keys_per_gauss) if cache_keys_per_gauss is not None else None
                level_ids_selected = _apply_mask(level_ids_per_gauss)

                n_selected = int(cache_keys_selected.numel()) if cache_keys_selected is not None else 0
                masked_gaussians = int(new_xyz.shape[0]) if isinstance(new_xyz, torch.Tensor) else 0

                metadata_payload = None
                if isinstance(miss_metadata, (list, tuple)) and len(miss_metadata) == gaussian_count:
                    if sel_mask.numel() == gaussian_count and bool(sel_mask.all()):
                        metadata_payload = list(miss_metadata)

                if allow_store and miss_cache_keys is not None and n_selected > 0 and n_selected == masked_gaussians:
                    cache.store_cache(
                        cache_keys_selected.to(device=cache_device, dtype=torch.long, non_blocking=True),
                        new_xyz,
                        new_color,
                        new_opacity,
                        new_scaling,
                        new_rot,
                        level_ids=level_ids_selected.to(device=cache_device, dtype=torch.long, non_blocking=True)
                        if level_ids_selected is not None
                        else None,
                        metadata=metadata_payload,
                    )
                else:
                    logging.debug(
                        f"render(): Skip cache.store_cache due to size mismatch/empty. decoded={gaussian_count} masked={masked_gaussians} cache_keys={n_selected}"
                    )
        else:
            new_xyz = new_color = new_opacity = new_scaling = new_rot = None

        # Compose cached and new gaussians
        if cache_hit_mask.any() and cached_gaussians["xyz"] is not None:
            if new_xyz is not None:
                xyz = torch.cat([cached_gaussians["xyz"], new_xyz], dim=0)
                color = torch.cat([cached_gaussians["color"], new_color], dim=0)
                opacity = torch.cat([cached_gaussians["opacity"], new_opacity], dim=0)
                scaling = torch.cat([cached_gaussians["scaling"], new_scaling], dim=0)
                rot = torch.cat([cached_gaussians["rotation"], new_rot], dim=0)
            else:
                xyz = cached_gaussians["xyz"]
                color = cached_gaussians["color"]
                opacity = cached_gaussians["opacity"]
                scaling = cached_gaussians["scaling"]
                rot = cached_gaussians["rotation"]
        else:
            # Only new results (first frame or after flush)
            xyz = new_xyz
            color = new_color
            opacity = new_opacity
            scaling = new_scaling
            rot = new_rot
            # If there was no decode in this frame yet, fallback to model's active sh degree if available
            if sh_degree_to_use is None:
                sh_degree_to_use = getattr(pc, "active_sh_degree", None)

        # Fallback: if nothing composed/decoded (e.g., empty visible set), decode all visible anchors
        def _empty(t: Optional[torch.Tensor]) -> bool:
            return t is None or not isinstance(t, torch.Tensor) or t.numel() == 0 or (t.dim() > 0 and t.shape[0] == 0)

        # If still empty, return a background-only frame to avoid rasterizer assertions
        if _empty(xyz) or _empty(color) or _empty(opacity) or _empty(scaling) or _empty(rot):
            # No requests produce all-None buffers; partial/invalid buffers are
            # malformed batches, not valid background-only renders.
            if not all(value is None for value in (xyz, color, opacity, scaling, rot)):
                validate_raster_inputs(xyz, color, opacity, scaling, rot, sh_degree_to_use)
            logging.debug("render(): Empty Gaussian set after cache compose; returning background-only frame.")
            H, W = int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)
            rendered_image = bg_color.view(-1, 1, 1).expand(3, H, W).contiguous().clone().requires_grad_(True)
            radii = torch.zeros(0, device=bg_color.device)
            info = {"means2d": torch.zeros(1, 0, 2, device=bg_color.device)}
            vis_mask_for_return = _resolve_visible_mask(final_visible_anchors_ref, visible_mask_tensor)
            sel_mask = torch.zeros_like(vis_mask_for_return)
            opacity = torch.zeros(0, 1, device=bg_color.device)
            _maybe_log_profile({"decoded_points": 0})
            return {
                "render": rendered_image,
                "scaling": torch.zeros(0, 3, device=bg_color.device),
                "viewspace_points": info["means2d"],
                "visibility_filter": radii > 0,
                "visible_mask": vis_mask_for_return,
                "selection_mask": sel_mask,
                "opacity": opacity,
                "render_depth": torch.zeros(1, H, W, device=bg_color.device) if "+" in render_mode else None,
                "render_alpha": torch.zeros(1, H, W, device=bg_color.device),
            }
    else:
        # Baseline decode when cache is disabled
        decode_payload = visibility_descriptor
        full_batch = _decode(
            decode_payload,
            "decode_full_ms",
            anchor_reference=final_visible_anchors_ref,
        )
        xyz, color, opacity, scaling, rot, sh_degree_to_use, selection_mask = full_batch.materialize()

    if visibility_descriptor is not None:
        visibility_summary = visibility_descriptor.summarize(sample_voxels=_VIS_SUMMARY_SAMPLES)
    elif final_visible_anchors_ref is not None:
        visibility_summary = _summarize_from_indices(pc, final_visible_anchors_ref, _VIS_SUMMARY_SAMPLES)
    else:
        visibility_summary = {"total": 0, "levels": {}, "samples": []}

    attribute_handles = {}
    handle_fn = getattr(pc, "_grid_attribute_handle", None)
    if callable(handle_fn):
        for name in ("anchor", "offset", "anchor_feat", "scaling", "rotation", "level", "extra_level"):
            handle = handle_fn(name)
            if handle is not None:
                attribute_handles[name] = handle

    gaussian_batch = NeuralGaussianBatch(
        descriptor=visibility_descriptor,
        anchor_indices=final_visible_anchors_ref,
        xyz=xyz,
        color=color,
        opacity=opacity,
        scaling=scaling,
        rotation=rot,
        selection_mask=selection_mask
        if selection_mask is not None
        else torch.ones(xyz.shape[0], dtype=torch.bool, device=xyz.device),
        sh_degree=sh_degree_to_use,
        attribute_handles=attribute_handles,
    )

    return rasterize_batch(
        viewpoint_camera=viewpoint_camera,
        pc=pc,
        gaussian_batch=gaussian_batch,
        bg_color=bg_color,
        render_mode=render_mode,
        visibility_summary=visibility_summary,
        cache_stats=cache_stats if cache_enabled else None,
        keyframe_decision=keyframe_decision if cache_enabled else None,
        visible_mask=_resolve_visible_mask(final_visible_anchors_ref, visible_mask_tensor),
    )


def render_2dgs(viewpoint_camera, pc, pipe, bg_color, iteration, render_mode):
    raise RuntimeError(
        "render_2dgs(): gsplat-based 2D renderer has been removed in favor of fvdb-native rendering."
    )
    
