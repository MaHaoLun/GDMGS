"""Shared serialization for the active fVDB precompute payload format."""

from __future__ import annotations

import time
from typing import Dict, Optional

import torch

from utils.fvdb_visibility import JaggedVisibilityDescriptor, _infer_batch_size


def package_precompute_frame_entry(
    indices: torch.Tensor,
    descriptor: Optional[JaggedVisibilityDescriptor],
) -> Dict[str, torch.Tensor]:
    """Serialize a single frame entry for torch.save payloads."""
    entry: Dict[str, torch.Tensor] = {
        "indices": indices.detach().to(dtype=torch.long, device="cpu").contiguous()
    }
    if descriptor is not None:
        entry.update(descriptor.to_serializable_tensors())
    return entry

def build_precompute_payload_header(
    model_path: Optional[str],
    iteration: int,
    total_anchors: int,
    num_frames: int,
    fvdb_grid,
) -> Dict[str, object]:
    """Create the metadata header stored in precompute payloads."""
    payload: Dict[str, object] = {
        "version": 2,
        "created_at": time.time(),
        "model_path": model_path or "unknown",
        "iteration": int(iteration),
        "total_anchors": int(total_anchors),
        "num_frames": int(num_frames),
    }
    if fvdb_grid is not None:
        payload["grid_meta"] = {
            "voxel_sizes": fvdb_grid.voxel_sizes.detach().cpu(),
            "origins": fvdb_grid.origins.detach().cpu(),
            "batch_size": _infer_batch_size(fvdb_grid),
        }
    return payload
