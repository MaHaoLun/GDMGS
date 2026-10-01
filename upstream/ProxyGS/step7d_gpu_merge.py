"""Candidate all-GPU pair union from a bounded anchor-universe bitmap.

This is an independent B-plus-merge ablation candidate. It preserves the
sorted unique ID contract and returns the same maps as torch.unique(cat()).
"""
import torch

from step7d_execution_optimizations import RefreshPlan


def bitmap_refresh_plan(current, next_ids, anchor_count):
    if (current.dtype != torch.int64 or next_ids.dtype != torch.int64
            or current.device != next_ids.device or current.ndim != 1 or next_ids.ndim != 1):
        raise TypeError("pair selected IDs must be CUDA int64 vectors on the same device")
    if current.device.type != "cuda" or anchor_count <= 0:
        raise ValueError("GPU anchor universe must be nonempty")
    marks = torch.zeros(anchor_count, dtype=torch.bool, device=current.device)
    marks[current] = True
    marks[next_ids] = True
    union = torch.nonzero(marks, as_tuple=False).flatten()
    return RefreshPlan(current, next_ids, union,
                       torch.searchsorted(union, current),
                       torch.searchsorted(union, next_ids))
