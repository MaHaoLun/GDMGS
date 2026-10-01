#!/usr/bin/env python3
"""
Offline precompute of per-frame visible anchor indices for fast, index-only rendering.

This script traverses the scene cameras in the same deterministic order used by
render.py (no shuffling), computes the visible anchors for each view, and saves
them to a single torch file. At render time, set the environment variable
PRECOMP_INDICES_PATH to the saved file to bypass per-frame culling/indexing.

Usage:
  python precompute_cache.py -m <model_path> [--iteration -1] [--out <path>]

Notes:
- This stores only indices (not decoded Gaussian parameters) to keep storage
  compact and flexible across devices.
- Rendering should run with CACHE_ENABLE=0 to bypass dynamic caching when
  using precomputed indices.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

from utils.render_workflow import (enumerate_cameras, load_scene, load_inference_scene,
                                   load_model_config)
from utils.runtime_options import (add_runtime_arguments, initialize_runtime,
                                   output_root_for)


def _enumerate_cameras_in_order(dataset, gaussians, iteration: int):
    """Use exactly the merged render.py policy, including City's eval=False load."""
    scene = load_scene(dataset, gaussians, iteration, "merged")
    return enumerate_cameras(dataset, scene, "merged"), scene


def _save_payload(
    out_path: str,
    frame_payloads: List[Dict[str, torch.Tensor]],
    model_path: str,
    iteration: int,
    total_anchors: int,
    fvdb_grid,
) -> None:
    import torch
    from utils.fvdb_conversion import build_precompute_payload_header

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    payload = build_precompute_payload_header(
        model_path,
        iteration,
        total_anchors,
        len(frame_payloads),
        fvdb_grid,
    )
    payload["frames"] = frame_payloads
    torch.save(payload, out_path)
    print(f"Saved precomputed jagged visibility for {len(frame_payloads)} frames to: {out_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="Precompute per-frame visible anchor indices")
    parser.add_argument('-m', '--model_path', required=True, help='Path to trained model directory')
    parser.add_argument('--iteration', default=-1, type=int, help='Iteration to load (default latest)')
    parser.add_argument('--out', default=None, help='Output .pt path (default: <artifact root>/precomputed_indices.pt)')
    parser.add_argument('--quiet', action='store_true', help='Reduce logging output')
    add_runtime_arguments(parser, pipeline=False)
    return parser


def precompute_output_path(model_path, output_root=None, out=None):
    if out is not None and output_root is not None:
        raise ValueError("Choose either --out or --output-root for precompute output")
    if out is not None:
        # --out predates output-root: retain this explicit legacy destination.
        return out
    return os.path.join(output_root_for(model_path, output_root), 'precomputed_indices.pt')


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        out_path = precompute_output_path(args.model_path, args.output_root, args.out)
    except ValueError as exc:
        parser.error(str(exc))
    import torch
    from utils.fvdb_conversion import package_precompute_frame_entry
    from utils.general_utils import get_render_func

    device = initialize_runtime(args.quiet, args.device)
    from gaussian_renderer.visibility import sample_visibility, sample_visibility_2dgs
    dataset, opt, pipe = load_model_config(args.model_path)
    if args.device is not None:
        dataset.data_device = str(device)
    gaussians, scene = load_inference_scene(dataset, opt, args.iteration, "merged")
    cameras = enumerate_cameras(dataset, scene, "merged")

    render_func_name = get_render_func(dataset.base_model)
    is_2d_renderer = render_func_name == 'render_2dgs'

    total_anchors = int(gaussians.get_anchor.shape[0])
    fvdb_grid = getattr(gaussians, "fvdb_grid", None)
    coords_table: Optional[torch.Tensor] = None
    level_table: Optional[torch.Tensor] = None
    if fvdb_grid is not None and hasattr(gaussians, "fvdb_anchor_tables"):
        try:
            coords_table, level_table = gaussians.fvdb_anchor_tables()
        except Exception:
            coords_table = None
            level_table = None

    frame_payloads: List[Dict[str, torch.Tensor]] = []

    for idx, view in enumerate(cameras):
        # Keep internal anchor mask consistent with runtime
        gaussians.set_anchor_mask(view.camera_center, scene.loaded_iter, view.resolution_scale)
        if is_2d_renderer:
            sample = sample_visibility_2dgs(
                view,
                gaussians,
                pipe,
                scene.background,
            )
        else:
            sample = sample_visibility(
                view,
                gaussians,
                pipe,
                scene.background,
                coords_table=coords_table,
                level_table=level_table,
            )

        device = gaussians.get_anchor.device
        frame_indices = (
            sample.indices.to(dtype=torch.long, device=device)
            if isinstance(sample.indices, torch.Tensor)
            else torch.empty(0, dtype=torch.long, device=device)
        )
        descriptor = sample.descriptor
        if not is_2d_renderer and descriptor is None:
            raise RuntimeError(
                "[precompute] Failed to build JaggedVisibilityDescriptor; ensure the checkpoint carries fvdb metadata."
            )
        frame_payloads.append(
            package_precompute_frame_entry(
                frame_indices,
                descriptor,
            )
        )

    _save_payload(out_path, frame_payloads, args.model_path, scene.loaded_iter, total_anchors, fvdb_grid)


if __name__ == '__main__':
    main()
