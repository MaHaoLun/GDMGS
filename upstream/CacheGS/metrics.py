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

from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path

from utils.runtime_options import configure_device, output_root_for

# Retain compatibility with callers that supplied a preconstructed LPIPS model.
lpips_fn = None


def image_pair_paths(renders_dir, gt_dir):
    """Pair actual filenames deterministically and fail on any missing frame."""
    renders_dir, gt_dir = Path(renders_dir), Path(gt_dir)
    rendered = {path.name for path in renders_dir.iterdir() if path.is_file()}
    truth = {path.name for path in gt_dir.iterdir() if path.is_file()}
    if rendered != truth:
        raise ValueError("Frame pairing mismatch: missing GT={}, missing renders={}".format(
            sorted(rendered - truth), sorted(truth - rendered)))
    if not rendered:
        raise ValueError("No rendered frames in {}".format(renders_dir))
    return [(name, renders_dir / name, gt_dir / name) for name in sorted(rendered)]


def _load_image_pair(render_path, gt_path, device):
    from PIL import Image
    import torchvision.transforms.functional as tf

    with Image.open(render_path) as image:
        render = tf.to_tensor(image).unsqueeze(0)[:, :3, :, :].to(device)
    with Image.open(gt_path) as image:
        gt = tf.to_tensor(image).unsqueeze(0)[:, :3, :, :].to(device)
    if render.shape != gt.shape:
        raise ValueError("Image dimensions differ for {}".format(render_path.name))
    return render, gt


def readImages(renders_dir, gt_dir):
    """Compatibility helper; evaluation itself loads only one pair at a time."""
    renders, gts, names = [], [], []
    for name, render_path, gt_path in image_pair_paths(renders_dir, gt_dir):
        render, gt = _load_image_pair(render_path, gt_path, "cuda")
        renders.append(render)
        gts.append(gt)
        names.append(name)
    return renders, gts, names


def _parse_visibility_entry(entry):
    """Normalize legacy integer and structured per-view visibility records."""
    if isinstance(entry, dict):
        total = entry.get("visible_gaussians", entry.get("total"))
        if total is None:
            raise ValueError("Visibility record has no visible_gaussians or total")
        return (int(total), entry.get("levels", {}),
                entry.get("sample_voxels", entry.get("samples", [])), entry.get("cache"))
    if entry is None:
        raise ValueError("Missing per-frame visibility record")
    return int(entry), {}, [], None


def metric_output_paths(model_paths, output_root=None):
    if not model_paths:
        raise ValueError("At least one rendered scene root is required")
    if output_root is None:
        return [Path(path) for path in model_paths]
    names = [Path(path).resolve().name for path in model_paths]
    if len(model_paths) > 1 and len(set(names)) != len(names):
        raise ValueError("Scene directory names must be unique for a shared --output-root")
    # Prevent any explicit destination from entering any model/input directory.
    for path in model_paths:
        output_root_for(path, output_root)
    destinations = [Path(output_root)] if len(model_paths) == 1 else [Path(output_root) / name for name in names]
    for destination in destinations:
        for model_path in model_paths:
            output_root_for(model_path, destination)
    return destinations


def evaluate(model_paths, max_visibility_samples=10, output_root=None, device=None):
    import torch
    from tqdm import tqdm
    from utils.loss_utils import ssim
    from utils.image_utils import psnr

    destinations = metric_output_paths(model_paths, output_root)
    selected_device = configure_device(device, allow_cpu=True)
    evaluator = lpips_fn
    if evaluator is None:
        import lpips
        evaluator = lpips.LPIPS(net='vgg').to(selected_device)
    evaluator.eval()
    for scene_dir, destination in zip(model_paths, destinations):
        print("Scene:", scene_dir)
        summary, per_view = {}, {}
        test_dir = Path(scene_dir) / "test"
        methods = sorted(path for path in test_dir.iterdir() if path.is_dir())
        if not methods:
            raise ValueError("No render methods in {}".format(test_dir))
        for method_dir in methods:
            method = method_dir.name
            print("Method:", method)
            pairs = image_pair_paths(method_dir / "renders", method_dir / "gt")
            with open(method_dir / "per_view_count.json") as handle:
                visibility = json.load(handle)
            names = [pair[0] for pair in pairs]
            if set(visibility) != set(names):
                raise ValueError("Visibility records do not match image frames in {}".format(method_dir))
            values = {key: [] for key in ("PSNR", "SSIM", "LPIPS", "GS_NUMS")}
            level_hist, examples, hit_rates = Counter(), [], []
            for image_name, render_path, gt_path in tqdm(pairs, desc="Metric evaluation progress"):
                total, levels, voxels, cache = _parse_visibility_entry(visibility[image_name])
                values["GS_NUMS"].append(total)
                for level, count in levels.items():
                    level_hist[str(level)] += int(count)
                if voxels and len(examples) < max_visibility_samples:
                    examples.append({"image": image_name, "voxels": voxels})
                if isinstance(cache, dict) and cache.get("hit_rate") is not None:
                    hit_rates.append(float(cache["hit_rate"]))
                with torch.no_grad():
                    render, gt = _load_image_pair(render_path, gt_path, selected_device)
                    values["SSIM"].append(ssim(render, gt).item())
                    values["PSNR"].append(psnr(render, gt).item())
                    values["LPIPS"].append(evaluator(render, gt).item())
                del render, gt
            # Keep original metric definitions, float32 aggregation and JSON fields.
            summary[method] = {key: torch.tensor(scores).float().mean().item()
                               for key, scores in values.items()}
            per_view[method] = {key: dict(zip(names, scores)) for key, scores in values.items()}
            if level_hist:
                summary[method]["VIS_LEVELS"] = dict(level_hist)
            if examples:
                summary[method]["VIS_SAMPLES"] = examples
            if hit_rates:
                summary[method]["CACHE_HIT_RATE_MEAN"] = sum(hit_rates) / len(hit_rates)
            for key in ("PSNR", "SSIM", "LPIPS", "GS_NUMS"):
                print("  {}: {:>12.7f}".format(key, summary[method][key]))
        destination.mkdir(parents=True, exist_ok=True)
        with open(destination / "results.json", "w") as handle:
            json.dump(summary, handle, indent=2)
        with open(destination / "per_view.json", "w") as handle:
            json.dump(per_view, handle, indent=2)


def build_parser():
    parser = ArgumentParser(description="Evaluate rendered test images")
    parser.add_argument("--model_paths", "-m", required=True, nargs="+", default=[])
    parser.add_argument("--max_visibility_samples", type=int, default=10)
    parser.add_argument("--output-root", default=None, help="Separate JSON result destination; inputs are rendered scene roots passed with -m")
    parser.add_argument("--device", default=None, help="Logical CUDA device (or cpu for metrics)")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        metric_output_paths(args.model_paths, args.output_root)
    except ValueError as exc:
        parser.error(str(exc))
    evaluate(args.model_paths, max_visibility_samples=args.max_visibility_samples,
             output_root=args.output_root, device=args.device)


if __name__ == "__main__":
    main()
