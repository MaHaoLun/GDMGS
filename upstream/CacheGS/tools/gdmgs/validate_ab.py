"""Bounded, real CUDA validation against an explicitly selected source tree.

This runner never writes the supplied checkpoint or dataset. Its sampled render
and temporary training checks are A/B evidence, not phase-G trajectory acceptance.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

from survey_inputs import (camera_infos, enumerate_records, file_stat,
                           load_model, read_config)


def controlled_legacy_import(source_root, name):
    """Confine the legacy import-time GPU chooser to the explicitly leased GPU."""
    import subprocess
    from unittest.mock import patch
    original = subprocess.run

    def run(command, *args, **kwargs):
        if command == 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used':
            device = int(os.environ["CUDA_VISIBLE_DEVICES"])
            values = [f"Used : {0 if i == device else 999999} MiB" for i in range(device + 1)]
            return subprocess.CompletedProcess(command, 0, stdout=("\n".join(values) + "\n").encode())
        return original(command, *args, **kwargs)

    spec = importlib.util.spec_from_file_location("validation_entry_" + name, Path(source_root) / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    with patch.object(subprocess, "run", run):
        spec.loader.exec_module(module)
    return module


def error_metrics(first, second):
    import torch
    from utils.loss_utils import ssim
    difference = first.float() - second.float()
    mse = difference.square().mean()
    return {"torch_equal": torch.equal(first, second), "max_abs": difference.abs().max().item(),
            "mae": difference.abs().mean().item(), "mse": mse.item(),
            "psnr": None if mse.item() == 0 else (-10 * torch.log10(mse)).item(),
            "ssim": ssim(first[None], second[None]).item()}


def training_fixture(source_root, output):
    """Fresh tiny production model; all gradients/statistics/checkpoints are real."""
    import numpy as np
    import torch
    import yaml
    from scene.gs_model_scaffoldgs.lod_model import GaussianLoDModel
    from scene.cameras import Camera
    from utils.general_utils import parse_cfg
    from gaussian_renderer import render
    config = yaml.safe_load((Path(source_root) / "config/scaffoldgs/lod_model.yaml").read_text())
    dataset, opt, pipe = parse_cfg(config)
    kwargs = dict(dataset.model_config["kwargs"])
    kwargs.update(levels=2, init_level=0, progressive=False, n_offsets=3, feat_dim=8)
    model = GaussianLoDModel(**kwargs)
    model.set_appearance(1)
    model.spatial_lr_scale = 1.0
    model.voxel_size = 0.1
    model.standard_dist = 5.0
    model.init_pos = torch.zeros(3, device="cuda")
    positions = torch.tensor([[-.25, -.25, 2.], [.25, -.25, 2.], [-.25, .25, 2.], [.25, .25, 2.]], device="cuda")
    model._anchor = torch.nn.Parameter(positions)
    model._anchor_feat = torch.nn.Parameter(torch.randn(4, 8, device="cuda") * 0.1)
    model._offset = torch.nn.Parameter(torch.randn(4, 3, 3, device="cuda") * 0.1)
    model._scaling = torch.nn.Parameter(torch.full((4, 6), -2., device="cuda"))
    model._rotation = torch.nn.Parameter(torch.tensor([[1., 0., 0., 0.]], device="cuda").repeat(4, 1), requires_grad=False)
    model._level = torch.zeros(4, 1, dtype=torch.int, device="cuda")
    model._extra_level = torch.zeros(4, device="cuda")
    model._anchor_mask = torch.ones(4, dtype=torch.bool, device="cuda")
    with torch.no_grad():
        model.mlp_opacity[2].bias.fill_(0.5)
    model._sync_grid_attributes(["anchor", "offset", "anchor_feat", "scaling", "rotation", "level", "extra_level", "anchor_mask"])
    model._mark_fvdb_dirty()
    model._refresh_fvdb_cache()
    model.set_coarse_interval(opt)
    model.training_setup(opt)
    model.train()
    camera = Camera(0, np.eye(3), np.zeros(3), 1., 1., torch.zeros(3, 64, 64), "fixture", 1., 0)
    before = {name: [p.detach().clone() for p in getattr(model, name).parameters()]
              for name in ("mlp_opacity", "mlp_cov", "mlp_color")}
    package = render(camera, model, pipe, torch.zeros(3, device="cuda"), 40000, "RGB", disable_cache=True)
    package["viewspace_points"].retain_grad()
    loss = (package["render"] - 0.3).square().mean()
    loss.backward()
    assert torch.isfinite(loss) and package["render"].dtype == torch.float32
    assert package["viewspace_points"].grad is not None
    gradient_names = [group["name"] for group in model.optimizer.param_groups
                      if any(p.grad is not None and torch.isfinite(p.grad).all() and torch.any(p.grad != 0) for p in group["params"])]
    assert "mlp_color" in gradient_names and "mlp_cov" in gradient_names
    model.training_statis(package, 64, 64)
    assert model.anchor_demon.sum() > 0 and model.offset_denom.sum() > 0
    stats = {name: float(getattr(model, name).sum()) for name in ("anchor_demon", "offset_denom", "opacity_accum", "offset_gradient_accum")}
    model.optimizer.step()
    changed = {name: any(not torch.equal(a, b) for a, b in zip(saved, getattr(model, name).parameters()))
               for name, saved in before.items()}
    assert all(changed.values()), changed
    model.optimizer.zero_grad(set_to_none=True)
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.save_ply(str(output / "point_cloud.ply"), 40000)
    model.save_mlp_checkpoints(str(output))
    restored = GaussianLoDModel(**kwargs)
    restored.set_appearance(1)
    restored.load_ply(str(output / "point_cloud.ply"))
    restored.load_mlp_checkpoints(str(output))
    restored.set_coarse_interval(opt)
    restored.eval()
    parameter_names = ("_anchor", "_offset", "_anchor_feat", "_scaling", "_rotation", "_level", "_extra_level")
    equality = {name: torch.equal(getattr(model, name), getattr(restored, name)) for name in parameter_names}
    assert all(equality.values()), equality
    with torch.no_grad():
        first = render(camera, model, pipe, torch.zeros(3, device="cuda"), 40000, "RGB", disable_cache=True)["render"]
        second = render(camera, restored, pipe, torch.zeros(3, device="cuda"), 40000, "RGB", disable_cache=True)["render"]
    comparison = error_metrics(first, second)
    assert comparison["torch_equal"], comparison
    return {"status": "pass", "loss": loss.item(), "gradient_groups": gradient_names,
            "optimizer_changed": changed, "training_stats": stats, "saved_tensor_equal": equality,
            "reload_render": comparison, "render_fields": sorted(package), "dtype": str(package["render"].dtype)}


def run(args):
    import numpy as np
    import torch
    import torchvision
    torch.manual_seed(0)
    np.random.seed(0)
    if args.mode == "training":
        return training_fixture(args.source_root, args.output / "training_fixture")
    if args.mode == "metrics":
        module = controlled_legacy_import(args.source_root, "metrics")
        # The old CLI creates this global under __main__; mirror that setup.
        if not hasattr(module, "build_parser"):
            module.lpips_fn = module.lpips.LPIPS(net="vgg").cuda()
        module.evaluate([str(args.output)])
        return {"status": "pass", "input_output_root": str(args.output), "gpu_chooser_intercepted": True}
    dataset, opt, pipe = read_config(args.model_path)
    inputs = [Path(args.model_path) / "config.yaml"] + sorted((Path(args.model_path) / "point_cloud" / f"iteration_{args.iteration}").iterdir())
    before = [file_stat(p) for p in inputs if p.is_file()]
    splits, kind = camera_infos(dataset)
    groups, records = enumerate_records(dataset, splits, kind)
    model = load_model(dataset, opt, args.iteration, len(splits["train"]))
    from utils.camera_utils import loadCam
    from gaussian_renderer.visibility import sample_visibility
    from utils.fvdb_conversion import build_precompute_payload_header, package_precompute_frame_entry
    selected = list(range(min(6, len(records))))
    rotations = np.stack([np.asarray(record["R"])[:, 2] for record in records])
    turn_index = int(np.argmin(rotations @ rotations[0]))
    if turn_index not in selected:
        selected.append(turn_index)
    bg = torch.ones(3, device="cuda") if dataset.white_background else torch.zeros(3, device="cuda")
    if dataset.random_background:
        bg = torch.rand(3, device="cuda")
    dataset.data_device = "cpu"
    cameras = []
    for index in selected:
        record = records[index]
        info = splits[record["split"]][record["uid"]]
        cameras.append(loadCam(dataset, record["uid"], info, record["resolution_scale"], bg))
    report = {"status": "pass", "mode": args.mode, "model_path": str(args.model_path),
              "iteration": args.iteration, "checkpoint_stats": before,
              "complete_camera_count": len(records), "frames": [records[i] for i in selected],
              "large_turn_frame": turn_index, "scope": "bounded A/B render regression", "results": []}
    if args.compare:
        original_manifest = json.loads((args.compare / "frame_manifest.json").read_text())
        for key in ("model_path", "iteration", "frames"):
            if original_manifest[key] != report[key]:
                raise ValueError(f"Baseline {key} differs; refusing to compare unrelated frames")
    (args.output / "frame_manifest.json").write_text(json.dumps(report, indent=2))
    precompute = args.output / "precomputed_indices.pt"
    if args.mode in ("precompute", "combined"):
        frames = []
        with torch.no_grad():
            for camera in cameras:
                model.set_anchor_mask(camera.camera_center, args.iteration, camera.resolution_scale)
                sample = sample_visibility(camera, model, pipe, bg)
                frames.append(package_precompute_frame_entry(sample.indices, sample.descriptor))
        payload = build_precompute_payload_header(str(args.model_path), args.iteration,
                                                len(model.get_anchor), len(frames), model.fvdb_grid)
        payload["frames"] = frames
        torch.save(payload, precompute)
        os.environ["PRECOMP_INDICES_PATH"] = str(precompute)
    os.environ["CACHE_ENABLE"] = "1" if args.mode in ("cache", "combined") else "0"
    renderer = importlib.reload(importlib.import_module("gaussian_renderer.render"))
    fresh_selected_ids = {}
    def capture_frame(position, package, elapsed):
        raw = package["render"].detach().cpu().clone()
        frame = {"frame_index": selected[position], "seconds": elapsed,
                 "visible_gaussians": int(package["visibility_filter"].sum()),
                 "dtype": str(raw.dtype), "fields": sorted(package),
                 "finite": bool(torch.isfinite(raw).all()), "cache_stats": package.get("cache_stats")}
        if "selected_anchor_ids" in package:
            fresh_selected_ids[position] = package["selected_anchor_ids"].detach().cpu().clone()
            frame["selected_anchor_count"] = int(fresh_selected_ids[position].numel())
        return raw, frame

    captured_dispatches = []
    if args.mode == "render2":
        from functools import wraps
        from unittest.mock import patch
        from utils.general_utils import get_render_func
        module = controlled_legacy_import(args.source_root, "render2")
        renderer_package = importlib.import_module("gaussian_renderer")
        function_name = get_render_func(dataset.base_model)
        actual_render = getattr(renderer_package, function_name)

        @wraps(actual_render)
        def capture_actual_dispatch(camera, *positional, **keywords):
            position = len(captured_dispatches)
            assert position < len(cameras) and camera is cameras[position], "render2 dispatch camera order changed"
            torch.cuda.synchronize()
            start = time.perf_counter()
            package = actual_render(camera, *positional, **keywords)
            torch.cuda.synchronize()
            captured_dispatches.append(capture_frame(position, package, time.perf_counter() - start))
            return package

        # Observe the actual CLI helper dispatch and return its original package.
        # No second rendering pass is used to manufacture comparison tensors.
        with patch.object(renderer_package, function_name, capture_actual_dispatch):
            module.render_set(dataset.base_model, str(args.output), "test", args.iteration,
                              cameras, model, pipe, bg, dataset.render_mode, -1, enable_cache=False)
        assert len(captured_dispatches) == len(cameras), "render2 did not dispatch every requested frame exactly once"
        report["gpu_chooser_intercepted"] = True
        report["actual_render2_dispatch_count"] = len(captured_dispatches)
        report["actual_render2_dispatch_order"] = [
            {"frame_index": selected[i], "uid": camera.uid, "image_name": camera.image_name,
             "resolution_scale": camera.resolution_scale}
            for i, camera in enumerate(cameras)]
    new_pipeline = None
    if args.mode == "gdmgs":
        from gdmgs.pipeline import FreshPipeline
        new_pipeline = FreshPipeline(model, str(args.model_path), args.iteration)
    for position, camera in enumerate(cameras):
        if args.mode == "render2":
            raw, frame = captured_dispatches[position]
        else:
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                package = (new_pipeline.render(camera, pipe, bg) if new_pipeline else
                           renderer.render(camera, model, pipe, bg, args.iteration, dataset.render_mode,
                                           disable_cache=args.mode not in ("cache", "combined")))
            torch.cuda.synchronize()
            raw, frame = capture_frame(position, package, time.perf_counter() - start)
        if args.compare:
            reference = torch.load(args.compare / f"frame_{position:05d}.pt", map_location="cpu", weights_only=True)
            frame["baseline_comparison"] = error_metrics(raw, reference)
        report["results"].append(frame)
        torch.save(raw, args.output / f"frame_{position:05d}.pt")
        folder = args.output / "test" / f"ours_{args.iteration}"
        (folder / "renders").mkdir(parents=True, exist_ok=True)
        (folder / "gt").mkdir(parents=True, exist_ok=True)
        if args.mode == "render2":
            from PIL import Image
            with Image.open(folder / "renders" / f"{position:05d}.png") as exported:
                saved = torch.from_numpy(np.array(exported)).permute(2, 0, 1)
            expected = (raw.clamp(0, 1) * 255 + 0.5).clamp(0, 255).to(torch.uint8)
            frame["actual_dispatch_png_equal"] = torch.equal(saved, expected)
            assert frame["actual_dispatch_png_equal"], "render2 exported PNG differs from its captured dispatch"
        else:
            torchvision.utils.save_image(raw.clamp(0, 1), folder / "renders" / f"{position:05d}.png")
            torchvision.utils.save_image(camera.original_image[:3], folder / "gt" / f"{position:05d}.png")
    counts = {f"{i:05d}.png": {"visible_gaussians": result["visible_gaussians"], "total": result["visible_gaussians"]}
              for i, result in enumerate(report["results"])}
    if args.mode != "render2":
        (folder / "per_view_count.json").write_text(json.dumps(counts, indent=2))
    report["cuda_max_memory_allocated"] = torch.cuda.max_memory_allocated()
    assert before == [file_stat(p["path"]) for p in before], "Input checkpoint mutated"
    report["checkpoint_stats_unchanged"] = True
    assert all(row["finite"] for row in report["results"])
    if args.compare:
        report["comparison_pass"] = all(row["baseline_comparison"]["torch_equal"] for row in report["results"])
        if not report["comparison_pass"]:
            report["status"] = "failure"
            report["error"] = "Rendered tensors differ from the fixed baseline"
    if args.mode == "gdmgs":
        from gdmgs.adapters import prepare_selection, materialize_selected
        from gaussian_renderer.render import rasterize_batch
        with torch.no_grad():
            pose0 = prepare_selection(new_pipeline.session, cameras[0], pipe, bg, -1)
            report["cross_pose_precheck"] = []
            for target in (1, len(cameras) - 1):
                target_selection = prepare_selection(new_pipeline.session, cameras[target], pipe, bg, -1)
                ids = target_selection.anchor_ids
                assert torch.equal(ids.cpu(), fresh_selected_ids[target]), "Target S_t differs from the actual fresh-render selection"
                decoded0 = materialize_selected(new_pipeline.session, ids, cameras[0], ape_code=-1, prepared=pose0)
                fresh = materialize_selected(new_pipeline.session, ids, cameras[target], ape_code=-1, prepared=target_selection)
                # Deliberately probe view-dependent decoding below the fresh-only
                # adapter, which correctly rejects cross-camera materialization.
                visible = torch.zeros(len(model.get_anchor), device="cuda", dtype=torch.bool)
                visible[ids] = True
                reused_image = rasterize_batch(cameras[target], model, decoded0.batch, bg, "RGB", visible_mask=visible)["render"]
                fresh_image = rasterize_batch(cameras[target], model, fresh.batch, bg, "RGB", visible_mask=visible)["render"]
                report["cross_pose_precheck"].append({"source_frame": selected[0], "target_frame": selected[target],
                    "same_selection_count": int(ids.numel()), "target_actual_selection_count": int(ids.numel()),
                    "selection_source": "target_pose_actual_fov_selection",
                    "same_target_ids_as_fresh_render": True,
                    "quality": error_metrics(reused_image, fresh_image), "c2_status": "not_frozen; measured_only",
                    "probe": "raw decoder bundle reused through rasterizer; not an enabled cache path"})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--mode", choices=("fresh", "cache", "precompute", "combined", "render2", "metrics", "gdmgs", "training"), required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--iteration", type=int, default=40000)
    args = parser.parse_args()
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("CUDA_VISIBLE_DEVICES must be explicitly set")
    if args.model_path and args.output.resolve().is_relative_to(args.model_path.resolve()):
        parser.error("Output must be isolated from the checkpoint directory")
    sys.path.insert(0, str(args.source_root.resolve()))
    if args.model_path:
        dataset, _, _ = read_config(args.model_path)
        if args.output.resolve().is_relative_to(Path(dataset.source_path).resolve()):
            parser.error("Output must be isolated from the dataset directory")
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_before = []
    if args.model_path:
        paths = [args.model_path / "config.yaml"] + sorted((args.model_path / "point_cloud" / f"iteration_{args.iteration}").iterdir())
        checkpoint_before = [file_stat(path) for path in paths if path.is_file()]
    try:
        result = run(args)
    except Exception as exc:
        result = {"status": "failure", "error": str(exc), "traceback": traceback.format_exc()}
    result.update(run_label=args.run_label, source_root=str(args.source_root), mode=args.mode,
                  cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"])
    if checkpoint_before:
        unchanged = checkpoint_before == [file_stat(item["path"]) for item in checkpoint_before]
        result["checkpoint_stats_unchanged"] = unchanged
        if not unchanged:
            result.update(status="failure", input_error="Checkpoint inputs changed during validation")
    (args.output / (args.mode + "_report.json")).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
