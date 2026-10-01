"""Render a frozen ProxyGS bundle through the GDM-GS gsplat backend."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import torchvision

from arguments import PipelineParams
from gaussian_renderer import generate_neural_gaussians
from gaussian_renderer.gdmgs_gsplat_backend import (
    camera_backend_settings,
    render_gdmgs_backend,
)
from gaussian_renderer.native_decoded_backend import render_native_decoded
from gaussian_renderer.raster_batch import (
    batch_from_proxygs_decode,
    validate_ordered_anchor_ids,
)
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.image_utils import psnr
from utils.loss_utils import ssim


BACKEND_ID = "gdmgs-gsplat-v1"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _file_identity(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _gpu_record() -> Dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True).strip().splitlines()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    return {
        "cuda_visible_devices": visible,
        "torch_device_name": torch.cuda.get_device_name(0),
        "torch_device_capability": list(torch.cuda.get_device_capability(0)),
        "nvidia_smi_visible_rows": output,
    }


def _environment_record() -> Dict[str, Any]:
    import gsplat
    import torchvision as torchvision_module

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision_module.__version__,
        "gsplat": gsplat.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": _gpu_record(),
    }


def _load_cfg(model_path: Path) -> Namespace:
    cfg_path = model_path / "cfg_args"
    text = cfg_path.read_text().strip()
    cfg = eval(text, {"__builtins__": {}, "Namespace": Namespace})
    if not isinstance(cfg, Namespace):
        raise ValueError("cfg_args did not evaluate to argparse.Namespace")
    cfg.model_path = str(model_path)
    return cfg


def _new_model(cfg: Namespace) -> GaussianModel:
    return GaussianModel(
        cfg.feat_dim,
        cfg.n_offsets,
        cfg.fork,
        cfg.use_feat_bank,
        cfg.appearance_dim,
        cfg.add_opacity_dist,
        cfg.add_cov_dist,
        cfg.add_color_dist,
        cfg.add_level,
        cfg.visible_threshold,
        cfg.dist2level,
        cfg.base_layer,
        cfg.progressive,
        cfg.extend,
    )


def _frozen_camera_names(model_path: Path) -> List[str]:
    records = json.loads((model_path / "cameras.json").read_text())
    names = [record["img_name"] for record in records]
    if len(names) != len(set(names)):
        raise ValueError("frozen camera inventory contains duplicate names")
    return names


def _ordered_views(scene: Scene, frozen_names: List[str]) -> List[Any]:
    views = scene.getTrainCameras() + scene.getTestCameras()
    by_name = {view.image_name: view for view in views}
    if len(by_name) != len(views):
        raise ValueError("runtime camera inventory contains duplicate names")
    if set(by_name) != set(frozen_names):
        missing = sorted(set(frozen_names) - set(by_name))
        extra = sorted(set(by_name) - set(frozen_names))
        raise ValueError(f"camera inventory mismatch: missing={missing[:5]} extra={extra[:5]}")
    return [by_name[name] for name in frozen_names]


def _load_explicit_ids(path: Path, camera_name: str, device: torch.device) -> torch.Tensor:
    payload = json.loads(path.read_text())
    values = payload.get(camera_name) if isinstance(payload, dict) else payload
    if values is None:
        raise ValueError(f"explicit selection has no IDs for camera {camera_name}")
    if not isinstance(values, list) or any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("explicit anchor IDs must be a JSON list of integers")
    return torch.tensor(values, dtype=torch.long, device=device)


def _selection_ids(args: argparse.Namespace, camera_name: str, model: GaussianModel) -> torch.Tensor:
    device = model.get_anchor.device
    if args.selection_mode == "all":
        ids = torch.arange(model.get_anchor.shape[0], dtype=torch.long, device=device)
    else:
        ids = _load_explicit_ids(args.anchor_ids_json, camera_name, device)
    return validate_ordered_anchor_ids(ids, anchor_count=model.get_anchor.shape[0], device=device)


def _mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return sum(values) / len(values) if values else None


def _render_one(
    *,
    view: Any,
    model: GaussianModel,
    pipeline: Any,
    background: torch.Tensor,
    anchor_ids: torch.Tensor,
    render_mode: str,
    warmup: int,
    repeat: int,
    native_diagnostic: bool,
    lpips_model: Any,
    record_explicit_ids: bool,
) -> tuple[Dict[str, Any], torch.Tensor, Optional[torch.Tensor]]:
    torch.cuda.synchronize()
    decode_start = time.perf_counter()
    decoded = generate_neural_gaussians(
        view,
        model,
        is_training=False,
        anchor_indices=anchor_ids,
    )
    batch = batch_from_proxygs_decode(
        anchor_ids=anchor_ids,
        decoded=decoded,
        n_offsets=model.n_offsets,
    )
    torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - decode_start

    for _ in range(warmup):
        warmup_output = render_gdmgs_backend(view, batch, background, render_mode)
        del warmup_output
    torch.cuda.synchronize()

    render_seconds = []
    output = None
    for _ in range(repeat):
        torch.cuda.synchronize()
        render_start = time.perf_counter()
        output = render_gdmgs_backend(view, batch, background, render_mode)
        torch.cuda.synchronize()
        render_seconds.append(time.perf_counter() - render_start)
    assert output is not None
    image = torch.clamp(output["render"], 0.0, 1.0)
    gt = torch.clamp(view.original_image.to(device=image.device), 0.0, 1.0)
    if image.shape != gt.shape:
        raise ValueError(f"render/GT shape mismatch: {tuple(image.shape)} != {tuple(gt.shape)}")

    metric_record = {
        "psnr": float(psnr(image, gt).mean()),
        "ssim": float(ssim(image.unsqueeze(0), gt.unsqueeze(0))),
        "lpips": None,
    }
    if lpips_model is not None:
        metric_record["lpips"] = float(
            lpips_model(image.unsqueeze(0), gt.unsqueeze(0), normalize=True).mean()
        )

    native_image = None
    native_record = None
    if native_diagnostic:
        torch.cuda.synchronize()
        native_start = time.perf_counter()
        native_output = render_native_decoded(view, batch, pipeline, background)
        torch.cuda.synchronize()
        native_seconds = time.perf_counter() - native_start
        native_image = torch.clamp(native_output["render"], 0.0, 1.0)
        delta = image - native_image
        native_record = {
            "seconds": native_seconds,
            "mse": float(torch.mean(delta.square())),
            "mean_abs": float(torch.mean(delta.abs())),
            "max_abs": float(torch.max(delta.abs())),
            "psnr_between_backends": float(psnr(image, native_image).mean()),
        }

    record = {
        "camera": view.image_name,
        "requested_anchor_count": int(anchor_ids.numel()),
        "decoded_row_count": int(batch.xyz.shape[0]),
        "visible_gaussian_count": int(output["visibility_filter"].sum()),
        "decode_seconds": decode_seconds,
        "render_seconds": render_seconds,
        "render_seconds_mean": _mean(render_seconds),
        "metrics": metric_record,
        "native_diagnostic": native_record,
        "tensor_identity": batch.tensor_identity(),
        "backend_settings": camera_backend_settings(view, background, render_mode),
        "selection_contract": (
            {
                "mode": "explicit",
                "requested_anchor_ids": anchor_ids.detach().cpu().tolist(),
                "row_owner_ids": batch.bundle_metadata.row_owner_ids.detach().cpu().tolist(),
                "row_offset_slots": batch.bundle_metadata.row_offset_slots.detach().cpu().tolist(),
                "counts": batch.bundle_metadata.counts.detach().cpu().tolist(),
                "offsets": batch.bundle_metadata.offsets.detach().cpu().tolist(),
            }
            if record_explicit_ids
            else {
                "mode": "all",
                "ordered_range": [0, int(anchor_ids.numel())],
                "decoded_row_count": int(batch.xyz.shape[0]),
            }
        ),
    }
    return record, image, native_image


def render_scene(args: argparse.Namespace) -> Dict[str, Any]:
    if args.repeat < 1 or args.warmup < 0:
        raise ValueError("repeat must be positive and warmup must be nonnegative")
    if args.selection_mode == "explicit" and args.anchor_ids_json is None:
        raise ValueError("explicit selection requires --anchor-ids-json")
    if args.formal and args.selection_mode != "all":
        raise ValueError("formal Step 3 only permits selection_mode=all")
    if args.formal and args.max_views is not None:
        raise ValueError("formal Step 3 cannot reduce the camera denominator")

    model_path = args.model_path.resolve()
    output_dir = (args.output_root / args.scene / args.run_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command.txt").write_text(" ".join(map(str, sys.argv)) + "\n")
    _atomic_json(output_dir / "status.json", {"state": "starting", "scene": args.scene})

    cfg = _load_cfg(model_path)
    if Path(cfg.source_path).resolve() != args.source_path.resolve():
        raise ValueError(f"source path mismatch: cfg={cfg.source_path} cli={args.source_path}")
    cfg.source_path = str(args.source_path.resolve())
    cfg.data_device = "cpu"
    model = _new_model(cfg)
    scene = Scene(
        cfg,
        model,
        load_iteration=args.iteration,
        shuffle=False,
        resolution_scales=cfg.resolution_scales,
    )
    model.eval()

    frozen_names = _frozen_camera_names(model_path)
    views = _ordered_views(scene, frozen_names)
    if args.expected_views is not None and len(views) != args.expected_views:
        raise ValueError(f"expected {args.expected_views} views, found {len(views)}")
    if args.max_views is not None:
        views = views[: args.max_views]

    background_values = [1.0, 1.0, 1.0] if cfg.white_background else [0.0, 0.0, 0.0]
    background = torch.tensor(background_values, dtype=torch.float32, device="cuda")

    lpips_model = None
    if args.lpips:
        import lpips

        lpips_model = lpips.LPIPS(net="vgg").to(background.device).eval()

    inputs = {
        "cfg_args": _file_identity(model_path / "cfg_args"),
        "cameras_json": _file_identity(model_path / "cameras.json"),
        "point_cloud": _file_identity(
            model_path / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
        ),
        "opacity_mlp": _file_identity(
            model_path / "point_cloud" / f"iteration_{args.iteration}" / "opacity_mlp.pt"
        ),
        "cov_mlp": _file_identity(
            model_path / "point_cloud" / f"iteration_{args.iteration}" / "cov_mlp.pt"
        ),
        "color_mlp": _file_identity(
            model_path / "point_cloud" / f"iteration_{args.iteration}" / "color_mlp.pt"
        ),
    }
    _atomic_json(
        output_dir / "run_contract.json",
        {
            "schema": "proxygs_step3_g0_run_contract_v1",
            "backend": BACKEND_ID,
            "scene": args.scene,
            "selection_mode": args.selection_mode,
            "formal": args.formal,
            "iteration": args.iteration,
            "camera_count": len(views),
            "frozen_camera_count": len(frozen_names),
            "render_mode": args.render_mode,
            "background": background_values,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "native_diagnostic": args.native_diagnostic,
            "environment": _environment_record(),
            "inputs": inputs,
        },
    )

    render_dir = output_dir / "renders"
    native_dir = output_dir / "native_diagnostic"
    if args.save_images:
        render_dir.mkdir(exist_ok=True)
        if args.native_diagnostic != "none":
            native_dir.mkdir(exist_ok=True)

    records: List[Dict[str, Any]] = []
    _atomic_json(output_dir / "status.json", {"state": "running", "scene": args.scene})
    for index, view in enumerate(views):
        anchor_ids = _selection_ids(args, view.image_name, model)
        diagnostic = args.native_diagnostic == "every" or (
            args.native_diagnostic == "first" and index == 0
        )
        record, image, native_image = _render_one(
            view=view,
            model=model,
            pipeline=args.pipeline,
            background=background,
            anchor_ids=anchor_ids,
            render_mode=args.render_mode,
            warmup=args.warmup if index == 0 else 0,
            repeat=args.repeat,
            native_diagnostic=diagnostic,
            lpips_model=lpips_model,
            record_explicit_ids=args.selection_mode == "explicit",
        )
        record["index"] = index
        records.append(record)
        if args.save_images:
            torchvision.utils.save_image(image, render_dir / f"{view.image_name}.png")
            if native_image is not None:
                torchvision.utils.save_image(native_image, native_dir / f"{view.image_name}.png")
        _atomic_json(output_dir / "per_view.json", records)
        _atomic_json(
            output_dir / "status.json",
            {
                "state": "running",
                "scene": args.scene,
                "completed_views": len(records),
                "expected_views": len(views),
                "last_camera": view.image_name,
            },
        )
        del image, native_image
        torch.cuda.empty_cache()

    summary = {
        "schema": "proxygs_step3_g0_scene_summary_v1",
        "state": "complete",
        "backend": BACKEND_ID,
        "scene": args.scene,
        "view_count": len(records),
        "frozen_view_count": len(frozen_names),
        "selection_mode": args.selection_mode,
        "metrics": {
            "psnr": _mean(record["metrics"]["psnr"] for record in records),
            "ssim": _mean(record["metrics"]["ssim"] for record in records),
            "lpips": _mean(
                record["metrics"]["lpips"]
                for record in records
                if record["metrics"]["lpips"] is not None
            ),
        },
        "timing": {
            "decode_seconds_mean": _mean(record["decode_seconds"] for record in records),
            "render_seconds_mean": _mean(record["render_seconds_mean"] for record in records),
        },
        "native_diagnostic_views": sum(record["native_diagnostic"] is not None for record in records),
        "input_identities_after": {name: _file_identity(Path(item["path"])) for name, item in inputs.items()},
    }
    if summary["input_identities_after"] != inputs:
        raise RuntimeError("a frozen Step 2 input changed during rendering")
    _atomic_json(output_dir / "summary.json", summary)
    _atomic_json(output_dir / "status.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--iteration", type=int, default=40000)
    parser.add_argument("--expected-views", type=int)
    parser.add_argument("--selection-mode", choices=("all", "explicit"), default="all")
    parser.add_argument("--anchor-ids-json", type=Path)
    parser.add_argument("--render-mode", choices=("RGB", "RGB+ED"), default="RGB")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--native-diagnostic", choices=("none", "first", "every"), default="first")
    parser.add_argument("--lpips", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--max-views", type=int, help="Development only; forbidden with --formal")
    PipelineParams(parser)
    return parser


if __name__ == "__main__":
    parsed = _parser().parse_args()
    parsed.pipeline = Namespace(compute_cov3D_python=parsed.compute_cov3D_python, debug=parsed.debug)
    try:
        final_summary = render_scene(parsed)
    except Exception as error:
        failure_dir = parsed.output_root / parsed.scene / parsed.run_id
        _atomic_json(
            failure_dir / "status.json",
            {"state": "failed", "scene": parsed.scene, "error": repr(error)},
        )
        raise
    print(json.dumps(final_summary, indent=2, sort_keys=True))
