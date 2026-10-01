"""Read-only checkpoint and complete camera survey; no digest operations.

Run with an explicitly selected CUDA device. Camera images are opened lazily;
only requested render frames are materialized by the companion validator.
"""
from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def file_stat(path):
    path = Path(path)
    value = path.stat()
    return {"path": str(path.resolve()), "size": value.st_size,
            "mtime_ns": value.st_mtime_ns}


def read_config(model_path):
    import yaml
    from utils.general_utils import parse_cfg
    config = yaml.safe_load((Path(model_path) / "config.yaml").read_text())
    dataset, opt, pipe = parse_cfg(config)
    dataset.model_path = str(model_path)
    return dataset, opt, pipe


def camera_infos(dataset, *, city_all=True):
    """Call original camera readers without point-cloud conversion or Scene writes."""
    from scene import dataset_readers as readers
    source = Path(dataset.source_path)
    if (source / "sparse").exists():
        sparse = source / "sparse" / "0"
        if (sparse / "images.bin").exists():
            extrinsics = readers.read_extrinsics_binary(str(sparse / "images.bin"))
            intrinsics = readers.read_intrinsics_binary(str(sparse / "cameras.bin"))
        else:
            extrinsics = readers.read_extrinsics_text(str(sparse / "images.txt"))
            intrinsics = readers.read_intrinsics_text(str(sparse / "cameras.txt"))
        all_infos = readers.readColmapCameras(extrinsics, intrinsics, str(source / dataset.images))
        split = bool(dataset.eval)
        kind = "colmap"
    elif (source / "transforms.json").exists():
        all_infos = readers.readCamerasFromTransforms(str(source), "transforms.json")
        split = bool(dataset.eval) and not city_all
        kind = "city"
    elif (source / "transforms_train.json").exists():
        train = readers.readCamerasFromTransforms(str(source), "transforms_train.json")
        test = readers.readCamerasFromTransforms(str(source), "transforms_test.json")
        return {"train": train if dataset.eval else train + test,
                "test": test if dataset.eval else []}, "blender"
    else:
        raise ValueError(f"Unknown dataset: {source}")
    return {"train": [c for i, c in enumerate(all_infos) if not split or i % 8 != 0],
            "test": [c for i, c in enumerate(all_infos) if split and i % 8 == 0]}, kind


def render_dimensions(info, dataset, resolution_scale):
    width, height = info.width, info.height
    if dataset.resolution in (1, 2, 4, 8):
        return [round(width / (resolution_scale * dataset.resolution)),
                round(height / (resolution_scale * dataset.resolution))]
    down = max(width / 1600, 1) if dataset.resolution == -1 else width / dataset.resolution
    return [int(width / (down * resolution_scale)), int(height / (down * resolution_scale))]


def enumerate_records(dataset, splits, kind):
    groups = {"train": [], "test": []}
    for scale in dataset.resolution_scales:
        for split, infos in splits.items():
            current = []
            for uid, info in enumerate(infos):
                current.append({"split": split, "uid": uid, "colmap_id": int(info.uid),
                                "image_name": info.image_name, "image": file_stat(info.image_path),
                                "source_dimensions": [info.width, info.height],
                                "render_dimensions": render_dimensions(info, dataset, scale),
                                "resolution_scale": scale, "pose_scale": getattr(dataset, "pose_scale", 1),
                                "R": info.R.tolist(), "T": info.T.tolist(),
                                "FoVx": float(info.FovX), "FoVy": float(info.FovY)})
            groups[split].extend(sorted(current, key=lambda c: c["image_name"]))
    merged = list(groups["train"])
    if kind == "colmap" and dataset.eval:
        merged += groups["test"]
        merged.sort(key=lambda c: c["image_name"])
    return groups, [dict(c, frame_index=i) for i, c in enumerate(merged)]


def load_model(dataset, opt, iteration, num_train_cameras):
    module = importlib.import_module("scene.gs_model_" + dataset.base_model)
    config = dataset.model_config
    model = getattr(module, config["name"])(**config["kwargs"])
    model.set_appearance(num_train_cameras)
    checkpoint = Path(dataset.model_path) / "point_cloud" / f"iteration_{iteration}"
    model.load_ply(str(checkpoint / "point_cloud.ply"))
    model.load_mlp_checkpoints(str(checkpoint))
    model.model_path = dataset.model_path
    model.eval()
    model.set_coarse_interval(opt)
    return model


def survey_one(model_path, iteration):
    import numpy as np
    import torch
    from plyfile import PlyData
    dataset, opt, _ = read_config(model_path)
    checkpoint = Path(model_path) / "point_cloud" / f"iteration_{iteration}"
    paths = [Path(model_path) / "config.yaml"] + sorted(checkpoint.iterdir())
    before = [file_stat(p) for p in paths if p.is_file()]
    start = time.perf_counter()
    splits, kind = camera_infos(dataset)
    groups, merged = enumerate_records(dataset, splits, kind)
    model = load_model(dataset, opt, iteration, len(splits["train"]))
    nonfinite_fields = {}
    invalid_fields = {}
    negative_infinity_log_scales = 0
    for name in ("_anchor", "_offset", "_anchor_feat", "_scaling", "_rotation", "_level", "_extra_level",
                 "get_scaling", "get_rotation", "init_pos"):
        tensor = getattr(model, name)
        count = int((~torch.isfinite(tensor)).sum())
        if count:
            nonfinite_fields[name] = count
        if name == "_scaling":
            negative_infinity_log_scales = int(torch.isneginf(tensor).sum())
            invalid_count = int((torch.isnan(tensor) | torch.isposinf(tensor)).sum())
        else:
            invalid_count = count
        if invalid_count:
            invalid_fields[name] = invalid_count
    for name in ("mlp_opacity", "mlp_cov", "mlp_color", "mlp_feature_bank", "embedding_appearance"):
        module = getattr(model, name, None)
        if module is not None:
            for parameter_name, parameter in module.named_parameters():
                count = int((~torch.isfinite(parameter)).sum())
                if count:
                    nonfinite_fields[f"{name}.{parameter_name}"] = count
                    invalid_fields[f"{name}.{parameter_name}"] = count
    ply = PlyData.read(str(checkpoint / "point_cloud.ply"))
    xyz = np.stack([np.asarray(ply.elements[0][axis]) for axis in ("x", "y", "z")], axis=1).astype(np.float32)
    row_equal = torch.equal(model.get_anchor.detach().cpu(), torch.from_numpy(xyz))
    # NaNs compare unequal as floats. Compare the original float32 bit patterns
    # directly to distinguish an unchanged corrupt input from reordered IDs.
    bit_equal = torch.equal(model.get_anchor.detach().cpu().contiguous().view(torch.int32),
                            torch.from_numpy(xyz).contiguous().view(torch.int32))
    finite_rows = np.isfinite(xyz).all(axis=1)
    coords, levels = model.fvdb_anchor_tables()
    if not bit_equal or coords.shape[0] != len(xyz) or levels.shape[0] != len(xyz):
        loaded = model.get_anchor.detach().cpu()
        source = torch.from_numpy(xyz)
        mismatched = torch.nonzero((loaded != source).any(dim=1)).flatten()
        detail = {"mismatch_rows": int(mismatched.numel()), "sample_rows": mismatched[:10].tolist(),
                  "ply_samples": source[mismatched[:10]].tolist(), "loaded_samples": loaded[mismatched[:10]].tolist(),
                  "ply_nan": int(torch.isnan(source).sum()), "loaded_nan": int(torch.isnan(loaded).sum()),
                  "max_abs": float((loaded - source).abs().max())}
        raise AssertionError(f"PLY row torch.equal={row_equal}; PLY={len(xyz)}; "
                             f"model={tuple(model.get_anchor.shape)}; coords={tuple(coords.shape)}; levels={tuple(levels.shape)}; {detail}")
    report = {"status": "pass" if not invalid_fields else "baseline_failure",
              "actual_load_status": "pass", "model_path": str(model_path), "iteration": iteration,
              "source_path": dataset.source_path, "kind": kind, "anchors": len(xyz),
              "ply_row_torch_equal": row_equal, "fvdb_loaded": model.fvdb_grid is not None,
              "ply_row_bit_torch_equal": bit_equal,
              "finite_anchor_rows": int(finite_rows.sum()), "nonfinite_anchor_rows": int((~finite_rows).sum()),
              "nonfinite_fields": nonfinite_fields,
              "invalid_fields": invalid_fields,
              "raw_negative_infinity_log_scales": negative_infinity_log_scales,
              "log_scale_policy": "-inf preserves the original zero activated scale; NaN/+inf or nonfinite activated values are invalid",
              "numerical_readiness": "pass" if not invalid_fields else "baseline_failure_invalid_source_parameters",
              "train_cameras": len(groups["train"]), "test_cameras": len(groups["test"]),
              "camera_groups": groups, "merged_render_frames": merged,
              "checkpoint_stats": before, "elapsed_seconds": time.perf_counter() - start,
              "cuda_max_memory_allocated": torch.cuda.max_memory_allocated()}
    assert before == [file_stat(p["path"]) for p in before], "Input checkpoint mutated"
    for infos in splits.values():
        for info in infos:
            info.image.close()
    del model, ply
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--iteration", type=int, default=40000)
    args = parser.parse_args()
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("CUDA_VISIBLE_DEVICES must be explicitly set")
    if args.output.resolve().is_relative_to(args.checkpoint_root.resolve()):
        parser.error("Survey output must be isolated from checkpoint inputs")
    sys.path.insert(0, str(args.source_root.resolve()))
    checkpoints = sorted(args.checkpoint_root.glob(f"**/point_cloud/iteration_{args.iteration}/point_cloud.ply"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"run_label": args.run_label, "source_root": str(args.source_root),
              "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "scenes": [],
              "scope": "complete camera inventories and actual model/MLP/fVDB loads; no full-trajectory claim"}
    for ply in checkpoints:
        try:
            result = survey_one(ply.parents[2], args.iteration)
        except Exception as exc:
            result = {"model_path": str(ply.parents[2]), "status": "failure", "error": str(exc), "traceback": traceback.format_exc()}
        report["scenes"].append(result)
        report["loaded"] = sum(s.get("actual_load_status") == "pass" for s in report["scenes"])
        report["baseline_failures"] = sum(s["status"] == "baseline_failure" for s in report["scenes"])
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k not in ("camera_groups", "merged_render_frames", "checkpoint_stats")}), flush=True)
    report["discovered"] = len(checkpoints)
    report["status"] = ("pass" if not report["baseline_failures"] else "baseline_failure") if len(checkpoints) == 13 and report["loaded"] == 13 else "failure"
    args.output.write_text(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
