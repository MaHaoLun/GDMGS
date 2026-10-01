"""Independently audit complete index experiments from recorded raw outputs.

No production query or metric helpers are imported. Images are evaluated one
frame at a time; raw float differences are kept separate from PNG-equivalent
quality metrics. Missing frames, settings, arrays or measurements fail closed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager, nullcontext
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil

import numpy as np


EXPECTED_SCENES = {
    "amsterdam": 161, "barcelona": 160, "bilbao": 129, "chicago": 160,
    "hollywood": 125, "pompidou": 161, "quebec": 160, "rome": 158,
    "drjohnson": 263, "playroom": 225, "train": 301, "truck": 251,
}
MODES = ("R0", "R1", "R2")
ORI_DEFINITION = "native_pixel_depth_tiles_v1"
ORI_IMAGE_SIZE_SOURCE = "frame.render_dimensions"
SUPPORT_CONTRACTS = {
    "decoder_all_view": ("gsplat-1.4-pinhole-classic-alpha-support-v3",
                         "all_decoder_offsets_max_last3_scale_full_bundle", "float16-outward"),
    "native_anchor_proxy": ("native_anchor_proxy_v1",
                            "native_fov_anchor_first3_scale_image_space_proxy", "float32-outward"),
}
PAYLOAD_TRANSFERS = ("fov_d2h", "triangle_ids_d2h", "triangle_ids_h2d",
                     "ori_tiles_d2h", "selected_ids_h2d")
TIMINGS = ("fov", "d2h", "retrieval", "ori", "anchor", "h2d",
           "decode_raster", "query", "total")
QUALITY_SETTINGS = {
    "psnr_drop_db": 0.1, "ssim_drop": 0.002,
    "preprocessing": "torchvision_png_equivalent",
    "ssim_window": 11, "ssim_sigma": 1.5,
}
WORKER_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolve_artifact(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def _convolve_axis(image, kernel, axis):
    try:
        from scipy.ndimage import convolve1d
    except ImportError:
        padding = [(0, 0)] * image.ndim
        padding[axis] = (len(kernel) // 2, len(kernel) // 2)
        windows = np.lib.stride_tricks.sliding_window_view(
            np.pad(image, padding), len(kernel), axis=axis)
        return np.einsum("...i,i->...", windows, kernel, optimize=True)
    return convolve1d(image, kernel, axis=axis, mode="constant", cval=0)


def independent_ssim(first, second):
    """11x11 sigma=1.5, zero padding, RGB mean: existing SSIM definition.

    Float64 separable convolution evaluates the same mathematical definition
    independently of utils.loss_utils and GPU convolution implementation.
    """
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    offsets = np.arange(-5, 6, dtype=np.float64)
    kernel = np.exp(-(offsets * offsets) / (2 * 1.5 ** 2))
    kernel /= kernel.sum()

    def blur(value):
        return _convolve_axis(_convolve_axis(value, kernel, 1), kernel, 2)

    mx, my = blur(x), blur(y)
    vx = blur(x * x) - mx * mx
    vy = blur(y * y) - my * my
    covariance = blur(x * y) - mx * my
    numerator = (2 * mx * my + 0.01 ** 2) * (2 * covariance + 0.03 ** 2)
    denominator = (mx * mx + my * my + 0.01 ** 2) * (vx + vy + 0.03 ** 2)
    return float(np.mean(numerator / denominator, dtype=np.float64))


def png_equivalent(image):
    """Match torchvision.save_image then RGB ToTensor, without writing PNG."""
    # save_image does float32 mul(255).add_(0.5).clamp_(0,255).to(uint8).
    value = np.asarray(image, dtype=np.float32)
    return np.clip(value * np.float32(255) + np.float32(0.5), 0, 255).astype(
        np.uint8).astype(np.float64) / 255.0


def psnr_from_mse(mse):
    return math.inf if mse == 0 else -10.0 * math.log10(mse)


def image_metrics(image, gt):
    x, y = png_equivalent(image), png_equivalent(gt)
    mse = float(np.mean((x - y) ** 2, dtype=np.float64))
    return {"psnr_db": psnr_from_mse(mse), "ssim": independent_ssim(x, y)}


def direct_metrics(first, second):
    difference = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    mse = float(np.mean(difference * difference, dtype=np.float64))
    return {"max_abs": float(np.max(np.abs(difference))),
            "mae": float(np.mean(np.abs(difference), dtype=np.float64)),
            "mse": mse, "psnr_db": psnr_from_mse(mse),
            "exact_equal": bool(np.array_equal(first, second))}


def metric_drop(baseline, candidate):
    if math.isinf(baseline) and math.isinf(candidate) and baseline == candidate:
        return 0.0
    return baseline - candidate


def audit_images(arrays, dimensions):
    width, height = dimensions
    images = {}
    for name in ("gt", "r0", "r0_repeat", "r1", "r2"):
        image = np.asarray(arrays[name])
        require(image.dtype == np.float32, f"{name}: raw image must be float32")
        require(image.shape == (3, height, width), f"{name}: changed image resolution/shape")
        require(np.isfinite(image).all(), f"{name}: nonfinite raw image")
        images[name] = image
    scores = {name: image_metrics(images[name], images["gt"])
              for name in ("r0", "r0_repeat", "r1", "r2")}
    result = {"metrics": scores, "direct": {}, "quality": {}}
    for name in ("r0_repeat", "r1", "r2"):
        result["direct"][name] = direct_metrics(images["r0"], images[name])
        psnr_drop = metric_drop(scores["r0"]["psnr_db"], scores[name]["psnr_db"])
        ssim_drop = metric_drop(scores["r0"]["ssim"], scores[name]["ssim"])
        result["quality"][name] = {
            "psnr_drop_db": psnr_drop, "ssim_drop": ssim_drop,
            "pass": bool(psnr_drop <= 0.1 and ssim_drop <= 0.002),
        }
    return result


def int64_ids(value, name, *, sorted_ids=False, maximum=None):
    value = np.asarray(value)
    require(value.dtype == np.int64 and value.ndim == 1, f"{name}: requires int64 1D")
    require(np.all(value >= 0), f"{name}: negative ID")
    require(len(np.unique(value)) == len(value), f"{name}: duplicate IDs")
    if sorted_ids:
        require(np.all(value[1:] > value[:-1]), f"{name}: must be strictly sorted")
    if maximum is not None:
        require(np.all(value < maximum), f"{name}: ID outside artifact")
    return value


def expand_ranges(ranges, dfs_to_row, name):
    ranges = np.asarray(ranges)
    require(ranges.dtype == np.int64 and ranges.ndim == 2 and ranges.shape[1] == 2,
            f"{name}: requires int64 (N,2)")
    require(np.all(ranges[:, 0] >= 0) and np.all(ranges[:, 1] <= len(dfs_to_row))
            and np.all(ranges[:, 0] < ranges[:, 1]), f"{name}: invalid half-open range")
    if len(ranges) == 0:
        return np.empty(0, dtype=np.int64)
    values = np.concatenate([dfs_to_row[start:end] for start, end in ranges])
    require(len(np.unique(values)) == len(values), f"{name}: duplicate denotation")
    return np.sort(values)


def validate_permutation(dfs_to_row):
    permutation = int64_ids(dfs_to_row, "dfs_to_row", maximum=len(dfs_to_row))
    require(np.array_equal(np.sort(permutation), np.arange(len(permutation))),
            "DFS order must be a complete source-row permutation")
    return permutation


def audit_selection(arrays, dfs_to_row, *, permutation_verified=False):
    permutation = dfs_to_row if permutation_verified else validate_permutation(dfs_to_row)
    fov = int64_ids(arrays["fov_ids"], "fov_ids", maximum=len(permutation))
    selected = {}
    for mode in ("r1", "r2"):
        actual_fov = int64_ids(arrays[mode + "_fov_ids"], mode + "_fov_ids", maximum=len(permutation))
        require(np.array_equal(actual_fov, fov), f"{mode}: actual GPU FoV differs from original frame")
        value = int64_ids(arrays[mode + "_selected_ids"], mode + "_selected_ids",
                          maximum=len(permutation))
        require(np.array_equal(value, fov[np.isin(fov, value)]),
                f"{mode}: selected IDs must preserve exact original FoV order")
        decoded = int64_ids(arrays[mode + "_decoded_anchor_ids"], mode + "_decoded_anchor_ids",
                            maximum=len(permutation))
        require(np.array_equal(decoded, value), f"{mode}: actual decoder consumed different IDs")
        for kind in ("raw", "formal"):
            expanded = expand_ranges(arrays[f"{mode}_{kind}_ranges"], permutation,
                                     f"{mode}_{kind}_ranges")
            require(np.array_equal(expanded, np.sort(value)),
                    f"{mode}: {kind} range denotation differs from selected IDs")
        selected[mode] = value
    # Same ORI plus a more conservative subtree certificate may retain more.
    require(np.all(np.isin(selected["r1"], selected["r2"])),
            "R2 deletes an anchor retained by independent per-anchor R1")
    scan = int64_ids(arrays["scan_triangle_ids"], "scan_triangle_ids", sorted_ids=True)
    bvh = int64_ids(arrays["bvh_triangle_ids"], "bvh_triangle_ids", sorted_ids=True)
    require(np.array_equal(scan, bvh), "BVH and scan triangle IDs differ")
    if "r3_selected_ids" in arrays:
        reused = int64_ids(arrays["r3_selected_ids"], "r3_selected_ids", maximum=len(permutation))
        require(np.array_equal(reused, selected["r2"]),
                "Range reuse selected IDs differ from rebuilding")
        for kind in ("raw", "formal"):
            expanded = expand_ranges(arrays[f"r3_{kind}_ranges"], permutation, f"r3_{kind}_ranges")
            require(np.array_equal(expanded, np.sort(reused)), f"R3 {kind} ranges differ from rebuilding")
    return {"fov": len(fov), "r1_selected": len(selected["r1"]),
            "r2_selected": len(selected["r2"]), "triangles": len(scan)}


def audit_manifest(manifest):
    require(manifest.get("schema_version") == 1, "Unsupported manifest schema")
    require(manifest.get("scope") == "full_setting", "Only a formal full_setting can pass final audit")
    require(bool(manifest.get("run_id")), "Missing run_id")
    require(manifest.get("excluded_scenes") == ["small_city"], "Only small_city is excluded")
    require(tuple(manifest.get("required_modes", [])) == MODES, "R0/R1/R2 required")
    require(manifest.get("timing_repeats") == 3, "Three frozen timing repetitions required")
    orders = manifest.get("timing_mode_order", [])
    require(len(orders) == 3 and all(set(row) == set(MODES) and len(row) == 3 for row in orders),
            "Each timing repetition must contain each mode exactly once")
    require(len({tuple(row) for row in orders}) == 3, "Timing mode order must rotate")
    machine = manifest.get("machine", {})
    require(all(machine.get(k) for k in ("hostname", "cpu_model", "gpu_uuid", "gpu_name")),
            "Missing measured machine/device identity")
    settings = manifest.get("settings", {})
    require(settings.get("query_device") == "gpu", "Final index comparison requires GPU R1 and GPU R2; CPU runs are diagnostic")
    require(all(isinstance(settings.get(k), int) and settings[k] > 0
                for k in ("cpu_threads", "torch_threads")), "Missing explicit CPU thread budget")
    require(all(settings.get(k) is False for k in
                ("cache_enabled", "precompute_enabled", "scheduling_enabled")),
            "Cache/precompute/scheduling must explicitly be disabled")
    require(settings.get("quality") == QUALITY_SETTINGS, "Quality contract changed")
    environment = settings.get("thread_environment", {})
    require(all(isinstance(environment.get(k), str) and environment[k].isdigit() and int(environment[k]) > 0
                for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Missing actual thread environment")
    scenes = manifest.get("scenes", [])
    require(Counter(s.get("scene") for s in scenes) == Counter(EXPECTED_SCENES.keys()),
            "Must contain each of the 12 in-setting scenes exactly once")
    for scene in scenes:
        name = scene["scene"]
        require(scene.get("scope") == "full_scene", f"{name}: development scene cannot pass final audit")
        ori_settings = scene.get("settings", {})
        require(ori_settings.get("query_device") == settings["query_device"],
                f"{name}: CPU/GPU query strategies may not be mixed in the final GPU comparison")
        require(ori_settings.get("materialization") == "fresh-no-cache-metadata-v1",
                f"{name}: missing explicit fresh materialization definition")
        require(ori_settings.get("support_profile") in SUPPORT_CONTRACTS,
                f"{name}: missing explicit support/profile choice")
        profile, definition, rounding = SUPPORT_CONTRACTS[ori_settings["support_profile"]]
        support = ori_settings.get("support", {})
        require((support.get("profile"), support.get("definition"), support.get("scale_rounding"))
                == (profile, definition, rounding), f"{name}: support/profile semantics disagree")
        require(ori_settings.get("ori_backend") == "pixel_depth",
                f"{name}: final ORI backend must be pixel_depth")
        require(ori_settings.get("ori_definition") == ORI_DEFINITION,
                f"{name}: final ORI must use the accepted native-pixel depth definition")
        require(type(ori_settings.get("tile_size")) is int and ori_settings["tile_size"] > 0,
                f"{name}: missing frozen positive tile_size")
        require(ori_settings.get("ori_image_size_source") == ORI_IMAGE_SIZE_SOURCE,
                f"{name}: ORI image dimensions must follow each native render frame")
        frames = scene.get("frames", [])
        require(len(frames) == EXPECTED_SCENES[name], f"{name}: incomplete trajectory")
        require([f["frame_index"] for f in frames] == list(range(len(frames))),
                f"{name}: missing/reordered frame IDs")
        require(scene.get("checkpoint_stats") and scene.get("model_path")
                and isinstance(scene.get("iteration"), int), f"{name}: missing source identity")
        require(scene.get("mesh_token") and scene.get("index_token")
                and scene.get("settings_token") and scene.get("dfs_to_row_path"),
                f"{name}: missing artifact/settings identities")
        require(scene.get("scene_token") and scene.get("mesh_query_index_token")
                and type(scene.get("anchor_nodes")) is int and scene["anchor_nodes"] >= 0,
                f"{name}: missing actual scene/index ownership metadata")
        require(len({f.get("camera_token") for f in frames}) == len(frames)
                and all(f.get("camera_token") for f in frames), f"{name}: invalid camera identities")
        for frame in frames:
            require(all(k in frame for k in ("uid", "image_name", "split", "R", "T",
                                             "FoVx", "FoVy", "render_dimensions",
                                             "resolution_scale")), f"{name}: incomplete camera calibration")
    return scenes


def audit_inventory_against_ab(manifest, reference):
    """Compare actual main-run enumeration with historical full A/B inventory.

    This reads existing generated evidence, not a new scan of source inputs.
    """
    prior = {Path(s["source_path"]).name: s for s in reference["scenes"]
             if "/matrixcity/" not in s["model_path"]}
    require(set(prior) == set(EXPECTED_SCENES), "A/B reference scene inventory is incomplete")
    fields = ("frame_index", "uid", "colmap_id", "image_name", "split", "R", "T",
              "FoVx", "FoVy", "resolution_scale", "pose_scale", "source_dimensions",
              "render_dimensions", "image")
    for scene in manifest["scenes"]:
        name = scene["scene"]
        original = prior[name]
        for key in ("model_path", "iteration", "checkpoint_stats"):
            require(scene.get(key) == original.get(key), f"{name}: source {key} changed from A/B")
        frames = original["merged_render_frames"]
        require(len(frames) == len(scene["frames"]), f"{name}: original trajectory count changed")
        for actual, old in zip(scene["frames"], frames):
            for key in fields:
                require(key in actual and actual[key] == old[key],
                        f"{name} frame {old['frame_index']}: original camera/image field {key} changed")
    return {"status": "pass", "scenes": len(manifest["scenes"]),
            "frames": sum(len(s["frames"]) for s in manifest["scenes"]),
            "evidence": "Actual experiment enumeration compared with complete historical A/B inventory; no extra source scan."}


def audit_scene_completion(root, manifest):
    statuses = {}
    for scene in manifest["scenes"]:
        name = scene["scene"]
        path = root / "scenes" / name / "status.json"
        status = json.loads(path.read_text())
        original = json.loads((path.parent / "manifest.json").read_text())
        require(original.get("scope") == "full_scene" and original.get("run_id") == manifest["run_id"]
                and original.get("scenes") == [scene], f"{name}: merged manifest changed original scene definition")
        for key in ("cpu_threads", "torch_threads", "thread_environment", "query_device",
                    "cache_enabled", "precompute_enabled", "scheduling_enabled", "quality"):
            require(original.get("settings", {}).get(key) == manifest["settings"].get(key),
                    f"{name}: original runtime setting {key} differs from merged comparison")
        for key in ("hostname", "cpu_model", "gpu_uuid", "gpu_name"):
            require(original.get("machine", {}).get(key) == manifest["machine"].get(key),
                    f"{name}: original machine/device {key} differs from merged comparison")
        require(status.get("status") == "complete", f"{name}: actual experiment did not finish successfully")
        require(status.get("checkpoint_stats_unchanged") is True,
                f"{name}: source checkpoint preservation did not pass")
        require(status.get("scope") == "full_scene" and status.get("run_id") == manifest["run_id"]
                and status.get("scene") == name, f"{name}: terminal status identity/scope changed")
        require(all(status.get(k) == len(scene["frames"])
                    for k in ("completed_frames", "expected_frames", "full_trajectory_frames")),
                f"{name}: terminal status has incomplete frame counts")
        statuses[name] = status
    return statuses


def audit_gpu_strategy(scene, frame, row, counts, key):
    """Check actual GPU producers, source masks, and necessary transfer scope."""
    mode = row["mode"]
    require(counts.get("query_device") == "gpu", f"{key}: actual query device is not GPU")
    for field, expected in (("scene_token", scene["scene_token"]), ("index_token", scene["index_token"]),
                            ("mesh_query_index_token", scene["mesh_query_index_token"]),
                            ("camera_token", frame["camera_token"]), ("mesh_token", scene["mesh_token"])):
        require(counts.get(field) == expected, f"{key}: actual pipeline source {field} differs")
    transfers = counts.get("payload_transfer_bytes", {})
    require(all(type(transfers.get(k)) is int and transfers[k] == 0 for k in PAYLOAD_TRANSFERS),
            f"{key}: a full GPU query payload made a CPU round trip or is unrecorded")
    require(counts.get("decoded") == counts["selected"], f"{key}: actual decoder count differs from selection")
    mesh, ori, anchor = (counts.get(k, {}) for k in ("mesh", "ori", "anchor"))
    device = mesh.get("actual_device", "")
    require(isinstance(device, str) and device.startswith("cuda:")
            and ori.get("actual_device") == device and ori.get("triangle_query_device") == device
            and anchor.get("actual_device") == device and anchor.get("query_device") == "gpu",
            f"{key}: mesh, ORI and anchor queries do not share the same CUDA device")
    expected_mesh = "gpu_parallel_triangle_scan" if mode == "R1" else "gpu_bvh_leaf_cluster_scan"
    require(mesh.get("index_structure") == expected_mesh
            and mesh.get("backend") == ("brute_force" if mode == "R1" else "bvh")
            and mesh.get("visited_nodes") == 0,
            f"{key}: incorrect GPU mesh strategy or false root-traversal counter")
    require(mesh.get("mesh_token") == scene["mesh_token"]
            and mesh.get("index_token") == scene["mesh_query_index_token"]
            and mesh.get("camera_id") == frame["camera_token"], f"{key}: mesh query source identity differs")
    require(mesh.get("mesh_triangles") == scene["triangles"]
            and mesh.get("returned_triangles") == counts["triangles"]
            and type(mesh.get("tested_triangles")) is int
            and counts["triangles"] <= mesh["tested_triangles"] <= scene["triangles"],
            f"{key}: mesh predicate counts contradict actual triangle table")
    require(mesh.get("triangle_table_identity") == {
                "mesh_token": scene["mesh_token"], "vertices": scene["vertices"],
                "triangles": scene["triangles"], "face_id_policy": "stable_triangle_table_row"},
            f"{key}: actual triangle table identity differs")
    if mode == "R1":
        require(mesh["tested_triangles"] == scene["triangles"]
                and mesh.get("tested_leaf_clusters") == 0,
                f"{key}: R1 is not the complete GPU triangle scan")
    else:
        require(type(mesh.get("tested_leaf_clusters")) is int and type(mesh.get("active_leaf_clusters")) is int
                and 0 <= mesh["active_leaf_clusters"] <= mesh["tested_leaf_clusters"] <= scene["triangles"],
                f"{key}: invalid actual leaf-cluster work counters")
    require(mesh.get("triangle_id_download_bytes") == 0
            and ori.get("triangle_id_upload_bytes") == 0 and ori.get("download_bytes") == 0,
            f"{key}: GPU triangle IDs or tile depths were copied through CPU")
    require(type(mesh.get("camera_upload_bytes")) is int and mesh["camera_upload_bytes"] > 0
            and type(ori.get("upload_bytes")) is int and ori["upload_bytes"] >= 160,
            f"{key}: necessary small camera uploads missing")
    require(ori.get("mesh_token") == scene["mesh_token"] and ori.get("camera_id") == frame["camera_token"]
            and ori.get("selected_triangles") == counts["triangles"]
            and ori.get("source_query_index_token") == scene["mesh_query_index_token"]
            and ori.get("source_triangle_id_policy") == "sorted_unique_global_triangle_rows",
            f"{key}: ORI source mask is not the actual mesh query")
    tile = scene["settings"]["tile_size"]
    original_pixels = math.prod(frame["render_dimensions"])
    padded_pixels = math.prod(ori["padded_image_size"])
    cells = padded_pixels // (tile * tile)
    require(all(type(ori.get(k)) is int and ori[k] >= 0 for k in
                ("covered_pixels", "unknown_pixels", "covered_cells", "unknown_cells", "invalid_raster_hits"))
            and ori["covered_pixels"] <= original_pixels
            and ori["covered_pixels"] + ori["unknown_pixels"] == padded_pixels
            and ori["covered_cells"] + ori["unknown_cells"] == cells,
            f"{key}: actual ROI/tile coverage counts contradict native image dimensions")
    profile, definition, _ = SUPPORT_CONTRACTS[scene["settings"]["support_profile"]]
    require(anchor.get("support_profile") == profile and anchor.get("support_definition") == definition,
            f"{key}: actual anchor proxy/support definition differs from manifest")
    require(counts.get("anchor_support_profile") == profile and counts.get("anchor_support_definition") == definition,
            f"{key}: pipeline support/profile label differs from actual anchor producer")
    for field, expected in (("scene_token", scene["scene_token"]), ("index_token", scene["index_token"]),
                            ("camera_token", frame["camera_token"]), ("mesh_token", scene["mesh_token"])):
        require(anchor.get(field) == expected, f"{key}: anchor source {field} differs")
    expected_anchor = "parallel_per_fov_anchor" if mode == "R1" else "batch_all_nonempty_nodes_then_uncertified_anchors"
    require(anchor.get("strategy") == expected_anchor
            and anchor.get("range_definition") == "canonical_selected_DFS_runs",
            f"{key}: incorrect GPU anchor/range strategy")
    require(anchor.get("fov_count") == counts["fov"] and anchor.get("selected_count") == counts["selected"]
            and anchor.get("certified_anchors") == counts["culled"],
            f"{key}: anchor work counts disagree with returned selection")
    require(anchor.get("visited_nodes") == (0 if mode == "R1" else scene["anchor_nodes"]),
            f"{key}: batched GPU node count must not be presented as early-exit DFS")
    require(type(anchor.get("rectangle_queries")) is int and anchor["rectangle_queries"] >= 0
            and anchor.get("sparse_table_rectangle_reads") == 4 * anchor["rectangle_queries"],
            f"{key}: rectangle-max work is unrecorded or inconsistent")
    require(all(type(anchor.get(k)) is int and anchor[k] == 0 for k in (
                "fov_id_upload_bytes", "fov_id_download_bytes", "selected_id_upload_bytes",
                "selected_id_download_bytes", "range_upload_bytes", "range_download_bytes",
                "ori_payload_upload_bytes", "ori_payload_download_bytes")),
            f"{key}: anchor/range payload left its CUDA query device")


def audit_timing_rows(manifest, rows, *, counts_by_frame=None):
    scenes = {s["scene"]: s for s in manifest["scenes"]}
    expected = {(name, f["frame_index"], mode, repeat)
                for name, scene in scenes.items() for f in scene["frames"]
                for mode in MODES for repeat in range(manifest["timing_repeats"])}
    observed, grouped = set(), defaultdict(lambda: defaultdict(list))
    sequence = defaultdict(list)
    query_devices = defaultdict(set)
    for row in rows:
        key = (row["scene"], row["frame_index"], row["mode"], row["repeat"])
        require(key in expected and key not in observed, f"Unexpected/duplicate timing row {key}")
        observed.add(key)
        scene = scenes[row["scene"]]
        frame = scene["frames"][row["frame_index"]]
        require(row.get("status") == "complete" and row.get("run_id") == manifest["run_id"],
                f"Incomplete/mismatched actual frame {key}")
        require(row.get("camera_token") == frame["camera_token"]
                and row.get("settings_token") == scene["settings_token"],
                f"Unpaired camera/settings {key}")
        count_names = ("selected",) if row["mode"] == "R0" else ("fov", "selected", "culled", "triangles")
        counts = row.get("counts", {})
        require(all(type(counts.get(k)) is int and counts[k] >= 0 for k in count_names),
                f"Missing/invalid actual execution counts {key}")
        if row["mode"] != "R0":
            require(counts["selected"] + counts["culled"] == counts["fov"],
                    f"Inconsistent actual execution counts {key}")
            ori = counts.get("ori", {})
            tile_size = scene["settings"]["tile_size"]
            multiple = math.lcm(tile_size, 8)
            original_size = frame["render_dimensions"]
            padded_size = [math.ceil(size / multiple) * multiple for size in original_size]
            require(ori.get("ori_definition") == ORI_DEFINITION
                    and ori.get("continuous_coverage_certificate") is False,
                    f"{key}: missing discrete ORI definition or false continuous guarantee")
            require(ori.get("tile_size") == tile_size and ori.get("original_image_size") == original_size
                    and ori.get("padded_image_size") == padded_size,
                    f"{key}: ORI dimensions/tile size differ from the native frame")
            if scene["settings"].get("query_device", "cpu") == "gpu":
                audit_gpu_strategy(scene, frame, row, counts, key)
                query_devices[row["scene"]].add(counts["mesh"]["actual_device"])
            else:
                require(type(ori.get("upload_bytes")) is int and ori["upload_bytes"] >= counts["triangles"] * 8 + 160
                        and type(ori.get("download_bytes")) is int
                        and ori["download_bytes"] >= (padded_size[0] // tile_size) * (padded_size[1] // tile_size) * 8 + 24,
                        f"{key}: necessary ORI upload/download evidence missing")
        if counts_by_frame is not None:
            counts_by_frame[key] = {k: counts[k] for k in count_names}
        timings = row.get("timings_ms", {})
        measured = TIMINGS if row["mode"] != "R0" else ("total",)
        require(all(k in timings and isinstance(timings[k], (int, float))
                    and math.isfinite(timings[k]) and timings[k] >= 0 for k in measured),
                f"Missing/nonfinite/negative timing {key}")
        require(timings["total"] > 0, f"Invalid enclosing timing {key}")
        for component in ("query", "decode_raster"):
            if timings.get(component) is not None:
                require(timings["total"] >= timings[component], f"Invalid enclosing timing {key}")
        if row["mode"] != "R0":
            require(timings["query"] > 0, f"Unmeasured query {key}")
            query_sum = math.fsum(timings[k] for k in ("fov", "d2h", "retrieval", "ori", "anchor", "h2d"))
            require(math.isclose(timings["query"], query_sum, rel_tol=1e-9, abs_tol=1e-6),
                    f"Impossible query/component timing sum {key}")
            require(math.isclose(timings["total"], timings["query"] + timings["decode_raster"],
                                 rel_tol=1e-9, abs_tol=1e-6), f"Impossible total/component timing sum {key}")
        for timing, value in timings.items():
            if value is not None:
                require(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0,
                        f"Invalid measured component {key}:{timing}")
                grouped[(row["scene"], row["mode"])][timing].append(value)
        sequence[(row["scene"], row["frame_index"], row["repeat"])].append(row["mode"])
    missing = expected - observed
    require(not missing, f"Missing {len(missing)} formal timing rows; first={sorted(missing)[:3]}")
    for key, order in sequence.items():
        require(order == manifest["timing_mode_order"][key[2]], f"Timing mode order changed {key}")
    require(all(len(devices) == 1 for devices in query_devices.values()),
            "R1/R2 or frame repetitions used different CUDA query devices")
    return grouped


def summarize_timings(grouped):
    report = {"scenes": {}, "combined": {}}
    combined = defaultdict(lambda: defaultdict(list))
    for (scene, mode), timings in grouped.items():
        destination = report["scenes"].setdefault(scene, {}).setdefault(mode, {})
        for name, values in timings.items():
            values = np.asarray(values, dtype=np.float64)
            destination[name] = {"sum_ms": float(values.sum()), "p50_ms": float(np.percentile(values, 50)),
                                 "p95_ms": float(np.percentile(values, 95)), "max_ms": float(values.max()),
                                 "measurements": len(values)}
            combined[mode][name].extend(values.tolist())

    def ratios(values):
        def ratio(first, second):
            return first / second if second > 0 else None
        return {"r1_query_over_r2_query": ratio(values["R1"]["query"], values["R2"]["query"]),
                "r1_total_over_r2_total": ratio(values["R1"]["total"], values["R2"]["total"]),
                "r0_total_over_r2_total": ratio(values["R0"]["total"], values["R2"]["total"]),
                "scan_retrieval_over_bvh_retrieval": ratio(values["R1"]["retrieval"], values["R2"]["retrieval"])}
    for scene, value in report["scenes"].items():
        value["ratios"] = ratios({mode: {k: v["sum_ms"] for k, v in value[mode].items()} for mode in MODES})
    sums = {mode: {name: float(np.sum(values, dtype=np.float64)) for name, values in timings.items()}
            for mode, timings in combined.items()}
    report["combined"] = {"sums_ms": sums, "ratios": ratios(sums)}
    report["scope"] = "Ratios of all paired elapsed sums; values below 1 are slowdowns. Offline work excluded."
    return report


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "infinity" if value > 0 else "-infinity" if value < 0 else "nan"
    return value


def _initialize_audit_worker():
    """Spawned workers import NumPy under inherited one-thread environment."""
    for name in WORKER_THREAD_ENV:
        os.environ[name] = "1"
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return
    global _audit_thread_limit
    _audit_thread_limit = threadpool_limits(limits=1)


@contextmanager
def _worker_environment():
    previous = {name: os.environ.get(name) for name in WORKER_THREAD_ENV}
    for name in WORKER_THREAD_ENV:
        os.environ[name] = "1"
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _audit_scene_worker(job):
    """Read one complete scene, one raw frame at a time, in unchanged order."""
    root, output = Path(job["root"]), Path(job["output"])
    scene, manifest, counts_by_frame = job["scene"], job["context"], job["counts"]
    name = scene["scene"]
    status = audit_scene_completion(root, {**manifest, "scenes": [scene]})[name]
    permutation = np.load(resolve_artifact(root, scene["dfs_to_row_path"]), allow_pickle=False)
    validate_permutation(permutation)
    permutation.flags.writeable = False
    scene_summary = {"frames": 0, "image_frames": 0, "r0_repeat_failed_frames": [], "r1_failed_frames": [], "r2_failed_frames": [],
                     "worst_psnr_drop": None, "worst_ssim_drop": None,
                     "maximum_raw_absolute_difference": 0.0, "r2_removed_anchor_observations": 0}
    errors = []
    partial = output.with_suffix(output.suffix + ".partial")
    with partial.open("w") as stream:
        for frame in scene["frames"]:
            index = frame["frame_index"]
            path = root / "quality" / name / f"{index:06d}.npz"
            evidence = counts = None
            try:
                with np.load(path, allow_pickle=False) as arrays:
                    for field, expected in (("run_id", manifest["run_id"]), ("scene", name),
                                            ("frame_index", index), ("camera_token", frame["camera_token"]),
                                            ("mesh_token", scene["mesh_token"])):
                        require(np.asarray(arrays[field]).shape == () and arrays[field].item() == expected,
                                f"Raw image bundle {field} does not match actual frame")
                    evidence = audit_images(arrays, frame["render_dimensions"])
                    scene_summary["image_frames"] += 1
                    scene_summary["maximum_raw_absolute_difference"] = max(
                        scene_summary["maximum_raw_absolute_difference"], evidence["direct"]["r2"]["max_abs"])
                    for mode in ("r0_repeat", "r1", "r2"):
                        if not evidence["quality"][mode]["pass"]:
                            scene_summary[mode + "_failed_frames"].append(index)
                    for metric, field in (("psnr_drop_db", "worst_psnr_drop"), ("ssim_drop", "worst_ssim_drop")):
                        value = evidence["quality"]["r2"][metric]
                        if scene_summary[field] is None or value > scene_summary[field]["value"]:
                            scene_summary[field] = {"frame_index": index, "value": value}
                    counts = audit_selection(arrays, permutation, permutation_verified=True)
                for mode in MODES:
                    expected_counts = {"selected": counts["fov"]} if mode == "R0" else {
                        "fov": counts["fov"], "selected": counts[mode.lower() + "_selected"],
                        "culled": counts["fov"] - counts[mode.lower() + "_selected"],
                        "triangles": counts["triangles"]}
                    for repeat in range(manifest["timing_repeats"]):
                        require(counts_by_frame[(index, mode, repeat)] == expected_counts,
                                f"{mode} repeat {repeat}: formal execution counts differ from actual quality arrays")
                row = {"scene": name, "frame_index": index, "camera_token": frame["camera_token"],
                       "counts": counts, **evidence}
                scene_summary["frames"] += 1
                scene_summary["r2_removed_anchor_observations"] += counts["fov"] - counts["r2_selected"]
            except (ValueError, KeyError, OSError) as exc:
                row = {"scene": name, "frame_index": index, "status": "failure", "error": str(exc)}
                if evidence is not None:
                    row.update(evidence)
                if counts is not None:
                    row["counts"] = counts
                errors.append(row)
            stream.write(json.dumps(json_safe(row), allow_nan=False) + "\n")
            stream.flush()
    partial.replace(output)
    return {"scene": name, "summary": scene_summary, "errors": errors, "actual_status": status,
            "output": str(output), "worker_pid": os.getpid(),
            "thread_environment": {name: os.environ.get(name) for name in WORKER_THREAD_ENV}}


def audit_run(root, output, *, audit_workers=4):
    require(type(audit_workers) is int and 1 <= audit_workers <= 8, "audit_workers must be an integer from 1 to 8")
    root, output = Path(root), Path(output)
    manifest = json.loads((root / "manifest.json").read_text())
    scenes = audit_manifest(manifest)
    reference = json.loads(resolve_artifact(root, manifest["reference_survey_path"]).read_text())
    inventory = audit_inventory_against_ab(manifest, reference)
    counts_by_frame = {}
    with (root / "records.jsonl").open() as handle:
        grouped = audit_timing_rows(manifest, (json.loads(line) for line in handle if line.strip()),
                                    counts_by_frame=counts_by_frame)
    output.mkdir(parents=True, exist_ok=True)
    scene_output = output / "per_scene"
    scene_output.mkdir(exist_ok=True)
    summary = {"run_id": manifest["run_id"], "status": "incomplete", "errors": [],
               "quality_settings": QUALITY_SETTINGS, "input_inventory": inventory,
               "actual_scene_statuses": {}, "timing": summarize_timings(grouped), "scenes": {}}
    context = {key: manifest[key] for key in ("run_id", "settings", "machine", "timing_repeats")}
    jobs = [{"root": str(root), "output": str(scene_output / (scene["scene"] + ".jsonl")),
             "scene": scene, "context": context,
             "counts": {(frame, mode, repeat): value for (name, frame, mode, repeat), value in counts_by_frame.items()
                        if name == scene["scene"]}} for scene in scenes]
    worker_count = min(audit_workers, len(jobs))
    with _worker_environment():
        if audit_workers == 1:
            try:
                from threadpoolctl import threadpool_limits
                thread_limit = threadpool_limits(limits=1)
            except ImportError:
                thread_limit = nullcontext()
            with thread_limit:
                results = [_audit_scene_worker(job) for job in jobs]
        else:
            # Spawn avoids inheriting initialized multi-thread BLAS state.
            with ProcessPoolExecutor(max_workers=worker_count, mp_context=multiprocessing.get_context("spawn"),
                                     initializer=_initialize_audit_worker) as executor:
                results = list(executor.map(_audit_scene_worker, jobs))
    with (output / "independent_per_frame.jsonl").open("w") as stream:
        for result in results:
            name = result["scene"]
            summary["scenes"][name] = result["summary"]
            summary["errors"].extend(result["errors"])
            summary["actual_scene_statuses"][name] = result["actual_status"]
            with Path(result["output"]).open() as source:
                shutil.copyfileobj(source, stream)
    summary["audit_execution"] = {"requested_workers": audit_workers, "actual_worker_limit": worker_count,
                                  "scene_order": [scene["scene"] for scene in scenes],
                                  "workers": [{key: result[key] for key in ("scene", "worker_pid", "thread_environment")}
                                              for result in results],
                                  "scope": "CPU raw-image/ID audit; each worker streams complete scenes with one BLAS/OpenMP thread"}
    quality_failed = any(s["r0_repeat_failed_frames"] or s["r1_failed_frames"] or s["r2_failed_frames"]
                         for s in summary["scenes"].values())
    summary["status"] = "pass" if not summary["errors"] and not quality_failed else "fail"
    summary["scope"] = ("Complete raw-frame image/selection/timing audit. Independent geometry tests, "
                        "mesh adoption, actual decoder instrumentation and offline artifact review remain separate gates.")
    (output / "independent_summary.json").write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-workers", type=int, choices=range(1, 9), default=4,
                        help="CPU scene workers, each single-threaded; default 4, maximum 8")
    args = parser.parse_args(argv)
    try:
        report = audit_run(args.run_root, args.output, audit_workers=args.audit_workers)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        report = {"status": "fail", "error": str(exc)}
        (args.output / "independent_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(json_safe(report), allow_nan=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
