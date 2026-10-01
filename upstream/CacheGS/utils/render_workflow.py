"""Shared inference loading, camera policies and frame export for legacy CLIs.

Heavy renderer dependencies are loaded only when rendering is requested.
"""

import importlib
import json
from pathlib import Path
import time

from utils.runtime_options import (isolated_output_path, output_root_for,
                                   output_subdirectory, validate_pipeline_options)


def load_model_config(model_path):
    import yaml
    from utils.general_utils import parse_cfg

    with open(Path(model_path) / "config.yaml") as handle:
        dataset, opt, pipe = parse_cfg(yaml.load(handle, Loader=yaml.FullLoader))
    dataset.model_path = str(model_path)
    return dataset, opt, pipe


def create_model(dataset):
    modules = importlib.import_module("scene.gs_model_" + dataset.base_model)
    config = dataset.model_config
    return getattr(modules, config["name"])(**config["kwargs"])


def dataset_layout(dataset):
    source = Path(dataset.source_path)
    if (source / "transforms.json").exists():
        return "city"
    if (source / "sparse" / "0").exists():
        return "colmap"
    return "other"


def resolve_checkpoint_iteration(model_path, iteration):
    """Never let Scene's falsy load_iteration branch initialize/write a model."""
    point_cloud = Path(model_path) / "point_cloud"
    if iteration == -1:
        candidates = [int(p.name.split("_")[-1]) for p in point_cloud.iterdir()
                      if p.is_dir() and p.name.startswith("iteration_")
                      and p.name.split("_")[-1].isdigit()]
        if not candidates:
            raise FileNotFoundError("No saved checkpoint iterations in {}".format(point_cloud))
        iteration = max(candidates)
    if iteration <= 0:
        raise ValueError("Inference requires a positive saved iteration or -1 for latest")
    ply = point_cloud / ("iteration_{}".format(iteration)) / "point_cloud.ply"
    if not ply.is_file():
        raise FileNotFoundError(str(ply))
    return iteration


def load_scene(dataset, gaussians, iteration, camera_policy="merged", scene_factory=None):
    if camera_policy not in ("merged", "split"):
        raise ValueError("Unknown camera policy: " + camera_policy)
    iteration = resolve_checkpoint_iteration(dataset.model_path, iteration)
    if scene_factory is None:
        from scene import Scene
        scene_factory = Scene
    force_city = camera_policy == "merged" and dataset_layout(dataset) == "city"
    original_eval = getattr(dataset, "eval", False)
    if force_city:
        dataset.eval = False
    try:
        return scene_factory(dataset, gaussians, load_iteration=iteration, shuffle=False,
                             resolution_scales=dataset.resolution_scales)
    finally:
        if force_city:
            dataset.eval = original_eval


def enumerate_cameras(dataset, scene, camera_policy="merged", skip_train=False, skip_test=False):
    """Do not renumber camera UIDs; match the two original CLI policies."""
    if camera_policy == "split":
        groups = []
        if not skip_train:
            groups.append(("train", scene.getTrainCameras()))
        if not skip_test:
            groups.append(("test", scene.getTestCameras()))
        return groups
    if camera_policy != "merged":
        raise ValueError("Unknown camera policy: " + camera_policy)
    layout = dataset_layout(dataset)
    if layout == "city":
        return scene.getTrainCameras()
    if layout == "colmap":
        if not getattr(dataset, "eval", False):
            return scene.getTrainCameras()
        cameras = list(scene.getTrainCameras()) + list(scene.getTestCameras())
        try:
            cameras.sort(key=lambda camera: camera.image_name)
        except (AttributeError, TypeError):
            cameras.sort(key=lambda camera: camera.uid)
        return cameras
    raise NotImplementedError("Unsupported merged trajectory: expected MatrixCity or COLMAP structure")


def load_inference_scene(dataset, opt, iteration, camera_policy="merged"):
    model = create_model(dataset)
    scene = load_scene(dataset, model, iteration, camera_policy)
    model.eval()
    if hasattr(model, "set_coarse_interval"):
        model.set_coarse_interval(opt)
    return model, scene


def reset_scene_render_context(pipeline="legacy"):
    """Reset once per scene, preserving render2's train-to-test cursor ordering."""
    if pipeline == "legacy":
        renderer = importlib.import_module("gaussian_renderer.render")
        renderer.reset_render_context()


class FrameRenderer:
    """Callable single-frame adapter that owns any fresh inference session."""
    def __init__(self, render_frame, close=None):
        self.render_frame = render_frame
        self._close = close

    def __call__(self, camera):
        return self.render_frame(camera)

    def close(self):
        if self._close is not None:
            close, self._close = self._close, None
            close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def render_artifact_path(model_path, output_root, name, iteration, *parts):
    root = output_root_for(model_path, output_root)
    relative = str(Path(name) / "ours_{}".format(iteration) / Path(*parts))
    output = output_subdirectory(root, relative)
    if output_root is not None:
        output = isolated_output_path(model_path, output)
    return Path(output)


def make_frame_renderer(base_model, gaussians, pipe, background, iteration,
                        render_mode, ape_code=-1, enable_cache=False,
                        pipeline="legacy", checkpoint_path=None):
    validate_pipeline_options(pipeline, enable_cache)
    if pipeline == "gdmgs":
        from gdmgs.pipeline import FreshPipeline
        session = FreshPipeline(gaussians, checkpoint_path, iteration)
        return FrameRenderer(
            lambda camera: session.render(camera, pipe, background, render_mode, ape_code=ape_code),
            close=session.close,
        )
    from utils.general_utils import get_render_func
    modules = importlib.import_module("gaussian_renderer")
    render_func = getattr(modules, get_render_func(base_model))
    kwargs = {"disable_cache": not enable_cache} if render_func.__name__ == "render" else {}
    def render_frame(camera):
        args = [camera, gaussians, pipe, background, iteration, render_mode]
        if ape_code != -1:
            args.append(ape_code)
        return render_func(*args, **kwargs)
    return FrameRenderer(render_frame)


def render_set(base_model, model_path, name, iteration, views, gaussians, pipe,
               background, render_mode, ape_code, enable_cache=False,
               output_root=None, pipeline="legacy"):
    import numpy as np
    import torch
    import torchvision
    from tqdm import tqdm

    def artifact(*parts):
        return render_artifact_path(model_path, output_root, name, iteration, *parts)

    artifact()
    render_path, gt_path = artifact("renders"), artifact("gt")
    with make_frame_renderer(base_model, gaussians, pipe, background,
                             iteration, render_mode, ape_code, enable_cache,
                             pipeline, model_path) as frame_renderer:
        render_path.mkdir(parents=True, exist_ok=True)
        gt_path.mkdir(parents=True, exist_ok=True)
        times, per_view, cache_records = [], {}, []
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad():
                package = frame_renderer(view)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            times.append(elapsed)
            visible_count = int(package["visibility_filter"].sum().item())
            image_name = "{:05d}.png".format(idx)
            visibility = package.get("visibility_summary")
            cache_stats = package.get("cache_stats")
            stats = {"visible_gaussians": visible_count,
                     "frame_time_ms": round(elapsed * 1000.0, 4),
                     "cache_enabled": bool(enable_cache), "total": visible_count}
            if visibility:
                stats.update(total=int(visibility.get("total", visible_count)),
                             levels=visibility.get("levels", {}),
                             sample_voxels=visibility.get("samples", []))
            if enable_cache:
                payload = cache_stats or {}
                stats["cache"] = {"hit_rate": payload.get("cache_hit_rate"),
                                  "size": payload.get("cache_size"),
                                  "per_level_summary": payload.get("per_level_summary", {}),
                                  "current_frame": payload.get("current_frame")}
                cache_records.append({"frame": idx, "image": image_name,
                                      "frame_time_ms": stats["frame_time_ms"],
                                      "cache_enabled": True, "cache": payload,
                                      "visibility": visibility})
            per_view[image_name] = stats
            torchvision.utils.save_image(torch.clamp(package["render"], 0.0, 1.0), artifact("renders", image_name))
            torchvision.utils.save_image(view.original_image[0:3, :, :], artifact("gt", image_name))
        # The old merged CLI produced NaN for <=5 frames. Match render2's warmup rule.
        measured = times[5:] if len(times) > 5 else times
        if measured:
            print("Test FPS: \033[1;35m{:.5f}\033[0m".format(1.0 / np.mean(measured)))
        else:
            print("No cameras in this split; no FPS measurement")
        with open(artifact("per_view_count.json"), "w") as handle:
            json.dump(per_view, handle, indent=2)
        if cache_records:
            with open(artifact("cache_stats.jsonl"), "w") as handle:
                for record in cache_records:
                    handle.write(json.dumps(record) + "\n")
