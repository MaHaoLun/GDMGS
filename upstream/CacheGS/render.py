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
from argparse import ArgumentParser

from utils.render_workflow import (enumerate_cameras, load_inference_scene,
                                   load_model_config, render_set, reset_scene_render_context)
from utils.runtime_options import (add_runtime_arguments, initialize_runtime,
                                   output_root_for, validate_pipeline_options)


def render_sets(dataset, opt, pipe, iteration, ape_code, enable_cache=False,
                output_name="test", output_root=None, pipeline="legacy"):
    import torch

    validate_pipeline_options(pipeline, enable_cache)
    output_root_for(dataset.model_path, output_root)
    with torch.no_grad():
        reset_scene_render_context(pipeline)
        gaussians, scene = load_inference_scene(dataset, opt, iteration, "merged")
        cameras = enumerate_cameras(dataset, scene, "merged")
        if cameras:
            name = output_name or ("test" if getattr(dataset, "eval", False) else "train")
            render_set(dataset.base_model, dataset.model_path, name, scene.loaded_iter,
                       cameras, gaussians, pipe, scene.background, dataset.render_mode,
                       ape_code, enable_cache=enable_cache, output_root=output_root,
                       pipeline=pipeline)


def build_parser():
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--ape", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--enable_cache", action="store_true", help="Enable the legacy rendering cache (requires CACHE_ENABLE=1)")
    parser.add_argument("--output_name", default="test", help="Subdirectory under the artifact root")
    add_runtime_arguments(parser)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_pipeline_options(args.pipeline, args.enable_cache)
        output_root_for(args.model_path, args.output_root)
    except ValueError as exc:
        parser.error(str(exc))
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    device = initialize_runtime(args.quiet, args.device)
    dataset, opt, pipe = load_model_config(args.model_path)
    if args.device is not None:
        dataset.data_device = str(device)
    print("Rendering " + args.model_path)
    render_sets(dataset, opt, pipe, args.iteration, args.ape,
                enable_cache=args.enable_cache, output_name=args.output_name,
                output_root=args.output_root, pipeline=args.pipeline)


if __name__ == "__main__":
    main()
