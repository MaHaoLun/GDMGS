"""CLI-only runtime choices; importing this module does not inspect or select GPUs."""

import os
from pathlib import Path


def validate_pipeline_options(pipeline="legacy", enable_cache=False, environ=None):
    if pipeline not in ("legacy", "gdmgs"):
        raise ValueError("pipeline must be legacy or gdmgs")
    env = os.environ if environ is None else environ
    if pipeline == "gdmgs":
        conflicts = []
        if enable_cache:
            conflicts.append("--enable_cache")
        if env.get("CACHE_ENABLE", "0").strip() not in ("", "0"):
            conflicts.append("CACHE_ENABLE")
        if env.get("PRECOMP_INDICES_PATH", "").strip():
            conflicts.append("PRECOMP_INDICES_PATH")
        if conflicts:
            raise ValueError("GDM-GS fresh rendering requires legacy cache/precompute disabled: " + ", ".join(conflicts))


def isolated_output_path(model_path, output_path):
    """Validate explicit artifact destinations, including symlink resolution."""
    model = Path(model_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if output == model or model in output.parents:
        raise ValueError("Explicit output must be outside the checkpoint directory: {}".format(model))
    return str(output)


def output_root_for(model_path, output_root=None):
    return str(model_path) if output_root is None else isolated_output_path(model_path, output_root)


def output_subdirectory(root, name):
    """Keep legacy relative output names while rejecting path escape."""
    root_path = Path(root).expanduser().resolve()
    output = (root_path / name).resolve()
    if output != root_path and root_path not in output.parents:
        raise ValueError("Output name must stay within the output root")
    return str(output)


def configure_device(device=None, allow_cpu=False):
    """Select a logical device within the caller's CUDA_VISIBLE_DEVICES mask."""
    import torch

    value = "cuda" if device is None else str(device)
    if value.isdigit():
        value = "cuda:" + value
    selected = torch.device(value)
    if selected.type == "cpu" and allow_cpu:
        return selected
    if selected.type != "cuda":
        raise ValueError("Rendering requires a CUDA device (for example --device cuda:0)")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this runtime")
    index = torch.cuda.current_device() if selected.index is None else selected.index
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError("CUDA device index is outside the visible device set")
    selected = torch.device("cuda", index)
    torch.cuda.set_device(selected)
    return selected


def initialize_runtime(quiet=False, device=None):
    # Preserve the old RNG/stdout setup. safe_state selects logical CUDA 0;
    # apply an explicit user selection afterwards without changing visibility.
    from utils.general_utils import safe_state

    safe_state(quiet)
    return configure_device(device)


def add_runtime_arguments(parser, pipeline=True):
    parser.add_argument("--output-root", default=None, help="Artifact root outside the model directory; default preserves the old location")
    parser.add_argument("--device", default=None, help="Logical CUDA device within CUDA_VISIBLE_DEVICES (for example cuda:0)")
    if pipeline:
        parser.add_argument("--pipeline", choices=("legacy", "gdmgs"), default="legacy", help="GDM-GS currently provides the fresh, explicit-selection stage-B path")
