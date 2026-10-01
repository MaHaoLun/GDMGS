"""Build the Step 5 GPU point-anchor extension into an explicit artifact directory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def build(directory: Path, *, verbose: bool = False):
    from torch.utils.cpp_extension import load

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    root = Path(__file__).resolve().parent
    return load(
        name="ProxyGS_gpu_anchor_point_native",
        sources=[str(root / "bindings.cpp"), str(root / "anchor_point_cuda.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--fmad=false"],
        build_directory=str(directory),
        verbose=verbose,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args()
    print(build(arguments.output_dir, verbose=arguments.verbose).__file__)
