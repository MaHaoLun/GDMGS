"""Build the independent GPU anchor extension in an explicit artifact directory."""
import argparse
import os
from pathlib import Path


def build(directory, verbose=False):
    from torch.utils.cpp_extension import load
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    root = Path(__file__).parent
    return load(name="_gdmgs_anchor_cuda", sources=[str(root / "bindings.cpp"), str(root / "anchor_cuda.cu")],
                extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "--fmad=false"],
                build_directory=str(directory), verbose=verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    print(build(args.output_dir, args.verbose).__file__)
