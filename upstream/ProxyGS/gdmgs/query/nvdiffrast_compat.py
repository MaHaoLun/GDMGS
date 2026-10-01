"""Build the Step 8 nvdiffrast plugin against Torch's CUDA 12.4 runtime.

The shared environment's generic ``include`` directory currently contains
CUDA 12.9 headers, while Torch 2.4.0 and its pip CUDA runtime are 12.4.  The
upstream JIT build adds that generic directory automatically, which changes
the size of CUDA runtime structs and trips the stack protector in
``RasterizeCudaContext``.  This separately named plugin puts the matching pip
12.4 headers first and leaves all historical extension directories untouched.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


def ensure_step8_nvdiffrast_plugin():
    import nvdiffrast.torch.ops as ops

    if getattr(ops, "_cached_plugin", {}).get(False) is not None:
        return ops._cached_plugin[False]

    site_packages = Path(torch.__file__).resolve().parents[1]
    nvidia_packages = site_packages / "nvidia"
    cuda_package = nvidia_packages / "cuda_runtime"
    headers = cuda_package / "include"
    runtime = cuda_package / "lib"
    version_header = headers / "cuda_runtime_api.h"
    if not version_header.is_file() or not runtime.is_dir():
        raise RuntimeError("Torch CUDA runtime headers/libraries are unavailable")
    if "#define CUDART_VERSION  12040" not in version_header.read_text():
        raise RuntimeError("Step 8 requires the Torch CUDA 12.4 runtime headers")

    torch_dir = Path(ops.__file__).resolve().parent
    source_files = [
        "../common/cudaraster/impl/Buffer.cpp",
        "../common/cudaraster/impl/CudaRaster.cpp",
        "../common/cudaraster/impl/RasterImpl.cu",
        "../common/cudaraster/impl/RasterImpl.cpp",
        "../common/common.cpp",
        "../common/rasterize.cu",
        "../common/interpolate.cu",
        "../common/texture.cu",
        "../common/texture.cpp",
        "../common/antialias.cu",
        "torch_bindings.cpp",
        "torch_rasterize.cpp",
        "torch_interpolate.cpp",
        "torch_texture.cpp",
        "torch_antialias.cpp",
    ]
    sources = [str((torch_dir / name).resolve()) for name in source_files]
    # nvdiffrast clears this variable before its own build.  The Step 8 build
    # is deliberately explicit because the formal hardware is A100.
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
    include_paths = [
        headers,
        nvidia_packages / "cusparse" / "include",
        nvidia_packages / "cublas" / "include",
        nvidia_packages / "cusolver" / "include",
        # The pip CUDA runtime does not ship libcu++'s nv/target.  CUDA 12.3's
        # copy is ABI-compatible with the 12.4 headers and avoids accidentally
        # picking the host's CUDA 13 copy from /usr/include.
        Path("/usr/local/cuda-12.3/include"),
    ]
    if not all(path.is_dir() for path in include_paths):
        raise RuntimeError("matching CUDA include directories are unavailable")
    plugin = torch.utils.cpp_extension.load(
        name="nvdiffrast_plugin_step8_cu124_v2",
        sources=sources,
        extra_include_paths=[str(path) for path in include_paths],
        extra_cflags=["-DNVDR_TORCH"],
        extra_cuda_cflags=["-DNVDR_TORCH", "-lineinfo"],
        extra_ldflags=[f"-L{runtime}", f"-Wl,-rpath,{runtime}"],
        with_cuda=True,
        verbose=False,
    )
    ops._cached_plugin[False] = plugin
    return plugin
