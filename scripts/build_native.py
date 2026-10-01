"""Build the actual native CPU library; optionally compile CUDA kernels."""
import argparse
import importlib.util
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def build_cpu(output):
    output = Path(output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    args = [os.environ.get('CXX', 'c++'), '-O3', '-std=c++17', '-shared', '-fPIC', '-ffp-contract=off']
    if platform.system() == 'Darwin':
        omp = Path(os.environ.get('LIBOMP_PREFIX', '/opt/homebrew/opt/libomp'))
        if not (omp/'include/omp.h').exists():
            raise RuntimeError('OpenMP missing; set LIBOMP_PREFIX to an installed libomp prefix')
        # Reuse Torch's OpenMP runtime when present; loading two libomp copies aborts.
        torch_spec = importlib.util.find_spec('torch')
        torch_omp = Path(torch_spec.origin).parent/'lib/libomp.dylib' if torch_spec else None
        library = torch_omp if torch_omp is not None and torch_omp.exists() else omp/'lib/libomp.dylib'
        args += ['-Xpreprocessor', '-fopenmp', '-I'+str(omp/'include'), str(library),
                 '-Wl,-rpath,'+str(library.parent)]
    else:
        args += ['-fopenmp']
    args += [str(ROOT/'system/cpu_select.cpp'), str(ROOT/'system/hole_planes_native.cpp'), '-o', str(output)]
    subprocess.run(args, check=True)
    if platform.system() == 'Darwin':
        # Some Torch wheels retain the build-machine install name in libomp.
        install_name = subprocess.check_output(['otool', '-D', str(library)], text=True).splitlines()[1].strip()
        subprocess.run(['install_name_tool', '-change', install_name, str(library), str(output)], check=True)
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'build/cpu_select.so')
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    print(build_cpu(args.output))
    if args.cuda:
        sys.path.insert(0, str(ROOT))
        from system.bootstrap import configure
        configure()
        from gdmgs.anchor_frustum.gpu_construction import native
        from fused_filter import native as holes
        from staged_isect import load_extension
        native(); holes(); load_extension()
        print('CUDA anchor, hole-filter and staged-render extensions built')
