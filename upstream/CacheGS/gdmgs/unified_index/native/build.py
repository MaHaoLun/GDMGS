"""Build the independent CPU anchor extension; no torch/CUDA or content hashes."""
from pathlib import Path
import argparse
import os
import subprocess
import sysconfig


def build(output_dir):
    import pybind11
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / ("_gdmgs_anchor_native" + sysconfig.get_config_var("EXT_SUFFIX"))
    temporary = output.with_name("." + output.name + ".building-" + str(os.getpid()))
    command = [os.environ.get("CXX", "c++"), "-O3", "-std=c++17", "-shared", "-fPIC",
               "-fvisibility=hidden", "-I" + pybind11.get_include(), "-I" + sysconfig.get_path("include"),
               str(Path(__file__).with_name("anchor_native.cpp")), "-o", str(temporary)]
    if sysconfig.get_platform().startswith("macosx"):
        command.extend(["-undefined", "dynamic_lookup"])
    try:
        subprocess.run(command, check=True)
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=os.environ.get("GDMGS_NATIVE_DIR"))
    args = parser.parse_args()
    if not args.output_dir:
        parser.error("--output-dir or GDMGS_NATIVE_DIR is required")
    print(build(args.output_dir))
