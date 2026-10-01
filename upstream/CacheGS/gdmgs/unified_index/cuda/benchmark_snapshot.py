"""Frozen full-frame CPU/GPU anchor equivalence and synchronized query timing."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from gdmgs.unified_index import AnchorIndex, SupportSettings
from gdmgs.unified_index.gpu_index import GPUAnchorIndex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--quality", required=True)
    parser.add_argument("--ori", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    cpu = AnchorIndex.load(args.index)
    cpu.support_settings = SupportSettings()  # Same all-view world bounds, explicit v3 pixel pad.
    with np.load(args.quality, allow_pickle=False) as file:
        ids = file["fov_ids"].copy()
    with np.load(args.ori, allow_pickle=False) as file:
        depth = file["depth_bounds"].copy()
        w2c, k = file["w2c"].copy(), file["intrinsics"].copy()
    h, w = depth.shape[0] * 8, depth.shape[1] * 8
    domain = (-k[0, 2] / k[0, 0], (w-k[0, 2]) / k[0, 0],
              -k[1, 2] / k[1, 1], (h-k[1, 2]) / k[1, 1])
    gpu = GPUAnchorIndex.from_cpu(cpu)
    gpu_ids = torch.tensor(ids, device="cuda")
    # PixelORI's upper values originate in float32; this preserves those exact values.
    gpu_depth = torch.tensor(depth, dtype=torch.float32, device="cuda")
    exact_depth = gpu_depth.cpu().numpy().astype(np.float64)
    cpu_ids, cpu_times = {}, {}
    for mode in ("linear", "tree"):
        result = cpu.query(ids, exact_depth, w2c, domain, (w, h), mode=mode)
        cpu_ids[mode] = result.selected_anchor_ids.copy()
        cpu_times[mode] = result.timings
    def query(mode):
        torch.cuda.synchronize()
        start = perf_counter()
        result = gpu.query(gpu_ids, gpu_depth, w2c, domain, (w, h), mode=mode)
        torch.cuda.synchronize()
        elapsed = (perf_counter() - start) * 1000
        return result, elapsed
    warmups = []
    for mode in ("linear", "tree"):
        result, elapsed = query(mode)
        warmups.append({"mode": mode, "wall_ms": elapsed})
    rows, outputs = [], {}
    for repeat in range(args.repeats):
        for mode in (("linear", "tree") if repeat % 2 == 0 else ("tree", "linear")):
            result, elapsed = query(mode)
            selected = result.selected_anchor_ids.cpu().numpy()
            np.testing.assert_array_equal(selected, cpu_ids[mode])
            outputs[mode] = selected
            rows.append({"mode": mode, "repeat": repeat, "wall_ms": elapsed,
                         "timings": result.timings, "counters": result.collect_counters()})
    np.testing.assert_array_equal(outputs["linear"], outputs["tree"])
    report = {"scope": "CPU/GPU saved Truck frame 0 anchor query only; no fresh-render speedup claim",
              "index_path": args.index, "quality_path": args.quality, "ori_path": args.ori,
              "support_settings": vars(cpu.support_settings), "exact_cpu_gpu_ids": True,
              "gpu_resident_bytes": gpu.resident_bytes, "gpu_initialization_ms": gpu.initialization_ms,
              "cpu_reference_timings_ms": cpu_times, "warmups": warmups, "rows": rows}
    out = Path(args.output);out.mkdir(parents=True, exist_ok=True)
    (out / "gpu_comparison.json").write_text(json.dumps(report, indent=2))
    np.savez(out / "gpu_ids.npz", fov_ids=ids, **outputs)
    for mode in ("linear", "tree"):
        times = [row["wall_ms"] for row in rows if row["mode"] == mode]
        print({"mode": mode, "median_ms": float(np.median(times)), "times_ms": times,
               "selected": len(outputs[mode]), "culled": len(ids)-len(outputs[mode])}, flush=True)


if __name__ == "__main__":
    main()
