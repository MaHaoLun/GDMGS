"""Exactness and timing screen for pair2 GPU union plans on all 128 pairs."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from render_step6_windows import load_step5_ids
from step7d_execution_optimizations import GPUSetPlanner
from step7d_gpu_merge import bitmap_refresh_plan


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    return value, (time.perf_counter() - start) * 1000


def equal(left, right):
    return all(torch.equal(getattr(left, name), getattr(right, name)) for name in
               ("source", "next_ids", "union", "current_positions", "next_positions"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    results = {}
    for scene in manifest["scenes"]:
        ids = []
        anchor_count = None
        for item in scene["id_payloads"]:
            _, selected, _, count, _ = load_step5_ids(Path(item["path"]))
            if anchor_count is not None and count != anchor_count:
                raise ValueError("anchor universe drift")
            anchor_count = count
            ids.append(torch.from_numpy(selected.copy()).cuda())
        if len(ids) != 32:
            raise ValueError("full 32-frame window required")
        pairs = list(zip(ids[::2], ids[1::2]))
        original = lambda a, b: GPUSetPlanner([a, b]).refresh_plan(2)
        candidate = lambda a, b: bitmap_refresh_plan(a, b, anchor_count)
        for a, b in pairs:
            if not equal(original(a, b), candidate(a, b)):
                raise ValueError(scene["scene"] + ": bitmap union differs")
        for _ in range(2):
            for a, b in pairs:
                original(a, b)
                candidate(a, b)
        records = {"unique_cat": [], "bitmap": []}
        for repeat in range(args.repeats):
            order = ("unique_cat", "bitmap") if repeat % 2 == 0 else ("bitmap", "unique_cat")
            for mode in order:
                fn = original if mode == "unique_cat" else candidate
                _, ms = timed(lambda: [fn(a, b) for a, b in pairs])
                records[mode].append(ms)
        results[scene["scene"]] = {"pairs": len(pairs), "exact": True,
                                   "ms": records,
                                   "median_ms": {name: float(np.median(values)) for name, values in records.items()}}
        print(scene["scene"], results[scene["scene"]]["median_ms"], flush=True)
        del ids, pairs
        torch.cuda.empty_cache()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    total = {name: sum(results[scene]["median_ms"][name] for scene in results)
             for name in ("unique_cat", "bitmap")}
    result = {"status": "PASS", "scenes": results, "total_scene_median_ms": total,
              "bitmap_speedup": total["unique_cat"] / total["bitmap"],
              "scope": "GPU union planning only; no decode/render or online wall"}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"total_scene_median_ms": total, "bitmap_speedup": result["bitmap_speedup"]}))


if __name__ == "__main__":
    main()
