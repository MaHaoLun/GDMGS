"""CPU-only paired anchor benchmark on saved full-frame FoV/ORI inputs."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--quality", required=True)
    parser.add_argument("--ori", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile", choices=("saved", "alpha-support"), default="saved")
    args = parser.parse_args()
    from gdmgs.unified_index import AnchorIndex, SupportSettings
    index = AnchorIndex.load(args.index)
    if args.profile == "alpha-support":
        # World support is unchanged; this explicitly selects a new screen
        # raster profile for a controlled diagnostic. Source artifact is read-only.
        index.support_settings = SupportSettings()
    with np.load(args.quality, allow_pickle=False) as data:
        ids = data["fov_ids"].copy()
    with np.load(args.ori, allow_pickle=False) as data:
        depth = data["depth_bounds"].copy()
        w2c, intrinsic = data["w2c"].copy(), data["intrinsics"].copy()
        original_height, original_width = data["pixel_depth"].shape
    # Diagnostic producer froze 8-pixel ORI tiles with only bottom/right padding.
    width, height = depth.shape[1] * 8, depth.shape[0] * 8
    fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    domain = (-cx / fx, (width - cx) / fx, -cy / fy, (height - cy) / fy)
    def run(mode):
        begin = perf_counter()
        result = index.query(ids, depth, w2c, domain, (width, height), mode=mode)
        return result, (perf_counter() - begin) * 1000
    for mode in ("linear", "tree"):
        run(mode)
    rows, selections = [], {}
    for repeat in range(args.repeats):
        for mode in (("linear", "tree") if repeat % 2 == 0 else ("tree", "linear")):
            result, wall_ms = run(mode)
            rows.append({"repeat": repeat, "mode": mode, "wall_ms": wall_ms,
                         "timings": result.timings, "counters": result.counters})
            if mode in selections:
                np.testing.assert_array_equal(selections[mode], result.selected_anchor_ids)
            selections[mode] = result.selected_anchor_ids.copy()
    report = {"label": args.label, "scope": "CPU-only saved full-frame anchor query; not full render speedup",
              "index": args.index, "quality": args.quality, "ori": args.ori,
              "original_size": [original_width, original_height], "padded_size": [width, height],
              "fov_count": len(ids), "anchor_count": index.anchor_count,
              "ori_finite_cells": int(np.isfinite(depth).sum()), "ori_cells": depth.size,
              "support_settings": vars(index.support_settings), "pixel_pad": index.support_settings.pixel_pad,
              "repeats": args.repeats, "warmups_per_mode": 1, "rows": rows}
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    (output / (args.label + ".json")).write_text(json.dumps(report, indent=2))
    np.savez(output / (args.label + "_ids.npz"), fov_ids=ids, **selections)
    for mode in ("linear", "tree"):
        times = [row["wall_ms"] for row in rows if row["mode"] == mode]
        print({"label": args.label, "mode": mode, "median_ms": float(np.median(times)),
               "times_ms": times, "selected": len(selections[mode]), "culled": len(ids) - len(selections[mode])}, flush=True)


if __name__ == "__main__":
    main()
