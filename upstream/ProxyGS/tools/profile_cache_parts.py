"""Diagnostic synchronized breakdown of fixed-pair2 cache work, not speed evidence."""
import json
import time
from pathlib import Path

import torch

import render_step8_gpu_schedule as base
import render_step7d_pair2_gpu as experiment
from gdmgs.cache import temporal_bundle_cache_v3 as cache_module
from step7d_pack_candidate import take_requests_index_select


def measured(fn, records, name):
    def wrapper(*args, **kwargs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        torch.cuda.synchronize()
        records[name].append((time.perf_counter() - start) * 1000)
        return result
    return wrapper


def main():
    args = base.parser().parse_args()
    runtime = experiment.SceneRuntime(args)
    runtime.warm_runtime()
    runtime.active_mode = "gpu_pair2_control"
    runtime.cpu_baseline = False
    records = {"resolve": [], "decode": [], "pack_original": [], "pack_index_select": []}
    original_pack = cache_module.take_requests
    pack_calls = [0]

    def compare_pack(*args, **kwargs):
        order = (("original", original_pack), ("index_select", take_requests_index_select))
        if pack_calls[0] % 2:
            order = order[::-1]
        output = {}
        for name, function in order:
            torch.cuda.synchronize()
            start = time.perf_counter()
            output[name] = function(*args, **kwargs)
            torch.cuda.synchronize()
            records["pack_" + name].append((time.perf_counter() - start) * 1000)
        same, field = experiment.exact_batch(output["original"], output["index_select"])
        if not same:
            raise RuntimeError("pack candidate differs: " + str(field))
        pack_calls[0] += 1
        return output["original"]

    cache_module.take_requests = compare_pack
    cache = runtime.new_cache()
    cache._decode = measured(cache._decode, records, "decode")
    cache.resolve = measured(cache.resolve, records, "resolve")
    timeline = base.Timeline()
    try:
        for pair in range(len(runtime.views) // 2):
            demand = runtime.select_pair(pair, timeline)
            runtime._render_cache_pair(pair, demand, cache, timeline, collect=False)
        runtime.verify_pending()
    finally:
        cache_module.take_requests = original_pack
    total_cache_render = sum(item["elapsed_ms"] for item in timeline.records
                             if item["stage"] == "cache" and item["action"] == "end")
    report = {"scene": args.scene, "frames": len(runtime.views),
              "measurements_ms": {name: {"sum": sum(values), "count": len(values)}
                                  for name, values in records.items()},
              "cache_render_sum_ms": total_cache_render,
              "scope": "diagnostic synchronized stages; not paired online wall timing"}
    output = runtime.output / "cache_parts_profile.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
