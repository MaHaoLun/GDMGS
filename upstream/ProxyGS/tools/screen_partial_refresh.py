"""Bounded quality/speed screen for pose-risk repair within fixed pair2."""
import json
import time

import torch

import render_step8_gpu_schedule as base
import render_step7d_pair2_gpu as experiment
from gaussian_renderer.gdmgs_gsplat_backend import render_gdmgs_backend
from gdmgs.cache.temporal_bundle_cache_v3 import TemporalGeneration, take_requests
from render_step7b_fidelity import decode_batch


def diagnostic(view, batch, background):
    rgb = render_gdmgs_backend(view, batch, background, "RGB")
    image = torch.clamp(rgb["render"], 0.0, 1.0)
    ed = render_gdmgs_backend(view, batch, background, "RGB+ED")
    return {"image": image.detach().cpu(),
            "render_depth": ed["render_depth"].detach().cpu(),
            "render_alpha": ed["render_alpha"].detach().cpu()}


def run_candidate(runtime, fraction, collect):
    runtime.cpu_baseline = False
    runtime.pending_checks.clear()
    runtime.pending_mesh_checks.clear()
    cache = runtime.new_cache()
    timeline = base.Timeline()
    outputs, repaired = [], []
    torch.cuda.synchronize()
    start = time.perf_counter()
    for pair in range(len(runtime.views) // 2):
        demand = runtime.select_pair(pair, timeline)
        source_frame, target_frame = 2 * pair, 2 * pair + 1
        source, target = runtime.views[source_frame], runtime.views[target_frame]
        ids0, ids1 = demand.selected_ids
        if runtime.model.dist2level == "progressive":
            runtime.model.set_anchor_mask(source.camera_center, 40000, source.resolution_scale)
        batch0, _ = cache.resolve(
            frame_id=source_frame, anchor_ids=ids0, next_ids=ids1,
            decode=lambda ids, levels: decode_batch(source, runtime.model, ids, levels))
        if collect:
            outputs.append(diagnostic(source, batch0, runtime.background))
        else:
            render_gdmgs_backend(source, batch0, runtime.background, "RGB")

        points = runtime.model.get_anchor.detach()[ids1]
        ray0 = points - source.camera_center
        ray1 = points - target.camera_center
        score = 1 - (ray0 * ray1).sum(dim=1) / (
            ray0.norm(dim=1) * ray1.norm(dim=1)).clamp_min(1e-12)
        count = min(ids1.numel(), max(1, round(fraction * ids1.numel())))
        risky_rows = torch.topk(score, count, sorted=False).indices
        safe_mask = torch.ones(ids1.numel(), dtype=torch.bool, device=ids1.device)
        safe_mask[risky_rows] = False
        safe_positions = torch.nonzero(safe_mask, as_tuple=False).flatten()
        safe_ids = ids1[safe_positions]
        safe_batch = take_requests(cache.generation.batch, safe_positions, safe_ids,
                                   runtime.levels[safe_ids])
        cache.generation = TemporalGeneration(source_frame, safe_batch)
        if runtime.model.dist2level == "progressive":
            runtime.model.set_anchor_mask(target.camera_center, 40000, target.resolution_scale)
        batch1, stats = cache.resolve(
            frame_id=target_frame, anchor_ids=ids1,
            decode=lambda ids, levels: decode_batch(target, runtime.model, ids, levels))
        if stats["decoded_anchors"] != count:
            raise RuntimeError("repaired anchor count differs")
        if collect:
            outputs.append(diagnostic(target, batch1, runtime.background))
        else:
            render_gdmgs_backend(target, batch1, runtime.background, "RGB")
        repaired.append({"pair": pair, "repaired_anchors": count,
                         "requested_anchors": ids1.numel()})
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - start) * 1000
    runtime.verify_pending()
    return outputs, wall_ms, repaired


def main():
    args = base.parser().parse_args()
    runtime = experiment.SceneRuntime(args)
    runtime.warm_runtime()
    fresh_summary, fresh = runtime.run_mode("gpu_fresh", collect=True)
    control_summary, control = runtime.run_mode("gpu_pair2_control", collect=True)
    control_quality = runtime.quality(control, fresh)
    runtime.control_batches = []
    rows = {}
    for fraction in (0.05, 0.10):
        outputs, wall_ms, repaired = run_candidate(runtime, fraction, collect=True)
        quality = runtime.quality(outputs, fresh)
        rows[str(fraction)] = {"wall_ms_diagnostic": wall_ms, "quality": quality,
                               "repaired_anchors": sum(r["repaired_anchors"] for r in repaired),
                               "pairs": repaired}
        print(args.scene, fraction, wall_ms, quality["passed"], flush=True)
    performance = {"control": [], "0.05": [], "0.1": []}
    for repeat in range(3):
        order = ("control", "0.05", "0.1")
        order = order[repeat:] + order[:repeat]
        for mode in order:
            if mode == "control":
                summary, _ = runtime.run_mode("gpu_pair2_control", collect=False)
                performance[mode].append(summary["wall_ms"])
            else:
                _, elapsed, _ = run_candidate(runtime, float(mode), collect=False)
                performance[mode].append(elapsed)
    for mode, times in performance.items():
        performance[mode] = {"runs_ms": times, "median_ms": sorted(times)[1]}
    report = {"scene": args.scene, "frames": len(runtime.views),
              "fresh_wall_ms_diagnostic": fresh_summary["wall_ms"],
              "control_wall_ms_diagnostic": control_summary["wall_ms"],
              "control_quality": control_quality, "candidates": rows,
              "performance_screen": performance,
              "scope": "Hollywood-only quality plus three-repeat timing screen, not full-matrix retention"}
    (runtime.output / "partial_refresh_screen.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
