"""Independent full-count review; no GPU imports and no inferred kernel overlap."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median, mean

SCENES = ('amsterdam', 'barcelona', 'bilbao', 'chicago', 'hollywood', 'pompidou', 'quebec', 'rome')
MODES = ('cpu_serial_fresh_mesh32', 'gpu_serial_fresh_mesh32', 'gpu_serial_cache_mesh32', 'gpu_cpu_mesh_pipeline_q1_pool2', 'gpu_cpu_mesh_pipeline_q2_pool4', 'gpu_cpu_mesh_pipeline_window_retained')
BUDGET = {'psnr': (.20, .75), 'ssim': (.003, .010), 'lpips': (.006, .020), 'depth': (.01, .05)}


def review(runs_root, run_id, legacy_contract, minimum_repeats=3, scene_names=None):
    expected_scenes = SCENES if scene_names is None else tuple(scene_names)
    if not expected_scenes or len(set(expected_scenes)) != len(expected_scenes) or not set(expected_scenes).issubset(SCENES):
        raise ValueError("review scene scope must be unique frozen scenes")
    errors, scenes, walls = [], {}, {m: [] for m in MODES}
    def check(ok, message):
        if not ok:
            errors.append(message)
    def read(path):
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            errors.append(f'{path}: {exc}')
            return {}
    legacy = read(legacy_contract)
    check(legacy.get('status') == 'qualified', 'legacy contract missing or unqualified')
    legacy_identity = {'path': str(legacy_contract), 'sha256': hashlib.sha256(legacy_contract.read_bytes()).hexdigest()} if legacy_contract.is_file() else None
    for scene in expected_scenes:
        folder = runs_root / scene / run_id
        contract, summary, performance, status = (read(folder / n) for n in ('contract.json', 'summary.json', 'performance.json', 'status.json'))
        cameras = contract.get('camera_ids', [])
        check(len(cameras) == 32 and len(set(cameras)) == 32, f'{scene}: camera denominator/uniqueness')
        check(status.get('state') == 'complete' and status.get('status') == 'pass', f'{scene}: incomplete status')
        check(summary.get('status') == 'pass' and summary.get('frame_count') == 32, f'{scene}: summary status/count')
        check(contract.get('formal') is True, f'{scene}: not formal')
        check(contract.get('protocol') == 'proxygs-gpu-cpu-mesh-pipeline-v2', f'{scene}: incorrect pipeline protocol')
        check(contract.get('stream_policy') == 'default_stream_serialized_gpu_stages' and contract.get('overlap_claim') is False, f'{scene}: stream policy or unsupported overlap claim')
        check(set(contract.get('modes', [])) == set(MODES), f'{scene}: experiment mode matrix')
        cache = contract.get('cache_policy', {})
        check(cache.get('refresh_period') == 2 and cache.get('max_source_age') == 1 and cache.get('capacity_rows') == 6826846 and cache.get('payload_dtype') == 'float32', f'{scene}: frozen cache contract changed')
        per_mode = {}
        for mode in MODES:
            prefix = f'{scene}/{mode}'
            qualification = summary.get('qualification', {}).get(mode, {})
            records = performance.get(mode, [])
            check(len(records) >= minimum_repeats, f'{prefix}: fewer than {minimum_repeats} performance repeats')
            for repeat, record in [('qualification', qualification)] + list(enumerate(records)):
                label = f'{prefix}/{repeat}'
                check(record.get('mode') == mode and record.get('scene') == scene, f'{label}: identity')
                expected_role = 'diagnostic_only' if repeat == 'qualification' else 'performance'
                check(record.get('timing_role') == expected_role, f'{label}: diagnostic/performance timing role mismatch')
                check(record.get('frame_count') == 32, f'{label}: denominator')
                selected = record.get('selection_records', [])
                check([r.get('frame') for r in selected] == list(range(32)), f'{label}: frame order or missing selection')
                check([r.get('camera') for r in selected] == cameras, f'{label}: camera order')
                check(record.get('selection_oracle_exact') is True and all(r.get('oracle_exact') is True for r in selected), f'{label}: oracle mismatch')
                wall = record.get('wall_ms')
                check(isinstance(wall, (int, float)) and math.isfinite(wall) and wall > 0, f'{label}: invalid wall')
                if mode.startswith('gpu_'):
                    check(all(r.get('depth_d2h_bytes') == 0 and r.get('selected_ids_h2d_bytes') == 0 for r in selected), f'{label}: absent/nonzero transfer evidence')
                check(record.get('max_resident_rows', 0) <= 6826846, f'{label}: cache capacity exceeded')
                deadlines = record.get('deadline_records', [])
                if 'scheduled' in mode:
                    expected_lead = 3 if mode == 'cpu_scheduled_cache_q2_mesh32' else (2 if 'q2' in mode else 1)
                    check(record.get('schedule_lead_pairs') == expected_lead, f'{label}: schedule lead policy differs')
                    check([d.get('pair') for d in deadlines] == list(range(16)), f'{label}: incomplete deadlines')
                    late = [d['pair'] for d in deadlines if d.get('late')]
                    check(record.get('deadline_misses') == len(late) and record.get('fallback_pairs') == late, f'{label}: fallback/deadline inconsistency')
            if mode.startswith('gpu_cpu_mesh_pipeline_'):
                if mode == 'gpu_cpu_mesh_pipeline_window_retained':
                    q, workers = 15, contract.get('retained_profile', {}).get('workers')
                    check(workers == (24 if scene == 'barcelona' else 32), f'{prefix}: retained worker profile differs')
                    config = contract.get('pipeline_modes', {}).get(mode, {})
                    check(config.get('future_camera_scope') == 'frozen_known_32_camera_window', f'{prefix}: missing known-window future camera scope')
                else:
                    q, workers = (1, 2) if 'q1_pool2' in mode else (2, 4)
                for repeat, record in [('qualification', qualification)] + list(enumerate(records)):
                    label = f'{prefix}/{repeat}'
                    pipeline = record.get('mesh_pipeline', {})
                    check(pipeline.get('lookahead_pairs') == q and pipeline.get('pool_workers') == workers, f'{label}: mesh pool or lead configuration')
                    check(pipeline.get('actual_mesh_workers') == workers and pipeline.get('native_query_threads') == 1, f'{label}: actual mesh worker evidence')
                    check(all(r.get('mesh_workers') == workers and r.get('mesh_native_query_threads') == 1 for r in record.get('selection_records', [])), f'{label}: per-frame mesh worker evidence')
                    waits = pipeline.get('wait_records', [])
                    if mode == 'gpu_cpu_mesh_pipeline_window_retained':
                        check(pipeline.get('future_camera_scope') == 'frozen_known_32_camera_window' and pipeline.get('worker_policy') == 'frozen_retained_profile' and pipeline.get('submission_policy') == 'online_full_window_pairwise_consumption', f'{label}: window scope or submission policy differs')
                        check(bool(waits) and waits[0].get('submitted_pairs') == list(range(16)) and waits[0].get('pending_pairs') == list(range(16)), f'{label}: full window not submitted before first pair wait')
                    check([r.get('pair') for r in waits] == list(range(16)), f'{label}: mesh wait denominator')
                    wait_values = []
                    for r in waits:
                        pair = r.get('pair', -1)
                        pending = r.get('pending_pairs', [])
                        submitted = r.get('submitted_pairs', [])
                        check(isinstance(pending, list) and len(pending) <= q + 1 and len(pending) == len(set(pending)) and pair in pending and all(pair <= p <= pair + q for p in pending), f'{label}: mesh pending queue bound')
                        check(isinstance(submitted, list) and pair in submitted and set(pending).issubset(submitted), f'{label}: mesh submissions do not explain pending work')
                        check(r.get('lookahead_pairs') == q and r.get('pool_workers') == workers, f'{label}: mesh wait policy differs')
                        wait_ms = r.get('wait_ms')
                        finite = isinstance(wait_ms, (int, float)) and math.isfinite(wait_ms) and wait_ms >= 0
                        check(finite, f'{label}: invalid mesh wait')
                        if finite:
                            wait_values.append(wait_ms)
                    for name in ('startup_mesh_wait_ms', 'total_mesh_wait_ms', 'cpu_mesh_gpu_stage_host_overlap_ms'):
                        value = pipeline.get(name)
                        check(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0, f'{label}: absent/invalid {name}')
                    if len(wait_values) == 16:
                        check(math.isclose(pipeline.get('total_mesh_wait_ms', -1), sum(wait_values), rel_tol=1e-7, abs_tol=1e-5), f'{label}: mesh wait sum mismatch')
                        check(math.isclose(pipeline.get('startup_mesh_wait_ms', -1), wait_values[0], rel_tol=1e-7, abs_tol=1e-5), f'{label}: startup mesh wait mismatch')
            if mode == 'gpu_serial_fresh_mesh32':
                exact = qualification.get('fresh_exact', {})
                exact_records = exact.get('records', [])
                check(exact.get('passed') is True and len(exact_records) == 32, f'{prefix}: GPU fresh exact gate missing or failed')
                check([r.get('frame') for r in exact_records] == list(range(32)) and [r.get('camera') for r in exact_records] == cameras, f'{prefix}: GPU fresh exact record identity')
                check(all(r.get(field) is True for r in exact_records for field in ('image', 'render_depth', 'render_alpha')), f'{prefix}: GPU fresh outputs are not exactly equal to CPU fresh')
            if mode != 'cpu_serial_fresh_mesh32':
                quality = qualification.get('quality', {})
                qrecords = quality.get('records', [])
                check(len(qrecords) == 32 and [q.get('frame') for q in qrecords] == list(range(32)), f'{prefix}: incomplete quality')
                check([q.get('camera') for q in qrecords] == cameras, f'{prefix}: quality camera identity')
                for metric, (avg_limit, worst_limit) in BUDGET.items():
                    try:
                        values = [q['depth']['relative_mae'] if metric == 'depth' else q['metrics']['reuse_minus_fresh_gt'][metric] * (1 if metric == 'lpips' else -1) for q in qrecords]
                        check(len(values) == 32 and all(math.isfinite(v) for v in values) and mean(values) <= avg_limit and max(values) <= worst_limit, f'{prefix}: independently recomputed {metric} quality failed')
                    except (KeyError, TypeError, ValueError):
                        errors.append(f'{prefix}: malformed {metric} quality')
            times = [r.get('wall_ms') for r in records]
            if times and all(isinstance(t, (int, float)) and math.isfinite(t) and t > 0 for t in times):
                value = median(times)
                walls[mode].append(value)
                check(math.isclose(summary.get('performance_median_wall_ms', {}).get(mode, -1), value, rel_tol=1e-9), f'{prefix}: reported median mismatch')
                per_mode[mode] = {'median_wall_ms': value, 'repeats': len(records), 'deadline_misses': [r.get('deadline_misses') for r in records]}
        scenes[scene] = per_mode
    paired = {m: sum(v) if len(v) == len(expected_scenes) else None for m, v in walls.items()}
    valid = not errors
    winner = min((m for m in MODES if m.startswith('gpu_')), key=lambda m: paired[m]) if valid else None
    return {'schema': 'proxygs_gpu_mesh_pipeline_independent_review_v1', 'status': 'pass' if valid else 'failed', 'failures': errors, 'scope': 'full_matrix' if expected_scenes == SCENES else 'qualification_subset', 'expected_scene_count': len(expected_scenes), 'expected_frames_per_mode_repeat': 32 * len(expected_scenes), 'minimum_performance_repeats': minimum_repeats, 'paired_wall_ms': paired, 'lowest_wall_gpu_mode': winner, 'legacy_contract': legacy_identity, 'legacy_comparison_kind': 'historical; new CPU baseline is reported separately', 'kernel_overlap': 'not established by host timelines; no overlap claim', 'per_scene': scenes}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--runs-root', type=Path, required=True)
    cli.add_argument('--run-id', required=True)
    cli.add_argument('--legacy-contract', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    result = review(args.runs_root, args.run_id, args.legacy_contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': result['status'], 'failures': result['failures'], 'output': str(args.output)}))
    return 0 if result['status'] == 'pass' else 1

if __name__ == '__main__':
    raise SystemExit(main())
