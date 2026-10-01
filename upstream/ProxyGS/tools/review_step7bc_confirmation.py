"""Independent persisted-record reviewer; does not import candidate runtime."""
import argparse
import hashlib
import numpy as np
import json
import math
import statistics
from pathlib import Path

SCENES = ['amsterdam','barcelona','bilbao','chicago','hollywood','pompidou','quebec','rome']
BUDGET = {'psnr': (.20,.75), 'ssim': (.003,.010), 'lpips': (.006,.020)}
POLICIES = ['pair2_control','pair2_fast','pair2_arena','epoch3','epoch4','pair2_fp16','pair2_c50','pair2_c75']

def read(p):
    return json.loads(p.read_text())

def main(root, run):
    failures=[]
    runtime=root/'runtime/Proxy-GS-eac937e8'
    source_manifest=read(root/'manifests/confirmation_source_sha256.json')
    for name,digest in source_manifest.items():
        if hashlib.sha256((runtime/name).read_bytes()).hexdigest()!=digest:
            failures.append('source_changed:'+name)
    test_log=(root/'tests/confirmation_contract_tests.txt').read_text()
    if '74 passed' not in test_log or 'failed' in test_log.lower():
        failures.append('regression_tests')
    first_contract=read(root/'runs'/SCENES[0]/run/'contract.json')
    policies=list(first_contract['policies'])
    records={p:[] for p in policies}
    summaries={}
    for scene in SCENES:
        base=root/'runs'/scene/run
        summary=read(base/'summary.json')
        contract=read(base/'contract.json')
        perf=read(base/'performance.json')
        for name,identity in contract['inputs'].items():
            path=Path(identity['path'])
            stat=path.stat()
            if stat.st_size!=identity['bytes'] or stat.st_mtime_ns!=identity['mtime_ns']:
                failures.append(scene+':input_changed:'+name)
        trace=read(Path(contract['inputs']['trace']['path']))
        upstream=next(s for s in trace['scenes'] if s['scene']==scene)
        if upstream['camera_ids']!=contract['camera_ids']:
            failures.append(scene+':upstream_window_changed')
        expected_ids=[]
        for payload in upstream['id_payloads']:
            with np.load(payload['path'],allow_pickle=False) as data:
                ids=np.flatnonzero(np.unpackbits(data['selected_bitmap'],count=int(data['anchor_universe_count'].item()),bitorder='little')).astype(np.int64)
                expected_ids.append((len(ids),hashlib.sha256(ids.tobytes()).hexdigest()))
        summaries[scene]=summary
        if summary['frame_count'] != 32 or len(contract['camera_ids']) !=32 or not summary['inputs_unchanged']:
            failures.append(scene+':denominator_or_identity')
        for p in policies:
            rows=[read(base/p/f'{i:03}.json') for i in range(32)]
            for i,r in enumerate(rows):
                if r['camera'] != contract['camera_ids'][i] or r['frame']!=i:
                    failures.append(scene+':camera_order')
                if not r['metrics']['gates']['finite']:
                    failures.append(scene+':nonfinite')
                if (r['selected_count'],r['selected_sha256'])!=expected_ids[i]:
                    failures.append(scene+':upstream_request_mismatch')
                s=r['stats']
                if s['selected_anchors']!=r['selected_count'] or s['miss_anchors']+s['hit_anchors']!=r['selected_count']:
                    failures.append(scene+':request_accounting')
                if s['resident_rows']>s['capacity_rows'] or s['source_age'] not in range(contract['policies'][p]['period']):
                    failures.append(scene+':capacity_or_age')
                if p!='pair2_control' and 'pair2_control' in policies:
                    other=read(base/'pair2_control'/f'{i:03}.json')
                    if other['selected_sha256']!=r['selected_sha256']:
                        failures.append(scene+':selection_changed')
                if i%contract['policies'][p]['period']==0 and r['metrics']['direct']['rgb_max_abs']!=0:
                    failures.append(scene+':refresh_not_exact')
                depth=r['depth']
                if not depth['finite'] or depth['overlap_pixels']<=0:
                    failures.append(scene+':invalid_depth')
                if p in ('pair2_fast','pair2_arena','pair2_prefix') and 'pair2_control' in policies and r['control_rgb_exact'] is not True:
                    failures.append(scene+':lossless_ablation_changed_output')
                r['scene']=scene
            if len(perf[p]) != contract['performance_repeats'] or len(perf['fresh'])!=contract['performance_repeats'] or any(len(v['frame_ms'])!=32 for v in perf[p]+perf['fresh']):
                failures.append(scene+':performance_denominator')
            if any(any(v<=0 or not math.isfinite(v) for v in rep['frame_ms']) for rep in perf[p]+perf['fresh']):
                failures.append(scene+':invalid_timing')
            for rep in perf[p]:
                if [s['decoder_calls'] for s in rep['stats']] != [r['stats']['decoder_calls'] for r in rows]:
                    failures.append(scene+':replay_not_equivalent')
            expected_fresh=statistics.median(v['total_ms'] for v in perf['fresh'])
            expected_candidate=statistics.median(v['total_ms'] for v in perf[p])
            if summary['policies'][p]['fresh_ms']!=expected_fresh or summary['policies'][p]['candidate_ms']!=expected_candidate:
                failures.append(scene+':timing_summary_mismatch')
            if any(abs(sum(v['frame_ms'])-v['total_ms'])>1e-6 for v in perf[p]+perf['fresh']):
                failures.append(scene+':timing_sum_mismatch')
            records[p].extend(rows)
    reports={}
    for p,rows in records.items():
        quality={}
        for metric,(mean_max,worst_max) in BUDGET.items():
            losses=[r['metrics']['reuse_minus_fresh_gt'][metric]*(1 if metric=='lpips' else -1) for r in rows]
            worst=max(range(len(rows)),key=lambda i:losses[i])
            scene_means={s: statistics.mean([losses[i] for i,r in enumerate(rows) if r['scene']==s]) for s in SCENES}
            quality[metric]=dict(mean_loss=statistics.mean(losses),worst_loss=losses[worst],
                worst_scene=rows[worst]['scene'],worst_frame=rows[worst]['frame'],scene_means=scene_means,
                passed=statistics.mean(losses)<=mean_max and max(losses)<=worst_max and max(scene_means.values())<=mean_max)
        stats=[r['stats'] for r in rows]
        base_ms=sum(summaries[s]['policies'][p]['fresh_ms'] for s in SCENES)
        cache_ms=sum(summaries[s]['policies'][p]['candidate_ms'] for s in SCENES)
        calls=sum(s['decoder_calls'] for s in stats)
        decoded=sum(s['decoded_anchors'] for s in stats)
        baseline=sum(s['selected_anchors'] for s in stats)
        depth_values=[r['depth']['relative_mae'] for r in rows]
        depth_scene_means={scene:statistics.mean(r['depth']['relative_mae'] for r in rows if r['scene']==scene) for scene in SCENES}
        depth_pass=(statistics.mean(depth_values)<=.01 and max(depth_values)<=.05 and max(depth_scene_means.values())<=.01)
        depth=dict(relative_mae_mean=statistics.mean(depth_values),relative_mae_worst=max(depth_values),
                   relative_p95_worst=max(r['depth']['relative_p95'] for r in rows),
                   relative_p99_worst=max(r['depth']['relative_p99'] for r in rows),
                   foreground_xor_mean=statistics.mean(r['depth']['foreground_xor_fraction'] for r in rows),
                   foreground_xor_worst=max(r['depth']['foreground_xor_fraction'] for r in rows),
                   overlap_fraction_min=min(r['depth']['overlap_fraction'] for r in rows),
                   scene_relative_mae=depth_scene_means,passed=depth_pass)
        hits=sum(s['hit_anchors'] for s in stats)
        empty_hits=sum(s['empty_hits'] for s in stats)
        nonempty_requests=sum(s['selected_anchors']-s['empty_output_requests'] for s in stats)
        reuse_requests=sum(s['selected_anchors'] for s in stats if s['source_age']>0)
        cache_metrics=dict(anchor_hit_rate=hits/baseline,positive_hit_rate=(hits-empty_hits)/baseline,
            positive_hit_rate_among_nonempty_output_requests=(hits-empty_hits)/max(1,nonempty_requests),
            empty_hit_rate=empty_hits/baseline,reuse_frame_hit_rate=hits/max(1,reuse_requests),
            row_hit_fraction=sum(s['hit_rows'] for s in stats)/sum(s['output_rows'] for s in stats),
            extra_prefetched_anchors=sum(s['prefetch_anchors'] for s in stats),
            extra_prefetch_utilization=sum(r['prefetch_extra_first_use'] for r in rows)/max(1,sum(s['prefetch_anchors'] for s in stats)),
            evicted_anchors=sum(s['evicted_anchors'] for s in stats),
            lookahead_requests=sum(s['lookahead_requests'] for s in stats),
            whole_bundle_eviction_rate=sum(s['evicted_anchors'] for s in stats)/max(1,sum(s['lookahead_requests'] for s in stats)),
            max_row_utilization=max(s['resident_rows']/s['capacity_rows'] for s in stats),
            max_descriptors=max(s['resident_descriptors'] for s in stats),
            max_empty_descriptors=max(s['resident_empty_descriptors'] for s in stats),
            hit_age_anchor_histogram={str(age):sum(s['hit_anchors'] for s in stats if s['source_age']==age) for age in (1,2,3)},
            frame_age_histogram={str(age):sum(s['source_age']==age for s in stats) for age in (0,1,2,3)})
        quality_pass=all(q['passed'] for q in quality.values()) and depth_pass
        performance_pass=base_ms/cache_ms>1 and calls<256 and decoded<baseline
        reports[p]=dict(quality=quality,depth=depth,cache_metrics=cache_metrics,quality_pass=quality_pass,performance_pass=performance_pass,
            qualified=quality_pass and performance_pass and not failures,
            fresh_ms=base_ms,cache_ms=cache_ms,speedup=base_ms/cache_ms,
            decoder_calls=calls,baseline_decoder_calls=256,decoder_call_reduction=1-calls/256,
            decoded_anchors=decoded,baseline_decoded_anchors=baseline,decoded_anchor_reduction=1-decoded/baseline,
            empty_hits=sum(s['empty_hits'] for s in stats),max_resident_rows=max(s['resident_rows'] for s in stats),
            max_resident_bytes=max(s['resident_bytes'] for s in stats),
            max_scratch_bytes=max(r['scratch_bytes'] for r in rows),
            per_scene={s:summaries[s]['policies'][p] for s in SCENES})
    passed=[p for p in policies if reports[p]['qualified']]
    winner=max(passed,key=lambda p: reports[p]['speedup']) if passed else None
    result=dict(status='pass' if winner else 'not_qualified', scene_count=8,frame_count=256,
                policies=reports,selected_policy=winner,failures=failures,
                scope='cache-stage with frozen Step6 lookahead; excludes online index/depth and Step8 scheduling')
    review=root/'review'
    review.mkdir(exist_ok=True)
    (review/f'final_{run}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='policies'},indent=2))
    for p in policies:
        print(p,json.dumps({k:v for k,v in reports[p].items() if k not in ('per_scene','quality','depth','cache_metrics')}))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--run',default='confirmation_v1')
    args=parser.parse_args()
    main(args.root,args.run)
