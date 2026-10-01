"""Joint 7B/7C sequential replay; reference output never enters the cache."""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from render_step7b_fidelity import (
    parser, validate_inputs, unpack_selected_ids, decode_batch, image_metrics,
    atomic_json, load, FORMAL_CAPACITY_ROWS,
)
from render_gdmgs_backend import _load_cfg, _new_model, _frozen_camera_names, _ordered_views, _file_identity
from gaussian_renderer.gdmgs_gsplat_backend import render_gdmgs_backend
from gdmgs.cache import CacheIdentity
from gdmgs.cache.temporal_bundle_cache_v3 import TemporalBundleCache

POLICIES = {
    'pair2_control': dict(period=2),
    'pair2_prefix': dict(period=2, fast_handoff=True, prefix_decode=True),
    'pair2_fast': dict(period=2, fast_handoff=True),
    'pair2_arena': dict(period=2, union_arena=True, fast_handoff=True),
    'epoch3': dict(period=3, fast_handoff=True),
    'epoch4': dict(period=4, fast_handoff=True),
    'pair2_fp16': dict(period=2, fast_handoff=True, payload_half=True),
    'pair2_c50': dict(period=2, fast_handoff=True, capacity_rows=3_413_423),
    'pair2_c75': dict(period=2, fast_handoff=True, capacity_rows=5_120_134),
}
BUDGET = {'psnr_mean': 0.20, 'psnr_worst': 0.75, 'ssim_mean': 0.003,
          'ssim_worst': 0.010, 'lpips_mean': 0.006, 'lpips_worst': 0.020}


def depth_metrics(candidate, reference):
    depth = candidate['render_depth']
    ref = reference['render_depth']
    a = candidate['render_alpha'] >= 0.5
    b = reference['render_alpha'] >= 0.5
    valid = a & b & (ref > 1e-6)
    delta = (depth - ref).abs()
    rel = delta[valid] / ref[valid].clamp_min(1e-6)
    return dict(finite=bool(torch.isfinite(depth).all() and torch.isfinite(ref).all()),
                overlap_pixels=int(valid.sum()), pixel_count=valid.numel(),
                overlap_fraction=float(valid.float().mean()),
                foreground_xor_fraction=float((a != b).float().mean()),
                reference_foreground_fraction=float(b.float().mean()),
                candidate_foreground_fraction=float(a.float().mean()),
                mae=float(delta[valid].mean()) if rel.numel() else 0.,
                relative_mae=float(rel.mean()) if rel.numel() else 0.,
                relative_p95=float(torch.quantile(rel,0.95)) if rel.numel() else 0.,
                relative_p99=float(torch.quantile(rel,0.99)) if rel.numel() else 0.)


def quality_summary(records):
    deltas = [r['metrics']['reuse_minus_fresh_gt'] for r in records]
    report = {}
    for name in ('psnr', 'ssim', 'lpips'):
        loss = np.asarray([(-d[name] if name != 'lpips' else d[name]) for d in deltas])
        report[name] = dict(mean_loss=float(loss.mean()), worst_loss=float(loss.max()),
                            worst_frame=int(loss.argmax()),
                            passed=bool(np.isfinite(loss).all() and loss.mean() <= BUDGET[name+'_mean']
                                        and loss.max() <= BUDGET[name+'_worst']))
    report['pass'] = all(v['passed'] for v in report.values())
    return report


def main(args):
    if not args.formal or args.max_views is not None:
        raise ValueError('requires complete frozen windows')
    if args.width != 1600 or args.height != 900 or args.iteration != 40000:
        raise ValueError('frozen resolution/checkpoint required')
    trace, _, _, _ = validate_inputs(args)
    policies = {p: POLICIES[p] for p in args.policies}
    out = args.output_root / args.scene / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    cfg = _load_cfg(args.model_path)
    if Path(cfg.source_path).resolve() != args.source_path.resolve():
        raise ValueError('model/source binding differs')
    cfg.data_device = 'cpu'
    model = _new_model(cfg)
    from scene import Scene
    scene = Scene(cfg, model, load_iteration=40000, shuffle=False, resolution_scales=cfg.resolution_scales)
    model.eval()
    all_views = _ordered_views(scene, _frozen_camera_names(args.model_path))
    if len(all_views) != args.expected_views:
        raise ValueError('full-scene camera denominator changed')
    by_name = {v.image_name: v for v in all_views}
    views = [by_name[n] for n in trace['camera_ids']]
    if len(views) != 32 or len(trace['id_payloads']) != 32:
        raise ValueError('window denominator changed')
    levels = model.get_level.detach().view(-1).long().contiguous()
    cpu_ids = [torch.from_numpy(unpack_selected_ids(p, levels.numel()).copy()) for p in trace['id_payloads']]
    for view, payload in zip(views, trace['id_payloads']):
        if Path(payload['path']).name != view.image_name + '.npz':
            raise ValueError('camera/selected IDs mismatch')
    background = torch.tensor([1.,1.,1.] if cfg.white_background else [0.,0.,0.], device='cuda')
    identity = CacheIdentity(args.scene, str(args.model_path), 'gdmgs-gsplat-v1',
                             str(args.model_path / 'point_cloud/iteration_40000/point_cloud.ply'), str(args.cache_trace))
    def new_cache(policy):
        config=dict(policies[policy])
        period=config.pop('period')
        capacity=config.pop('capacity_rows',FORMAL_CAPACITY_ROWS)
        return TemporalBundleCache(identity=identity, anchor_levels=levels,
                                   capacity_rows=capacity, n_offsets=model.n_offsets,
                                   max_age=period-1, **config)
    files = {'trace': args.cache_trace, 'cfg': args.model_path/'cfg_args',
             'cameras': args.model_path/'cameras.json',
             'ply': args.model_path/'point_cloud/iteration_40000/point_cloud.ply'}
    for n in ('opacity', 'cov', 'color'):
        files[n] = args.model_path/f'point_cloud/iteration_40000/{n}_mlp.pt'
    inputs = {n: _file_identity(p) for n,p in files.items()}
    atomic_json(out/'contract.json', dict(protocol='step7bc-confirmation-v1', camera_ids=trace['camera_ids'],
                capacity_rows=FORMAL_CAPACITY_ROWS, budget=BUDGET,
                policies=policies, depth_budget=dict(relative_mae_mean=.01,relative_mae_worst=.05),
                lookahead='frozen Step6 requests; available-input cache-stage experiment',
                performance_repeats=args.repeat, inputs=inputs))
    import lpips
    metric = lpips.LPIPS(net='vgg').cuda().eval()
    caches = {p: new_cache(p) for p in policies}
    records = {p: [] for p in policies}
    fresh_times = []
    prefetch_used = {}

    def fresh_frame(i):
        ids = cpu_ids[i].cuda()
        model.set_anchor_mask(views[i].camera_center, 40000, views[i].resolution_scale)
        batch = decode_batch(views[i], model, ids, levels[ids])
        return torch.clamp(render_gdmgs_backend(views[i], batch, background, 'RGB')['render'], 0., 1.), batch

    def candidate_frame(i, policy, cache):
        ids = cpu_ids[i].cuda()
        nxt = None
        period=policies[policy]['period']
        if i % period == 0 and i + 1 < 32:
            future=[cpu_ids[j].cuda() for j in range(i+1,min(32,i+period))]
            nxt=future[0] if len(future)==1 else torch.unique(torch.cat(future),sorted=True)
        model.set_anchor_mask(views[i].camera_center, 40000, views[i].resolution_scale)
        batch, stats = cache.resolve(frame_id=i, anchor_ids=ids, next_ids=nxt,
                            decode=lambda ids, ls: decode_batch(views[i], model, ids, ls))
        image = torch.clamp(render_gdmgs_backend(views[i], batch, background, 'RGB')['render'], 0., 1.)
        return image, stats, batch

    def timed(fn):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        return result, (time.perf_counter()-start)*1000

    with torch.no_grad():
        fresh_frame(0)  # identical renderer/kernel warm-up, outside measurements
        candidate_frame(0, next(iter(policies)), new_cache(next(iter(policies))))
        for i, view in enumerate(views):
            (ref, ref_batch), ms = timed(lambda: fresh_frame(i))
            ref_depth=render_gdmgs_backend(view,ref_batch,background,'RGB+ED')
            control_image=None
            fresh_times.append(ms)
            gt = view.original_image.cuda().clamp(0.,1.)
            for p in policies:
                torch.cuda.reset_peak_memory_stats()
                before = torch.cuda.memory_allocated()
                previous_generation=caches[p].generation
                (image, stats, batch), ms = timed(lambda: candidate_frame(i,p,caches[p]))
                peak = torch.cuda.max_memory_allocated()
                after = torch.cuda.memory_allocated()
                extra_first_use=0
                if previous_generation is not None and stats['hit_anchors']:
                    source=stats['source_frame']
                    resident_ids=previous_generation.batch.anchor_indices.cpu().numpy()
                    hit_ids=np.intersect1d(cpu_ids[i].numpy(),resident_ids,assume_unique=True)
                    useful_extra=set(np.setdiff1d(hit_ids,cpu_ids[source].numpy(),assume_unique=True).tolist())
                    seen=prefetch_used.setdefault((p,source),set())
                    extra_first_use=len(useful_extra-seen)
                    seen.update(useful_extra)
                metrics = image_metrics(image,ref,gt,metric)
                depth=depth_metrics(render_gdmgs_backend(view,batch,background,'RGB+ED'),ref_depth)
                if p=='pair2_control':
                    control_image=image
                control_exact=None
                if p in ('pair2_fast','pair2_arena','pair2_prefix') and control_image is not None:
                    control_exact=torch.equal(image,control_image)
                    if not control_exact:
                        raise ValueError('lossless ablation changed RGB output')
                record = dict(frame=i, camera=view.image_name, selected_count=len(cpu_ids[i]),
                              selected_sha256=hashlib.sha256(cpu_ids[i].numpy().tobytes()).hexdigest(),
                              cache_stage_ms=ms, stats=stats, metrics=metrics, depth=depth,
                              control_rgb_exact=control_exact, prefetch_extra_first_use=extra_first_use,
                              memory_peak_bytes=peak, scratch_bytes=max(0,peak-max(before,after)))
                records[p].append(record)
                atomic_json(out/p/f'{i:03}.json',record)
                if i % policies[p]['period'] == 0 and not torch.equal(image,ref):
                    raise ValueError('fresh union decode changed current-pose render')
            atomic_json(out/'status.json',dict(state='quality',completed_frames=i+1))
            print(args.scene, 'quality', i+1, flush=True)
        # Repeat whole stateful sequences, without LPIPS/GT/reference interference.
        # Alternate fresh/candidate sequence order to reduce systematic warming bias.
        performance = {p: [] for p in ['fresh', *policies]}
        for repeat in range(args.repeat):
            order = list(performance)
            if repeat % 2:
                order.reverse()
            for p in order:
                cache = new_cache(p) if p != 'fresh' else None
                samples, stats_list = [], []
                for i in range(32):
                    if p == 'fresh':
                        _, ms = timed(lambda: fresh_frame(i))
                    else:
                        (_, stats, _), ms = timed(lambda: candidate_frame(i,p,cache))
                        stats_list.append(stats)
                    samples.append(ms)
                performance[p].append(dict(total_ms=sum(samples), frame_ms=samples, stats=stats_list))
                atomic_json(out/'performance.json',performance)
            print(args.scene, 'performance',repeat+1,flush=True)
    summary = dict(scene=args.scene, frame_count=32, full_view_count=len(all_views),
                   camera_ids=trace['camera_ids'], budget=BUDGET, policies={},
                   inputs_unchanged=inputs == {n:_file_identity(p) for n,p in files.items()})
    base_ms=float(np.median([r['total_ms'] for r in performance['fresh']]))
    for p in policies:
        rows=records[p]
        candidate_ms=float(np.median([r['total_ms'] for r in performance[p]]))
        summary['policies'][p]=dict(quality=quality_summary(rows), fresh_ms=base_ms,
            candidate_ms=candidate_ms, speedup=base_ms/candidate_ms,
            fresh_decoder_calls=32, decoder_calls=sum(r['stats']['decoder_calls'] for r in rows),
            fresh_decoded_anchors=sum(len(ids) for ids in cpu_ids),
            decoded_anchors=sum(r['stats']['decoded_anchors'] for r in rows),
            max_resident_bytes=max(r['stats']['resident_bytes'] for r in rows),
            max_resident_rows=max(r['stats']['resident_rows'] for r in rows),
            max_scratch_bytes=max(r['scratch_bytes'] for r in rows))
    atomic_json(out/'summary.json',summary)
    atomic_json(out/'status.json',dict(state='complete',completed_frames=32))
    print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    cli=parser()
    cli.add_argument('--policies',nargs='+',choices=list(POLICIES),default=['pair2_control','pair2_fast','pair2_arena','pair2_prefix'])
    args=cli.parse_args()
    try:
        main(args)
    except Exception as error:
        atomic_json(args.output_root/args.scene/args.run_id/'failure.json',dict(error=repr(error)))
        raise
