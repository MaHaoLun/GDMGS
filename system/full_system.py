"""Checkpoint-to-image execution using the retained native and fused kernels."""
import argparse
import copy
import json
import math
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from cache_build_optim import build_union_plan, PlannedArena
from epoch_cache import epoch_spec, source_view
from fullblock_timed import Admission
from model_bridge import prepare_decode, finish_decode, decode_batch, source_state
from priority_renderer import PriorityRenderer
from ascending_cache import take_requests
from raster_backend import render_gdmgs_backend
from data import load
from selection_runtime import Selection
from planning import allocate


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')




class Lane:
    def __init__(self, e, render_workers):
        self.e = copy.copy(e)
        self.e.rt = copy.copy(e.rt)
        self.e.rt.model = copy.copy(e.rt.model)
        for name in ('_anchor_mask', '_prog_ratio', 'transition_mask'):
            value = getattr(e.rt.model, name, None)
            if isinstance(value, torch.Tensor):
                setattr(self.e.rt.model, name, value.clone())
        self.stream = torch.cuda.Stream()
        self.renderer = PriorityRenderer(e.views, e.rt.background, requests_sorted=True)
        if render_workers == 1:
            self.renderer.workers[1] = self.renderer.workers[0]
        elif render_workers != 2:
            raise ValueError('retained renderer supports one or two target workers')
        self.executor = ThreadPoolExecutor(max_workers=1)

    def run(self, index, spec, requests, pool, verify):
        torch.set_grad_enabled(False)
        try:
            with torch.cuda.stream(self.stream):
                for ids in requests:
                    ids.record_stream(self.stream)
                plan = build_union_plan(requests)
                source = source_view(self.e, spec)
                m = self.e.rt.model
                source_state(source, m, getattr(m, '_gdmgs_iteration', 40000))
                state = prepare_decode(source, m, plan.ordered) if len(plan.ordered) else None
                rows = state['rows'] if state is not None else 0
                pool.acquire(index, rows)
                arena = PlannedArena(spec, requests, self.e.rt.levels,
                    lambda ids, levels: finish_decode(state, m, ids, levels), pool.capacity, plan)
                pool.materialized(index, rows)
                parity = {}
                if verify and len(plan.ordered):
                    ref = decode_batch(source, m, plan.ordered, self.e.rt.levels[plan.ordered])
                    for name in ('xyz', 'color', 'opacity', 'scaling', 'rotation', 'selection_mask'):
                        parity[name] = torch.equal(getattr(arena.batch, name), getattr(ref, name))
                    for name, value in vars(ref.bundle_metadata).items():
                        parity['metadata_' + name] = torch.equal(value, getattr(arena.batch.bundle_metadata, name))
                    if not all(parity.values()):
                        raise RuntimeError(f'split decoder differs from original: {parity}')
                    del ref
                frames = list(range(spec['start'], spec['end']))
                render_state = self.renderer.prepare_epoch(arena, requests, frames, 'staged_count2')
                outputs, stats = self.renderer.render_epoch(render_state, True, 'staged_count2')
                done = torch.cuda.Event(); done.record()
            done.synchronize()
            self.renderer._reap()
            return dict(arena=arena, outputs=outputs, stats=stats, decoder_parity=parity,
                        source=source, frames=frames, requests=requests)
        except BaseException:
            pool.cancel()
            raise

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)


def _run(config, out, verify=False):
    if not torch.cuda.is_available():
        raise RuntimeError('the complete runtime requires an NVIDIA CUDA GPU')
    if out.exists():
        raise FileExistsError(f'refusing to overwrite an existing run: {out}')
    out.mkdir(parents=True)
    write(out/'status.json', dict(state='loading'))
    torch.set_grad_enabled(False); torch.set_num_threads(1)
    e = load(config)
    selector = Selection(e, config['occluder_cells'], config.get('cpu_threads', 1),
                         config.get('occlusion', True), config.get('occluder_level'))
    k, slots = config.get('group_size', 4), config.get('group_slots', 2)
    if k not in (1, 2, 3, 4) or not 1 <= slots <= 8:
        raise ValueError('supported K=1..4, B=1..8')
    batch_groups = config.get('batch_groups', math.ceil(len(e.views)/k))
    if batch_groups < 1:
        raise ValueError('batch_groups must be positive')
    lanes = [Lane(e, config.get('render_workers', 1)) for _ in range(slots)]
    workers = config.get('cpu_workers', 4), config.get('gpu_workers', 1)
    streams = __import__('threading').local()
    reports, parity, rendered = [], [], []
    def query(f, cpu):
        torch.set_grad_enabled(False)
        if not hasattr(streams, 'stream'):
            streams.stream = torch.cuda.Stream()
        with torch.cuda.stream(streams.stream):
            ids = torch.from_numpy(selector.select_cpu(f)).to('cuda') if cpu else selector.select_gpu(f)
            if ids.dtype != torch.long or (len(ids)>1 and not bool((ids[1:] > ids[:-1]).all())):
                raise RuntimeError('selection IDs must be increasing original IDs')
            event = torch.cuda.Event(); event.record()
        event.synchronize()
        return ids
    torch.cuda.synchronize()
    try:
        if verify:
            write(out/'status.json', dict(state='checking_all_cpu_gpu_selections'))
            for f in range(len(e.views)):
                a, b = query(f, True), query(f, False)
                exact = torch.equal(a, b)
                parity.append(dict(frame=f, cpu_ids=len(a), gpu_ids=len(b), exact=exact))
                if not exact:
                    write(out/'selection_parity.json', parity)
                    raise RuntimeError(f'CPU/GPU selected IDs differ at target {f}')
                del a, b
            write(out/'selection_parity.json', parity)
        policy = config.get('selection_device', 'gpu')
        if policy not in ('cpu', 'gpu', 'hybrid'):
            raise ValueError('selection_device must be cpu, gpu or hybrid')
        if policy == 'hybrid' and not all(name in config for name in ('cpu_cost', 'gpu_cost')):
            raise ValueError('hybrid requires measured cpu_cost/gpu_cost; no assumed rates')
        write(out/'status.json', dict(state='rendering', targets=len(e.views)))
        start = time.perf_counter()
        with ThreadPoolExecutor(workers[0]) as cp, ThreadPoolExecutor(workers[1]) as gp:
            for first in range(0, len(e.views), batch_groups*k):
                end = min(len(e.views), first+batch_groups*k)
                frames = list(range(first, end))
                assignment = allocate(len(frames), config['cpu_cost'], config['gpu_cost']) if policy == 'hybrid' else [policy == 'cpu']*len(frames)
                begin = time.perf_counter()
                futures = [(cp if cpu else gp).submit(query, f, cpu) for f, cpu in zip(frames, assignment)]
                selected = {f: future.result() for f, future in zip(frames, futures)}
                del futures
                barrier = time.perf_counter()  # All query/transfer completion events have finished.
                specs = [epoch_spec(s, k, len(e.views), True) for s in range(first, end, k)]
                pool = Admission(e.base.CAPACITY_ROWS)
                queue = deque(); next_group = 0
                def submit(j, lane):
                    spec = specs[j]
                    requests = [selected[f] for f in range(spec['start'], spec['end'])]
                    future = lanes[lane].executor.submit(lanes[lane].run, j, spec, requests, pool, verify)
                    queue.append((j, lane, future))
                for j in range(min(slots, len(specs))):
                    submit(j, j); next_group += 1
                try:
                    while queue:
                        j, lane, future = queue.popleft()
                        result = future.result()
                        arena = result['arena']
                        differences = []
                        for target, (f, output) in enumerate(zip(result['frames'], result['outputs'])):
                            if any(not bool(torch.isfinite(output[name]).all()) for name in ('render','render_alpha','render_depth')):
                                raise RuntimeError(f'non-finite renderer output at target {f}')
                            if verify:
                                batch = take_requests(arena.batch, arena.maps[target], result['requests'][target], e.rt.levels[result['requests'][target]])
                                reference = render_gdmgs_backend(e.views[f], batch, e.rt.background, 'RGB+ED')
                                delta = {}
                                for name in ('render', 'render_alpha', 'render_depth'):
                                    a, b = output[name], reference[name]
                                    if not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
                                        raise RuntimeError('non-finite image output')
                                    delta[name] = float((a-b).abs().max())
                                    if not torch.allclose(a, b, atol=1e-5, rtol=1e-5):
                                        raise RuntimeError(f'target {f} shared renderer differs: {delta}')
                                differences.append(dict(frame=f, max_abs=delta))
                                del batch, reference
                            if config.get('save_images', True):
                                from PIL import Image
                                image = (output['render'].detach().clamp(0, 1)*255).byte().permute(1,2,0).cpu().numpy()
                                Image.fromarray(image).save(out/f'{f:06d}.png')
                            rendered.append(f)
                            selected.pop(f)
                        reports.append(dict(start=arena.spec['start'], targets=len(result['frames']),
                                            union_anchors=arena.decoded_anchors, retained_rows=arena.decoded_rows,
                                            decoder_parity=result['decoder_parity'], images=differences))
                        del arena, result, future, output
                        pool.release(j)
                        if next_group < len(specs):
                            submit(next_group, lane); next_group += 1
                    assert not selected
                except BaseException:
                    pool.cancel()
                    raise
                reports.append(dict(batch_start=first, selection_ms=(barrier-begin)*1000,
                                    cpu_queries=sum(assignment), gpu_queries=len(frames)-sum(assignment),
                                    capacity=pool.summary()))
                write(out/'progress.json', dict(rendered=len(rendered), expected=len(e.views)))
        if rendered != list(range(len(e.views))):
            raise RuntimeError('incomplete or reordered trajectory')
        report = dict(state='complete', targets=len(rendered), groups=math.ceil(len(rendered)/k),
                      verification=verify, wall_seconds=time.perf_counter()-start,
                      timing_scope='validation and image saving included; not benchmark FPS',
                      config=config, selection_parity=parity, reports=reports,
                      camera_names=[v.image_name for v in e.views],
                      torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name())
        write(out/'report.json', report)
        write(out/'status.json', dict(state='complete', targets=len(rendered), verification=verify))
        return report
    except BaseException as error:
        write(out/'status.json', dict(state='failed', error=repr(error), rendered=len(rendered)))
        raise
    finally:
        for lane in lanes:
            lane.close()


def run(config, out, verify=False):
    out = Path(out)
    try:
        return _run(config, out, verify)
    except FileExistsError:
        raise  # Never modify a pre-existing output directory.
    except BaseException as error:
        if out.is_dir():
            status = out/'status.json'
            previous = json.loads(status.read_text()) if status.exists() else {}
            previous.update(state='failed', error=repr(error))
            write(status, previous)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    run(config, args.output, args.verify)
