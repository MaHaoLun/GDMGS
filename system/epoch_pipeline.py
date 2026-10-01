"""Bounded two-generation schedule; the existing producer stays on CUDA default.

render_epoch(arena, requests, frames, depth) must return (outputs, owned_stats).
It runs on one render stream and must join any child streams onto that stream.
No callback may retain an arena after returning beyond properly recorded work.
All producer/native synchronization is preserved. This is scheduling, not an
assertion of concurrent kernel execution or an exemption from the cache budget.
"""
from collections import deque
import copy
import time


def _tensors(torch, arena):
    seen = set()
    for obj in (arena.batch, arena.batch.bundle_metadata):
        for value in vars(obj).values():
            if isinstance(value, torch.Tensor) and id(value) not in seen:
                seen.add(id(value))
                yield value
    for value in list(arena.requests) + list(arena.maps):
        if id(value) not in seen:
            seen.add(id(value))
            yield value


def pipeline_sequence(e, render_epoch, quality, *, total=125, k=4, starts=None,
                      producer_delay_cycles=0, render_delay_cycles=0,
                      render_tail_delay_cycles=0,
                      on_epoch=None, capacity_rows=None, mode='center4_pipeline',
                      overlap=True, release_completed=None):
    """Online producer + one independent render lane with at most two arenas.

    starts is exclusively a bounded-contract-test hook; production uses all
    range(0, total, k) groups. on_epoch(arena, outputs, owned_stats) runs after
    that generation completes, and must not retain arena tensors. The function
    excludes oracle comparisons and final guard from wall_ms, like the baseline.
    Conservative row reservations happen BEFORE decoder allocation. If two
    generations cannot fit, the older generation is drained before decoding.
    """
    from epoch_cache import EpochArena, epoch_spec, source_view
    torch = e.torch
    if total != len(e.views) or k < 1:
        raise ValueError('total must match immutable target trajectory')
    starts = list(range(0, total, k)) if starts is None else list(starts)
    specs = [epoch_spec(s, k, total, True) for s in starts]
    expected = [f for spec in specs for f in range(spec['start'], spec['end'])]
    if expected != sorted(set(expected)):
        raise ValueError('groups must be ordered and nonoverlapping')
    capacity = int(e.base.CAPACITY_ROWS if capacity_rows is None else capacity_rows)
    if capacity < 1:
        raise ValueError('positive row budget required')
    device = e.rt.background.device
    producer = torch.cuda.default_stream(device)
    worker = torch.cuda.Stream(device=device)
    pending = deque()
    checks, epochs, metrics, delivered = [], [], [], []
    stats = dict(capacity_rows=capacity, max_live_generations=0,
                 max_live_cache_rows=0, max_live_cache_bytes=0,
                 max_reserved_plus_live_rows=0, capacity_waits=0,
                 capacity_wait_ms=0., slot_waits=0, publication_count=0,
                 completed_reaps=0, producer_stream=int(producer.cuda_stream),
                 render_stream=int(worker.cuda_stream),
                 admission='live rows plus online union anchors times n_offsets',
                 producer_adapter_attached=('_mesh' in vars(e.rt)), overlap_enabled=bool(overlap),
                 max_producer_peak_allocated_bytes=0)
    first_ready = None
    e.guard()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()

    def drain(reason):
        nonlocal first_ready
        item = pending.popleft()
        before = time.perf_counter()
        item['done'].synchronize()
        if reason == 'capacity':
            stats['capacity_waits'] += 1
            stats['capacity_wait_ms'] += (time.perf_counter() - before) * 1000
        elif reason == 'slot':
            stats['slot_waits'] += 1
        elif reason == 'complete':
            stats['completed_reaps'] += 1
        if first_ready is None:
            first_ready = (time.perf_counter() - begin) * 1000
        arena, outputs, record = item['arena'], item['outputs'], item['record']
        for output in outputs:
            for value in output.values():
                if isinstance(value, torch.Tensor):
                    value.record_stream(producer)
        frames = list(range(arena.spec['start'], arena.spec['end']))
        if quality:
            metrics.extend(e.metrics(f, output) for f, output in zip(frames, outputs))
        if on_epoch is not None:
            on_epoch(arena, outputs, record['renderer_stats'])
        delivered.extend(frames)
        epochs.append(record)
        if release_completed is not None:
            release_completed()
        # item/arena/outputs expire on return; no historical arena snapshots.

    try:
        with torch.cuda.stream(producer):
            for spec in specs:
                while pending and pending[0]['done'].query():
                    drain('complete')
                while len(pending) >= 2:
                    drain('slot')
                frames = list(range(spec['start'], spec['end']))
                requests = []
                if producer_delay_cycles:
                    torch.cuda._sleep(producer_delay_cycles)
                for frame in frames:
                    timeline = e.base.Timeline()
                    mesh, mesh_ms = e.rt._mesh(frame, timeline)
                    ids, _ = e.rt._gpu_finish_selection(frame, mesh, mesh_ms, timeline)
                    requests.append(ids)
                    checks.append((frame, ids))
                    e.rt.pending_checks.clear()
                    e.rt.pending_mesh_checks.clear()
                # The online union is used only for a hard upper bound. It is
                # not a historical predicted row count or an oracle request.
                reservation = int(torch.unique(torch.cat(requests)).numel()) * int(e.rt.model.n_offsets)
                if reservation > capacity:
                    raise RuntimeError('single-generation conservative bound exceeds cache budget')
                while pending and pending[0]['done'].query():
                    drain('complete')
                while pending and sum(x['arena'].decoded_rows for x in pending) + reservation > capacity:
                    drain('capacity')
                live_rows = sum(x['arena'].decoded_rows for x in pending)
                stats['max_reserved_plus_live_rows'] = max(stats['max_reserved_plus_live_rows'], live_rows + reservation)
                view = source_view(e, spec)
                def decode(ids, levels):
                    e.rt.model.set_anchor_mask(view.camera_center, 40000, view.resolution_scale)
                    return e.decode_batch(view, e.rt.model, ids, levels)
                arena = EpochArena(spec, requests, e.rt.levels, decode, capacity)
                if arena.overflow or arena.decoded_rows > reservation:
                    raise RuntimeError('decoder violated reserved row bound')
                stats['max_producer_peak_allocated_bytes'] = max(stats['max_producer_peak_allocated_bytes'], torch.cuda.max_memory_allocated())
                publication = torch.cuda.Event()
                publication.record(producer)
                worker.wait_event(publication)
                with torch.cuda.stream(worker):
                    for tensor in _tensors(torch, arena):
                        tensor.record_stream(worker)
                    if render_delay_cycles:
                        torch.cuda._sleep(render_delay_cycles)
                    outputs, owned = render_epoch(arena, requests, frames, quality)
                    if len(outputs) != len(frames):
                        raise RuntimeError('renderer omitted target frame')
                    # Freeze host statistics now, before the next callback.
                    owned = copy.deepcopy(owned)
                    # Test-only asynchronous tail: unlike a pre-render sleep,
                    # the host does not subsequently wait on isect .item().
                    if render_tail_delay_cycles:
                        torch.cuda._sleep(render_tail_delay_cycles)
                    done = torch.cuda.Event()
                    done.record(worker)
                record = dict(spec, union_anchors=arena.decoded_anchors,
                              cache_rows=arena.decoded_rows, cache_bytes=arena.bytes,
                              decoder_calls=arena.decoder_calls,
                              renderer_stats=owned, reservation_rows=reservation,
                              publication_frame=spec['start'], frame_ids=frames,
                              additional_compact_payload_bytes=0,
                              projected_rows=sum(t['projected_rows'] for t in owned.get('targets', [])),
                              intersection_entries=sum(t['intersection_entries'] for t in owned.get('targets', [])),
                              logical_target_sorts=len(frames))
                pending.append(dict(arena=arena, outputs=outputs, record=record,
                                    done=done, publication=publication))
                stats['publication_count'] += 1
                stats['max_live_generations'] = max(stats['max_live_generations'], len(pending))
                stats['max_live_cache_rows'] = max(stats['max_live_cache_rows'], sum(x['arena'].decoded_rows for x in pending))
                stats['max_live_cache_bytes'] = max(stats['max_live_cache_bytes'], sum(x['arena'].bytes for x in pending))
                if stats['max_live_cache_rows'] > capacity:
                    raise RuntimeError('two-generation cache budget exceeded')
                del arena, outputs, view, requests, decode
                if not overlap:
                    while pending:
                        drain('serial')
            while pending:
                drain('final')
        torch.cuda.synchronize()
    except BaseException:
        # All in-flight inputs stay referenced through synchronization, then
        # exceptions propagate. No failed sequence yields a success record.
        torch.cuda.synchronize()
        pending.clear()
        if release_completed is not None:
            release_completed()
        raise
    wall = (time.perf_counter() - begin) * 1000
    assert delivered == expected
    assert [f for f, _ in checks] == expected
    assert all(torch.equal(ids, e.oracle_gpu[f]) for f, ids in checks)
    e.guard()
    return dict(mode=mode, frames=len(delivered), wall_ms=None if quality else wall,
                fps=None if quality else len(delivered) * 1000 / wall,
                metrics=metrics, numeric=[], epochs=epochs,
                quality_pass=all(q['safe'] for q in metrics) if quality else None,
                mechanical_pass=None, decoder_calls=sum(x['decoder_calls'] for x in epochs),
                decoded_anchors=sum(x['union_anchors'] for x in epochs),
                first_group_ready_wall_ms=first_ready,
                max_allocated_bytes=torch.cuda.max_memory_allocated(),
                max_cache_rows=stats['max_live_cache_rows'],
                max_additional_compact_payload_bytes=0,
                max_projected_rows=max((x['projected_rows'] for x in epochs), default=0),
                oracle_exact=True, pipeline=stats, gpu=e.guard_records[-2:])


class StreamScopedProducer:
    """Instance-only, audited clones replacing four timing barriers.

    The producer remains default-stream-only. This does not move nvdiffrast to
    another stream, remove completion events, patch torch, alter dependencies,
    or modify any historical source file. New method source and old hashes are
    returned in audit so every narrowed barrier is reviewable.
    """
    def __init__(self, runtime):
        import ast
        import hashlib
        import inspect
        import textwrap
        import types
        from pathlib import Path
        self.runtime = runtime
        self._bindings = []
        self.audit = dict(changes=[], producer='CUDA default stream only',
                          shared_torch_module_unchanged=True,
                          completion_events_unchanged=True)
        class Narrow(ast.NodeTransformer):
            def __init__(self):
                self.count = 0
            def visit_Call(self, node):
                self.generic_visit(node)
                if ast.unparse(node.func) == 'torch.cuda.synchronize':
                    self.count += 1
                    stream = ast.Call(func=ast.parse('torch.cuda.default_stream', mode='eval').body,
                                      args=node.args, keywords=node.keywords)
                    return ast.copy_location(ast.Call(func=ast.Attribute(value=stream, attr='synchronize', ctx=ast.Load()), args=[], keywords=[]), node)
                return node
        for instance, name, required in ((runtime, '_mesh', 2),
                                         (runtime, '_gpu_finish_selection', 1),
                                         (runtime.rasterizer, 'render', 1)):
            original = getattr(instance, name)
            fn = original.__func__
            # The legacy gpu_guard intentionally has no functools.wraps. Its
            # closure owns the original method, whose source still includes
            # @gpu_guard; recompilation below preserves the lock/decorator.
            if fn.__name__ == 'guarded':
                enclosed = inspect.getclosurevars(fn).nonlocals
                if set(enclosed) != {'method'}:
                    raise RuntimeError('unrecognized GPU guard closure')
                fn = enclosed['method']
            source = textwrap.dedent(inspect.getsource(fn))
            tree = ast.parse(source)
            narrow = Narrow()
            tree = narrow.visit(tree)
            ast.fix_missing_locations(tree)
            if narrow.count != required:
                raise RuntimeError(f'{name}: producer source changed; expected {required} timing barriers, found {narrow.count}')
            new_source = ast.unparse(tree)
            namespace = dict(fn.__globals__)
            exec(compile(tree, '<stream-scoped-producer:' + name + '>', 'exec'), namespace)
            clone = namespace[name]
            torch = fn.__globals__['torch']
            def guarded(this, *args, _clone=clone, _torch=torch, **kwargs):
                if _torch.cuda.current_stream() != _torch.cuda.default_stream():
                    raise RuntimeError('producer is qualified only on the default stream')
                return _clone(this, *args, **kwargs)
            self._bindings.append((instance, name, original, types.MethodType(guarded, instance), name in vars(instance)))
            path = Path(inspect.getsourcefile(fn))
            self.audit['changes'].append(dict(method=f'{type(instance).__name__}.{name}',
                source_file=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                original_method_sha256=hashlib.sha256(source.encode()).hexdigest(),
                narrowed_barriers=narrow.count, transformed_source=new_source))

    def __enter__(self):
        for instance, name, _, replacement, _ in self._bindings:
            setattr(instance, name, replacement)
        return self.audit

    def __exit__(self, *exc):
        for instance, name, original, _, had_instance_value in reversed(self._bindings):
            if had_instance_value:
                setattr(instance, name, original)
            else:
                delattr(instance, name)
