"""Complete-group batch barrier followed by bounded materialize/render lanes."""
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass

from .camera import source_camera


def cpu_quota(n, cpu_cost, gpu_cost):
    """Floor/ceil optimum of max(c_C*x, c_G*(n-x)); costs include ID transfer."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError('n must be a nonnegative integer')
    if not all(math.isfinite(c) and c > 0 for c in (cpu_cost, gpu_cost)):
        raise ValueError('effective per-target costs must be finite and positive')
    x = n * gpu_cost / (cpu_cost + gpu_cost)
    return min({math.floor(x), math.ceil(x)},
               key=lambda c: (max(cpu_cost*c, gpu_cost*(n-c)), c))


def assignment(n, quota):
    if not 0 <= quota <= n:
        raise ValueError('quota outside batch')
    return [((i+1)*quota)//n > (i*quota)//n for i in range(n)]


class Admission:
    """FIFO exact retained-row reservations with cancellation on any failure."""
    def __init__(self, capacity):
        self.capacity, self.next = capacity, 0
        self.rows = {}
        self.peak = 0
        self.cancelled = False
        self.condition = threading.Condition()

    def acquire(self, index, rows):
        if rows > self.capacity or rows < 0:
            raise ValueError('single group exceeds retained-row capacity')
        with self.condition:
            while not self.cancelled and (index != self.next or sum(self.rows.values()) + rows > self.capacity):
                self.condition.wait()
            if self.cancelled:
                raise RuntimeError('group execution cancelled')
            self.rows[index] = rows
            self.next += 1
            self.peak = max(self.peak, sum(self.rows.values()))
            self.condition.notify_all()

    def release(self, index):
        with self.condition:
            del self.rows[index]
            self.condition.notify_all()

    def cancel(self):
        with self.condition:
            self.cancelled = True
            self.condition.notify_all()


@dataclass(frozen=True)
class Schedule:
    group_size: int = 4
    batch_groups: int = 8
    cpu_workers: int = 4
    gpu_workers: int = 1
    group_slots: int = 2
    render_workers: int = 1
    capacity_rows: int = 6_826_846

    def __post_init__(self):
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in vars(self).values()):
            raise ValueError('all schedule limits must be positive integers')

    def run(self, cameras, cpu_select, gpu_select, materializer, render,
            *, cpu_cost=1., gpu_cost=1., device='cpu'):
        """render(camera, SharedRows, target) must finish work before returning.

        Selectors must return canonical IDs for the same predicate/LoD policy.
        Rendering must read shared tensors without mutating them, perform target
        projection/sorting, and return independent outputs (prefer host images).
        Costs are measured at the configured worker counts before this call.
        """
        import torch
        from .materialization import GroupMaterializer
        if not isinstance(materializer, GroupMaterializer):
            raise TypeError('expected GroupMaterializer')
        device = torch.device(device)
        outputs, diagnostics = [], []

        def context():
            return torch.cuda.stream(torch.cuda.Stream(device=device)) if device.type == 'cuda' else nullcontext()

        def complete():
            if device.type == 'cuda':
                torch.cuda.current_stream(device).synchronize()

        def select(camera, cpu):
            with context(), torch.no_grad():
                raw = (cpu_select if cpu else gpu_select)(camera)
                ids = torch.as_tensor(raw, device=device)
                if ids.dtype != torch.long or ids.ndim != 1:
                    raise ValueError('selector must return a rank-one int64 ID vector')
                ids = ids.contiguous()
                if bool((ids < 0).any()) or (len(ids) > 1 and not bool((ids[1:] > ids[:-1]).all())):
                    raise ValueError('selector IDs must be nonnegative, sorted and unique')
                complete()  # Includes CPU-produced ID transfer before barrier.
                return ids

        if device.type == 'cuda':
            torch.cuda.current_stream(device).synchronize()  # Publish read-only model/index inputs.
        groups = [cameras[s:s+self.group_size] for s in range(0, len(cameras), self.group_size)]
        with ThreadPoolExecutor(self.cpu_workers) as cp, ThreadPoolExecutor(self.gpu_workers) as gp:
            for begin in range(0, len(groups), self.batch_groups):
                current = groups[begin:begin+self.batch_groups]
                views = [v for group in current for v in group]
                quota = len(views) if gpu_select is None else (0 if cpu_select is None else cpu_quota(len(views), cpu_cost, gpu_cost))
                if cpu_select is None and gpu_select is None:
                    raise ValueError('at least one selector is required')
                futures = [(cp if cpu else gp).submit(select, view, cpu)
                           for view, cpu in zip(views, assignment(len(views), quota))]
                selected = [f.result() for f in futures]
                del futures
                # Barrier: every Selection and ID transfer completed above.
                pool = Admission(self.capacity_rows)
                requests, cursor = [], 0
                for group in current:
                    requests.append(selected[cursor:cursor+len(group)])
                    cursor += len(group)
                stored_id_bytes = sum(t.numel()*t.element_size() for t in selected)
                del selected

                def execute(index):
                    shared = prepared = None
                    try:
                        with context(), torch.no_grad():
                            prepared, rows = materializer.prepare(source_camera(current[index]), requests[index])
                            pool.acquire(index, rows)
                            shared = materializer.finish(prepared)
                            prepared = None
                            complete()  # Publish shared rows to target streams.

                            def target(j):
                                with context(), torch.no_grad():
                                    result = render(current[index][j], shared, j)
                                    complete()
                                    return result

                            with ThreadPoolExecutor(self.render_workers) as rp:
                                result = list(rp.map(target, range(len(current[index]))))
                            shared = None
                            requests[index].clear()
                            pool.release(index)
                            return result
                    except BaseException:
                        pool.cancel()
                        raise

                with ThreadPoolExecutor(self.group_slots) as lanes:
                    pending = [lanes.submit(execute, j) for j in range(len(current))]
                    for future in pending:
                        outputs.extend(future.result())
                diagnostics.append(dict(targets=len(views), cpu_queries=quota,
                                        selected_id_bytes=stored_id_bytes, peak_retained_rows=pool.peak))
        return outputs, diagnostics
