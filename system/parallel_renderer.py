"""Current-camera split gsplat rendering of an immutable epoch generation.

No mesh/decoder concurrency and no reuse of previous-frame intersection ordering.
Streams overlap is a scheduling option, not a claim of observed GPU overlap.
"""
import math
import torch
from gsplat.cuda._wrapper import (
    fully_fused_projection, isect_tiles, isect_offset_encode, rasterize_to_pixels,
)


class ParallelRenderer:
    def __init__(self, views, background, requests_sorted=False):
        self.views = views
        self.background = background
        self.device = background.device
        # True requires sorted request certificates, validated by the caller's
        # GPU producer/oracle protocol; avoids an otherwise redundant sort.
        self.requests_sorted = bool(requests_sorted)
        self.width = int(views[0].image_width)
        self.height = int(views[0].image_height)
        self.tile_size = 16
        self.tile_width = math.ceil(self.width / self.tile_size)
        self.tile_height = math.ceil(self.height / self.tile_size)
        self.viewmats = []
        self.intrinsics = []
        for view in views:
            if (view.image_width, view.image_height) != (self.width, self.height):
                raise ValueError('all target cameras must have equal resolution')
            fx = self.width / (2 * math.tan(view.FoVx * .5))
            fy = self.height / (2 * math.tan(view.FoVy * .5))
            self.viewmats.append(view.world_view_transform.T.contiguous())
            self.intrinsics.append(torch.tensor(
                [[fx, 0, self.width / 2], [0, fy, self.height / 2], [0, 0, 1]],
                dtype=torch.float32, device=self.device))
        self.workers = [torch.cuda.Stream(device=self.device) for _ in range(4)]
        self._pending = []
        self.last_stats = {}

    @staticmethod
    def _batch_tensors(batch):
        # The renderer reads only these fields. Keep the entire batch strongly
        # referenced, and record the allocations read on a different stream.
        return [getattr(batch, name) for name in
                ('xyz', 'rotation', 'scaling', 'opacity', 'color')]

    def _reap(self):
        self._pending = [item for item in self._pending
                         if not all(event.query() for event in item['done'])]

    @staticmethod
    def _validate(batches, frame_ids):
        if len(batches) != len(frame_ids) or not frame_ids:
            raise ValueError('one input per nonempty target-camera list required')
        if len(set(frame_ids)) != len(frame_ids):
            raise ValueError('duplicate target frame')

    def _membership(self, owner, ids):
        # Dataset request certificates are sorted; sorting also makes this API
        # safe for independently constructed, unordered requests.
        if len(ids) == 0:
            return torch.zeros_like(owner, dtype=torch.bool)
        if not self.requests_sorted:
            ids = torch.sort(ids).values
        pos = torch.searchsorted(ids, owner)
        return (pos < len(ids)) & (ids[pos.clamp(max=len(ids)-1)] == owner)

    def _render(self, batch, frame_ids, depth, request_ids=None):
        count = len(frame_ids)
        mats = torch.stack([self.viewmats[f] for f in frame_ids])
        kin = torch.stack([self.intrinsics[f] for f in frame_ids])
        rad, means, z, conics, _ = fully_fused_projection(
            batch.xyz, None, batch.rotation, batch.scaling, mats, kin,
            self.width, self.height, packed=False)
        valid = rad > 0
        if request_ids is not None:
            owner = batch.bundle_metadata.row_owner_ids
            selected = torch.stack([self._membership(owner, ids) for ids in request_ids])
            valid = valid & selected
        rad = torch.where(valid, rad, 0).contiguous()
        # gsplat leaves culled rows uninitialized. They must be sanitized before
        # a batched arithmetic operation, even when radii subsequently mask them.
        means = torch.where(valid[..., None], means, 0).contiguous()
        conics = torch.where(valid[..., None], conics, 0).contiguous()
        z = torch.where(valid, z, 0).contiguous()
        opacity = torch.where(valid, batch.opacity.reshape(1, -1), 0).contiguous()
        _, keys, flat = isect_tiles(means, rad, z, self.tile_size,
            self.tile_width, self.tile_height, sort=True, packed=False,
            n_cameras=count)
        offsets = isect_offset_encode(keys, count, self.tile_width, self.tile_height)
        colors = batch.color[None].expand(count, -1, -1)
        bg = self.background[None].expand(count, -1)
        if depth:
            colors = torch.cat((colors, z[..., None]), -1)
            bg = torch.cat((bg, bg.new_zeros((count, 1))), -1)
        rgb, alpha = rasterize_to_pixels(means, conics, colors.contiguous(), opacity,
            self.width, self.height, self.tile_size, offsets, flat,
            backgrounds=bg.contiguous(), packed=False)
        ed = rgb[..., -1:] / alpha.clamp(min=1e-10) if depth else None
        outputs = [dict(render=rgb[j, ..., :3].permute(2, 0, 1),
                        render_alpha=alpha[j].permute(2, 0, 1),
                        render_depth=ed[j].permute(2, 0, 1) if depth else None)
                   for j in range(count)]
        stats = dict(frame_ids=list(frame_ids), cameras=count,
                     projected_rows=count * len(batch.xyz),
                     intersection_entries=flat.numel())
        return outputs, stats

    @torch.no_grad()
    def serial(self, compact_batches, frame_ids, depth=True):
        self._validate(compact_batches, frame_ids)
        self._reap()
        out, stats = [], []
        for batch, frame in zip(compact_batches, frame_ids):
            result, stat = self._render(batch, [frame], depth)
            out.extend(result)
            stats.append(stat)
        self.last_stats = dict(mode='serial', targets=stats)
        return out

    @torch.no_grad()
    def streams(self, compact_batches, frame_ids, depth=True, n_streams=2):
        self._validate(compact_batches, frame_ids)
        if n_streams not in (2, 4):
            raise ValueError('n_streams must be 2 or 4')
        self._reap()
        caller = torch.cuda.current_stream(self.device)
        publication = torch.cuda.Event()
        # Caller has completed scheduling every target's compact assembly before
        # publishing this single immutable generation.
        publication.record(caller)
        outputs, stats, done = [], [], []
        for j, (batch, frame) in enumerate(zip(compact_batches, frame_ids)):
            worker = self.workers[j % n_streams]
            worker.wait_event(publication)
            with torch.cuda.stream(worker):
                for tensor in self._batch_tensors(batch) + [self.background,
                        self.viewmats[frame], self.intrinsics[frame]]:
                    tensor.record_stream(worker)
                result, stat = self._render(batch, [frame], depth)
                for tensor in result[0].values():
                    if tensor is not None:
                        tensor.record_stream(worker)
                completion = torch.cuda.Event()
                completion.record(worker)
            outputs.extend(result)
            stat['worker_index'] = j % n_streams
            stat['cuda_stream'] = int(worker.cuda_stream)
            stats.append(stat)
            done.append(completion)
        for completion in done:
            caller.wait_event(completion)
        # All subsequent caller work is ordered after every worker. Inputs and
        # outputs remain strongly held until events report actual completion.
        for output in outputs:
            for tensor in output.values():
                if tensor is not None:
                    tensor.record_stream(caller)
        self._pending.append(dict(done=done, publication=publication,
            inputs=list(compact_batches), outputs=outputs))
        self.last_stats = dict(mode='streams', n_streams=n_streams, targets=stats,
            publication_event=True, caller_join_events=len(done),
            host_completion_wait=False,
            overlap_claim='scheduled; verify kernel overlap with profiler')
        return outputs

    @torch.no_grad()
    def batched(self, arena_batch, request_ids_list, frame_ids, depth=True):
        self._validate(request_ids_list, frame_ids)
        self._reap()
        out, stats = self._render(arena_batch, frame_ids, depth, request_ids_list)
        self.last_stats = dict(mode='batched_union', targets=[stats],
                              union_rows=len(arena_batch.xyz))
        return out

    @torch.no_grad()
    def serial_union(self, arena_batch, request_ids_list, frame_ids, depth=True):
        self._validate(request_ids_list, frame_ids)
        self._reap()
        out, stats = [], []
        for ids, frame in zip(request_ids_list, frame_ids):
            result, stat = self._render(arena_batch, [frame], depth, [ids])
            out.extend(result)
            stats.append(stat)
        self.last_stats = dict(mode='serial_union', targets=stats,
                              union_rows=len(arena_batch.xyz))
        return out
