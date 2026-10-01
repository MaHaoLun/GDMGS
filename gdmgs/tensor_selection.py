"""Portable tensor implementation of the same predicates on CPU or CUDA.

This is a correctness backend, not the historical optimized CUDA kernel.
Traversal is host-driven and scalar decisions synchronize on CUDA. All box
tests, hole construction and ID buffers live on the requested tensor device.
"""
import numpy as np
import torch


def extrema(box, planes):
    normals = planes[:, :3]
    lower = torch.where(normals >= 0, box[:3], box[3:])
    upper = torch.where(normals >= 0, box[3:], box[:3])
    return (lower * normals).sum(1) + planes[:, 3], (upper * normals).sum(1) + planes[:, 3]


def hole_planes(box, eye):
    corners = torch.stack([torch.stack([box[a + 3 * bit] for a, bit in enumerate(bits)])
                           for bits in np.ndindex(2, 2, 2)])
    rays = corners - eye
    planes = []
    for i in range(8):
        for j in range(i + 1, 8):
            normal = torch.linalg.cross(rays[i], rays[j])
            norm = torch.linalg.vector_norm(normal)
            if bool(norm < 1e-12):
                continue
            normal = normal / norm
            sides = rays @ normal
            if bool((sides.min() < -1e-12) & (sides.max() > 1e-12)):
                continue
            if bool(sides.max() <= 1e-12):
                normal = -normal
            if not any(bool(torch.all(torch.abs(normal - p[:3]) <= 1e-10)) for p in planes):
                planes.append(torch.cat((normal, -(normal @ eye).reshape(1))))
    for axis in range(3):
        normal = torch.zeros(3, dtype=box.dtype, device=box.device)
        if bool(eye[axis] < box[axis]):
            normal[axis] = 1
            planes.append(torch.cat((normal, -box[axis:axis+1])))
        elif bool(eye[axis] > box[axis+3]):
            normal[axis] = -1
            planes.append(torch.cat((normal, box[axis+3:axis+4])))
    return torch.stack(planes)


class TensorSelector:
    def __init__(self, anchors, occluders=None, device='cuda', eligibility=None):
        from .selection import CPUSelector
        CPUSelector(anchors, occluders)  # Validate the common coordinate domain.
        self.index, self.occluders = anchors, occluders
        self.device, self.eligibility = torch.device(device), eligibility
        self.bounds = self.tensor(anchors.bounds)
        self.nodes = self.tensor(anchors.nodes)
        self.order = torch.tensor(anchors.order.copy(), device=self.device)
        self.lo, self.hi = self.tensor(anchors.domain_lo), self.tensor(anchors.domain_hi)

    def tensor(self, value):
        return torch.tensor(np.array(value, copy=True), dtype=torch.float64, device=self.device)

    def __call__(self, camera):
        # No mutable query scratch is stored in the selector.
        planes, eye = self.tensor(camera.planes), self.tensor(camera.center)
        near = self.tensor(np.r_[camera.w2c[2, :3], camera.w2c[2, 3] - camera.near]).reshape(1, 4)
        holes = []

        def retrieve(node, depth, xyz, full=False):
            size = (self.hi - self.lo) / (1 << depth)
            lower = self.lo + self.tensor(xyz) * size
            box = torch.cat((lower, lower + size))
            if bool((extrema(box, planes)[1] < -1e-9).any()):
                return
            full = full or node.full
            if full and bool((extrema(box, near)[0] > 1e-9).all()):
                holes.append(hole_planes(box, eye))
                return
            if depth == self.occluders.level:
                return
            for child in (range(8) if full else node.children):
                bits = ((child >> 2) & 1, (child >> 1) & 1, child & 1)
                retrieve(None if full else node.children[child], depth + 1,
                         tuple(2 * xyz[a] + bits[a] for a in range(3)), full)

        if self.occluders is not None:
            retrieve(self.occluders.root, 0, (0, 0, 0))
        result = []

        def walk(node, active):
            box = self.nodes[node]
            minimum, maximum = extrema(box, planes)
            if bool((maximum < -1e-9).any()):
                return
            nearby = []
            for hole in active:
                low, high = extrema(box, hole)
                if bool((low > 1e-8).all()):
                    return
                if not bool((high < -1e-8).any()):
                    nearby.append(hole)
            begin, end = self.index.intervals[node]
            if bool((minimum > 1e-9).all()) and not nearby:
                result.append(self.order[begin:end])
            elif self.index.left[node] < 0:
                ids = self.order[begin:end]
                # Index positions are host metadata, original IDs remain device-side.
                for j in range(len(ids)):
                    bound = self.bounds[ids[j]]
                    if bool((extrema(bound, planes)[1] < -1e-9).any()):
                        continue
                    if any(bool((extrema(bound, hole)[0] > 1e-8).all()) for hole in nearby):
                        continue
                    result.append(ids[j:j+1])
            else:
                walk(int(self.index.left[node]), nearby)
                walk(int(self.index.right[node]), nearby)

        if len(self.index.nodes):
            walk(0, holes)
        ids = torch.cat(result).sort().values if result else torch.empty(0, dtype=torch.long, device=self.device)
        if self.eligibility is not None:
            eligible = self.eligibility(camera).to(self.device)
            ids = ids[torch.isin(ids, eligible)]
        if self.device.type == 'cuda':
            torch.cuda.current_stream(self.device).synchronize()
        return ids
