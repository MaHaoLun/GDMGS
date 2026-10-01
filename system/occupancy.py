"""Conservative mesh-cell occupancy diagnostic for the Amsterdam proxy mesh.

Triangle AABBs over-mark boundary cells. A 26-neighbour flood from the outside
and every training camera then deliberately over-marks free space.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage


def make_grid(vertices, faces, eyes, level):
    dim = 1 << level
    lo = np.minimum(vertices.min(0), eyes.min(0))
    hi = np.maximum(vertices.max(0), eyes.max(0))
    width = (hi - lo).max()
    centre = (lo + hi) * .5
    lo = centre - width * .55
    hi = centre + width * .55
    cell = (hi - lo) / dim
    boundary = np.zeros((dim, dim, dim), dtype=bool)

    # The whole triangle is inside its AABB. Slightly expand the AABB so a
    # triangle exactly on a grid plane cannot create a false solid cell.
    for begin in range(0, len(faces), 200_000):
        xyz = vertices[faces[begin:begin + 200_000]]
        lower = np.clip(np.floor((xyz.min(1) - lo - 1e-10 * width) / cell).astype(np.int16), 0, dim - 1)
        upper = np.clip(np.floor((xyz.max(1) - lo + 1e-10 * width) / cell).astype(np.int16), 0, dim - 1)
        span = upper - lower + 1
        maxspan = span.max(0)
        for i in range(int(maxspan[0])):
            for j in range(int(maxspan[1])):
                for k in range(int(maxspan[2])):
                    valid = (span[:, 0] > i) & (span[:, 1] > j) & (span[:, 2] > k)
                    c = lower[valid] + (i, j, k)
                    boundary[c[:, 0], c[:, 1], c[:, 2]] = True

    seeds = np.zeros_like(boundary)
    seeds[0, :, :] = seeds[-1, :, :] = True
    seeds[:, 0, :] = seeds[:, -1, :] = True
    seeds[:, :, 0] = seeds[:, :, -1] = True
    camera_cells = np.clip(np.floor((eyes - lo) / cell).astype(int), 0, dim - 1)
    for x, y, z in camera_cells:
        seeds[max(0, x-1):min(dim, x+2), max(0, y-1):min(dim, y+2), max(0, z-1):min(dim, z+2)] = True
    labels, component_count = ndimage.label(~boundary, structure=np.ones((3, 3, 3), bool))
    free_labels = np.unique(labels[seeds & ~boundary])
    free = np.isin(labels, free_labels) & (labels != 0)
    solid = ~(free | boundary)
    # An all-solid parent can replace its eight children. Counts include
    # maximal full nodes only, which become the query-time hole candidates.
    maximal = []
    full_nodes = []
    current = solid
    for depth in range(level, -1, -1):
        full = int(current.sum())
        maximal.append((depth, full))
        if depth:
            parent = current.reshape(dim >> (level-depth+1), 2,
                                     dim >> (level-depth+1), 2,
                                     dim >> (level-depth+1), 2).all((1, 3, 5))
            maximal[-1] = (depth, full - 8 * int(parent.sum()))
            if full:
                covered = np.repeat(np.repeat(np.repeat(parent, 2, 0), 2, 1), 2, 2)
                full_nodes.extend((depth, *map(int, xyz)) for xyz in np.argwhere(current & ~covered))
            current = parent
        elif full:
            full_nodes.append((0, 0, 0, 0))
    return dict(level=level, dim=dim, domain_lo=lo.tolist(), domain_hi=hi.tolist(),
                triangles=len(faces), cameras=len(eyes), components=int(component_count),
                boundary_cells=int(boundary.sum()),
                free_cells=int(free.sum()), solid_cells=int(solid.sum()),
                maximal_full_nodes=dict(maximal), full_nodes=full_nodes,
                camera_boundary_cells=int(boundary[tuple(camera_cells.T)].sum()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mesh', type=Path, required=True)
    p.add_argument('--cameras', type=Path, required=True)
    p.add_argument('--levels', type=int, nargs='+', default=[5, 6, 7])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    z = np.load(args.mesh)
    records = json.loads(args.cameras.read_text())
    eyes = np.array([np.linalg.solve(np.asarray(r['w2c'])[:3, :3], -np.asarray(r['w2c'])[:3, 3]) for r in records])
    results = [make_grid(z['vertices'], z['triangles'], eyes, level) for level in args.levels]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
