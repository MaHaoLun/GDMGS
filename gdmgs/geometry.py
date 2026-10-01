"""Reference geometry for a conservative mesh-cell holed-frustum query.

All culling uses an inward tolerance: uncertain boxes remain selected.  The
offline cell table is produced by occupancy.py, which over-marks mesh
boundary and camera-reachable free cells.
"""
import json
from pathlib import Path

import numpy as np


def boxes_from_probe(path):
    data = json.loads(Path(path).read_text())[0]
    lo = np.asarray(data['domain_lo'], np.float64)
    hi = np.asarray(data['domain_hi'], np.float64)
    dim = 1 << data['level']
    boxes = []
    for level, x, y, z in data['full_nodes']:
        size = (hi - lo) / (1 << level)
        lower = lo + np.array((x, y, z)) * size
        upper = lower + size
        boxes.append(np.r_[lower, upper])
    return np.asarray(boxes, np.float64).reshape(-1, 6), data


def box_plane_min(box, plane):
    n = plane[:3]
    return np.dot(n, np.where(n >= 0, box[:3], box[3:])) + plane[3]


def box_plane_max(box, plane):
    n = plane[:3]
    return np.dot(n, np.where(n >= 0, box[3:], box[:3])) + plane[3]


def visible_occluders(boxes, camera_planes, w2c, near, tolerance=1e-9):
    """Return full cells that may meet the view and lie wholly past near."""
    if not len(boxes):
        return boxes
    out = []
    near_plane = np.r_[w2c[2, :3], w2c[2, 3] - near]
    for box in boxes:
        if box_plane_min(box, near_plane) <= tolerance:
            continue
        if any(box_plane_max(box, p) < -tolerance for p in camera_planes):
            continue
        out.append(box)
    return np.asarray(out, np.float64).reshape(-1, 6)


def hole_planes(box, eye, tolerance=1e-12):
    """Convex cone through the box plus its camera-facing entry faces.

    Each cone plane passes through the eye and one silhouette edge.  A support
    plane is retained only if all eight corners lie on its inner side.  The
    resulting planes are an exact polyhedral description in real arithmetic.
    """
    box = np.asarray(box, np.float64)
    eye = np.asarray(eye, np.float64)
    corners = np.array([[box[a] if i == 0 else box[a + 3]
                         for a, i in enumerate(bits)]
                        for bits in np.ndindex(2, 2, 2)], np.float64)
    rays = corners - eye
    planes = []
    for i in range(8):
        for j in range(i + 1, 8):
            n = np.cross(rays[i], rays[j])
            norm = np.linalg.norm(n)
            if norm < tolerance:
                continue
            n /= norm
            sides = rays @ n
            if sides.min() < -tolerance and sides.max() > tolerance:
                continue
            if sides.max() <= tolerance:
                n = -n
            if not any(np.allclose(n, p[:3], atol=1e-10, rtol=0) for p in planes):
                planes.append(np.r_[n, -np.dot(n, eye)])
    for axis in range(3):
        normal = np.zeros(3)
        if eye[axis] < box[axis]:
            normal[axis] = 1
            planes.append(np.r_[normal, -box[axis]])
        elif eye[axis] > box[axis + 3]:
            normal[axis] = -1
            planes.append(np.r_[normal, box[axis + 3]])
    return np.asarray(planes, np.float64).reshape(-1, 4)


def inside_hole(bounds, planes, tolerance=1e-8):
    """Boolean mask: every corner of each AABB is safely inside one hole."""
    bounds = np.asarray(bounds, np.float64)
    if not len(bounds):
        return np.zeros(0, bool)
    inside = np.ones(len(bounds), bool)
    for p in planes:
        corner = np.where(p[:3] >= 0, bounds[:, :3], bounds[:, 3:])
        minimum = corner @ p[:3] + p[3]
        inside &= minimum > tolerance
    return inside


def prune_bounds(bounds, holes, eye, tolerance=1e-8):
    """Vectorized oracle for the paper's single-hole exclusion predicate."""
    prune = np.zeros(len(bounds), bool)
    for cell in holes:
        planes = hole_planes(cell, eye)
        prune |= inside_hole(bounds, planes, tolerance)
    return prune
