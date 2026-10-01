"""Coordinate-explicit depth surfaces with no interpolation over missing pixels.

The GS rasterizer samples pixels at (column + .5, row + .5). Depth is camera-z,
not ray length. Camera matrices map homogeneous world points to camera space.
These surfaces are geometry candidates, not claims about true scene geometry.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DepthFrame:
    depth: np.ndarray
    alpha: np.ndarray
    intrinsics: np.ndarray
    world_to_camera: np.ndarray
    frame_index: int

    def __post_init__(self):
        if self.depth.ndim != 2 or self.alpha.shape != self.depth.shape:
            raise ValueError("Depth and alpha must have matching [H,W] shapes")
        if self.intrinsics.shape != (3, 3) or self.world_to_camera.shape != (4, 4):
            raise ValueError("Camera calibration must be K[3,3], W2C[4,4]")
        if (not np.isfinite(self.intrinsics).all() or
                not np.isfinite(self.world_to_camera).all()):
            raise ValueError("Camera calibration is nonfinite")
        if not np.array_equal(self.world_to_camera[3], [0, 0, 0, 1]):
            raise ValueError("W2C must be an affine homogeneous transform")
        if self.intrinsics[0, 0] <= 0 or self.intrinsics[1, 1] <= 0:
            raise ValueError("Camera focal lengths must be positive")
        if not np.array_equal(self.intrinsics[2], [0, 0, 1]):
            raise ValueError("Invalid pinhole calibration")
        if self.intrinsics[0, 1] != 0 or self.intrinsics[1, 0] != 0:
            raise ValueError("Only the renderer's zero-skew pinhole intrinsics are supported")

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as value:
            return cls(value["depth"], value["alpha"], value["intrinsics"],
                       value["world_to_camera"], int(value["frame_index"]))


@dataclass(frozen=True)
class TriangleTable:
    vertices: np.ndarray
    faces: np.ndarray
    source_record: dict
    mesh_token: str

    def __post_init__(self):
        if self.vertices.dtype != np.float64 or self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices must be float64[V,3]")
        if self.faces.dtype != np.int64 or self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("faces must be int64[F,3]")
        if not np.isfinite(self.vertices).all():
            raise ValueError("Nonfinite mesh vertices")
        if self.faces.size and (self.faces.min() < 0 or self.faces.max() >= len(self.vertices)):
            raise ValueError("Triangle index outside the vertex table")
        if not isinstance(self.mesh_token, str) or not self.mesh_token:
            raise ValueError("Explicit nonempty mesh token required")


def validate_depth_source(table, manifest):
    """Bind depth-based geometry operations to the mesh's recorded source run."""
    record = table.source_record
    for field in ("model_path", "iteration"):
        if field not in record or field not in manifest or record[field] != manifest[field]:
            raise ValueError(f"Mesh and depth manifest {field} differ")
    if ("depth_run_token" in record and
            record["depth_run_token"] != manifest.get("run_token")):
        raise ValueError("Mesh and depth manifest run labels differ")


def backproject(frame, rows, columns, depth=None):
    """Backproject arbitrary pixel indices, including noninteger diagnostic grids."""
    rows, columns = np.broadcast_arrays(rows, columns)
    if depth is None:
        depth = frame.depth[rows.astype(np.int64), columns.astype(np.int64)]
    pixels = np.stack([columns + .5, rows + .5, np.ones_like(rows)], axis=-1)
    camera = (pixels @ np.linalg.inv(frame.intrinsics).T) * np.asarray(depth)[..., None]
    c2w = np.linalg.inv(frame.world_to_camera)
    return camera @ c2w[:3, :3].T + c2w[:3, 3]


def valid_depth(frame, alpha_threshold=.995):
    return (np.isfinite(frame.depth) & (frame.depth > 0) &
            np.isfinite(frame.alpha) & (frame.alpha >= alpha_threshold))


def grid_triangles(frame, *, stride=16, alpha_threshold=.995,
                   relative_depth_jump=.02):
    """Triangle patches whose whole source pixel rectangle is valid and continuous.

    All original pixels inside each candidate rectangle are checked, not only
    triangle corners. An incomplete boundary strip is left open. No holes are
    filled and no faces connect different views. Returned confidence is the
    minimum alpha over the source rectangle; source IDs are explicit face data.
    """
    from scipy.ndimage import maximum_filter, minimum_filter
    if stride < 1 or not 0 < alpha_threshold <= 1 or relative_depth_jump <= 0:
        raise ValueError("Invalid grid construction settings")
    height, width = frame.depth.shape
    rows, columns = np.arange(0, height, stride), np.arange(0, width, stride)
    if len(rows) < 2 or len(columns) < 2:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int64), np.empty(0)
    valid = valid_depth(frame, alpha_threshold)
    # A forward-looking stride+1 window includes both corner rows/columns.
    origin = -((stride + 1) // 2)
    minimum = minimum_filter(np.where(valid, frame.depth, 0), stride + 1,
                             origin=origin, mode="constant", cval=0)
    maximum = maximum_filter(np.where(valid, frame.depth, np.inf), stride + 1,
                             origin=origin, mode="constant", cval=np.inf)
    confidence = minimum_filter(np.where(valid, frame.alpha, 0), stride + 1,
                                origin=origin, mode="constant", cval=0)
    rr, cc = np.meshgrid(rows[:-1], columns[:-1], indexing="ij")
    low, high = minimum[rr, cc], maximum[rr, cc]
    keep = (low > 0) & np.isfinite(high) & (high - low <= relative_depth_jump * low)
    rr, cc = rr[keep], cc[keep]
    # Build independent quad vertices. This preserves source-pixel provenance
    # and avoids welding gaps between separate projected depth observations.
    pr = np.stack([rr, rr, rr + stride, rr + stride], axis=1)
    pc = np.stack([cc, cc + stride, cc, cc + stride], axis=1)
    vertices = backproject(frame, pr, pc).reshape(-1, 3).astype(np.float64)
    base = np.arange(len(rr), dtype=np.int64)[:, None] * 4
    faces = np.stack([base + [0, 1, 2], base + [1, 3, 2]], axis=1).reshape(-1, 3)
    return vertices, faces, np.repeat(confidence[rr, cc], 2)


def cross_view_support(points, frame, *, relative_tolerance=.02, alpha_threshold=.995):
    """Independent observation agreement and visible free-space conflict flags.

    Occluded or unobserved points remain unknown; they are not conflicts. This
    is a geometry diagnostic/filter and never a full-pixel occlusion certificate.
    """
    camera = points @ frame.world_to_camera[:3, :3].T + frame.world_to_camera[:3, 3]
    projected = camera @ frame.intrinsics.T
    front = camera[:, 2] > 0
    uv = np.zeros((len(points), 2), dtype=np.int64)
    uv[front] = np.floor(projected[front, :2] / projected[front, 2:3]).astype(np.int64)
    h, w = frame.depth.shape
    visible = front & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    indices = np.flatnonzero(visible)
    rows, cols = uv[indices, 1], uv[indices, 0]
    observed = frame.depth[rows, cols]
    usable = valid_depth(frame, alpha_threshold)[rows, cols]
    delta = camera[indices, 2] - observed
    tolerance = relative_tolerance * observed
    agrees = np.zeros(len(points), dtype=bool)
    conflict = np.zeros(len(points), dtype=bool)
    agrees[indices] = usable & (np.abs(delta) <= tolerance)
    conflict[indices] = usable & (delta < -tolerance)
    return agrees, conflict


def compact_triangles(vertices, faces):
    """Remove only invalid-area faces and unused vertices; never add geometry."""
    if not len(faces):
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=bool)
    corners = vertices[faces]
    area = np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
    keep = np.isfinite(area) & (area > 0)
    faces = faces[keep]
    used, inverse = np.unique(faces, return_inverse=True)
    return vertices[used], inverse.reshape(-1, 3).astype(np.int64), keep


def weld_same_source_vertices(vertices, faces, source_frame):
    """Share identical vertices within one depth view without adding any surface."""
    if source_frame.dtype != np.int64 or source_frame.shape != (len(faces),):
        raise ValueError("Source frame must be an int64 per-face table")
    source_vertex = np.full(len(vertices), -1, dtype=np.int64)
    source_vertex[faces.reshape(-1)] = np.repeat(source_frame, 3)
    if np.any(source_vertex[faces] != source_frame[:, None]):
        raise ValueError("Input vertex already spans distinct source views")
    records = np.empty(len(vertices), dtype=[("source", np.int64), ("x", np.float64),
                                             ("y", np.float64), ("z", np.float64)])
    records["source"] = source_vertex
    for column, name in enumerate(("x", "y", "z")):
        records[name] = vertices[:, column]
    _, first, inverse = np.unique(records, return_index=True, return_inverse=True)
    welded = vertices[first]
    mapped_faces = inverse[faces].astype(np.int64)
    if not np.array_equal(welded[mapped_faces], vertices[faces]):
        raise RuntimeError("Exact source-view welding changed triangle geometry")
    return welded, mapped_faces


def mesh_topology(vertices, faces):
    """Audit connected components, Euler characteristic, and explicit boundary.

    Edges are sorted integer arrays; no content digest or identity hash is used.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    if not len(faces):
        return {"components": 0, "euler": 0, "boundary_edges": 0,
                "nonmanifold_edges": 0}, np.empty((0, 2), dtype=np.int64)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges.sort(axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    graph = coo_matrix((np.ones(len(unique), dtype=bool), (unique[:, 0], unique[:, 1])),
                       shape=(len(vertices), len(vertices))).tocsr()
    components = connected_components(graph, directed=False, return_labels=False)
    return {"components": int(components), "euler": int(len(vertices) - len(unique) + len(faces)),
            "boundary_edges": int(len(boundary)), "nonmanifold_edges": int((counts > 2).sum())}, boundary


def assert_boundary_preserved(before_vertices, before_edges, after_vertices, after_edges):
    """Require unchanged boundary segments, rather than trust a soft QEM weight."""
    if len(before_edges) != len(after_edges):
        raise ValueError("Simplification changed the number of boundary edges")
    if not len(before_edges):
        return
    def ordered_segments(vertices, edges):
        # Coincident but topologically distinct vertices may occur in a TSDF.
        # Compare geometric segment multisets, never nearest-neighbor IDs whose
        # tie-breaking would falsely reject an unchanged coincident boundary.
        segments = vertices[edges].copy()
        a, b = segments[:, 0], segments[:, 1]
        swap = ((a[:, 0] > b[:, 0]) | ((a[:, 0] == b[:, 0]) & (a[:, 1] > b[:, 1])) |
                ((a[:, 0] == b[:, 0]) & (a[:, 1] == b[:, 1]) & (a[:, 2] > b[:, 2])))
        segments[swap] = segments[swap, ::-1]
        flat = segments.reshape(-1, 6)
        return flat[np.lexsort(tuple(flat[:, column] for column in range(5, -1, -1)))]
    if not np.array_equal(ordered_segments(before_vertices, before_edges),
                          ordered_segments(after_vertices, after_edges)):
        raise ValueError("Simplification changed original boundary segments")


def save_triangle_table(path, table, **face_attributes):
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Triangle-table output must use a .npz filename")
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError("Mesh output already exists; use a new explicit candidate path/run label")
    path.parent.mkdir(parents=True, exist_ok=True)
    for name, values in face_attributes.items():
        if len(values) != len(table.faces):
            raise ValueError(f"Face attribute {name} length differs from face table")
    np.savez(path, vertices=table.vertices, faces=table.faces,
             triangle_ids=np.arange(len(table.faces), dtype=np.int64), **face_attributes)
    metadata = dict(table.source_record, mesh_token=table.mesh_token,
                    vertices=len(table.vertices), triangles=len(table.faces),
                    triangle_id_policy="stable cleaned face row, 0..F-1",
                    mesh_file=str(path.resolve()), mesh_file_bytes=path.stat().st_size)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))


def load_triangle_table(path):
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    with np.load(path, allow_pickle=False) as data:
        vertices, faces = data["vertices"], data["faces"]
        if not np.array_equal(data["triangle_ids"], np.arange(len(faces), dtype=np.int64)):
            raise ValueError("Triangle row identity changed")
    if metadata["vertices"] != len(vertices) or metadata["triangles"] != len(faces):
        raise ValueError("Mesh metadata counts disagree with triangle arrays")
    return TriangleTable(vertices, faces, metadata, metadata["mesh_token"])
