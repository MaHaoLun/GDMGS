"""Continuous triangle-union ORI certificates.

Finite cells prove full continuous angular-cell coverage. Their values bound
occluder camera-z from above. Infinity denotes Unknown. The native certifier
uses exact rational polygon differences after a floating prefilter; no area
tolerance, finite ray sampling, or image-space depth center fills holes.
"""

from dataclasses import dataclass

import numpy as np

from gdmgs.mesh_index import CameraDomain, MeshIndex, MeshQueryResult
from gdmgs.mesh_index.index import _array, _native


@dataclass(frozen=True)
class ORI:
    depth_bounds: np.ndarray
    angular_domain: tuple
    mesh_token: str
    camera_domain: CameraDomain
    counters: dict
    elapsed_ms: float
    complete: bool = True

    @property
    def camera_id(self):
        return self.camera_domain.camera_id


def build_ori(mesh_index, mesh_query_result, camera_domain, ori_shape=(128, 128)):
    """Build immutable float64 [height,width] depth bounds from retrieved faces.

    The MeshIndex owns the triangle table; this avoids copying millions of
    vertices/faces or rebuilding an index for every ORI invocation.
    """
    if not isinstance(mesh_index, MeshIndex):
        raise TypeError("mesh_index must own a validated MeshIndex triangle table")
    camera = CameraDomain.parse(camera_domain)
    if isinstance(mesh_query_result, MeshQueryResult):
        if not mesh_query_result.complete:
            raise ValueError("cannot build ORI from an incomplete triangle query")
        if mesh_query_result.mesh_token != mesh_index.mesh_token:
            raise ValueError("triangle query belongs to a different mesh")
        if not camera.equivalent(mesh_query_result.camera_domain):
            raise ValueError("triangle query belongs to a different camera")
        ids = mesh_query_result.triangle_ids
    else:
        ids = mesh_query_result
    ids = _array(ids, np.int64, (None,), "triangle_ids")
    if ids.size and (ids[0] < 0 or ids[-1] >= len(mesh_index.triangles)
                     or np.any(ids[1:] <= ids[:-1])):
        raise ValueError("triangle_ids must be valid, sorted, and unique")
    if len(ori_shape) != 2 or any(type(n) is not int or n <= 0 for n in ori_shape):
        raise ValueError("ori_shape must be positive integer (height, width)")
    raw = _native().build_ori(mesh_index._mesh, ids, camera.w2c,
                              camera.angular_domain, camera.near, camera.far,
                              ori_shape[0], ori_shape[1])
    depth = raw.pop("depth_bounds")
    elapsed = float(raw.pop("elapsed_ms"))
    complete = bool(raw.pop("complete"))
    if not complete:
        raise RuntimeError("native ORI build did not complete")
    if np.isnan(depth).any() or np.any(depth < camera.near):
        raise RuntimeError("native ORI returned invalid depth bounds")
    depth.flags.writeable = False
    return ORI(depth, camera.angular_domain, mesh_index.mesh_token, camera,
               raw, elapsed, complete)
