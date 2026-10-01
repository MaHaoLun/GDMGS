"""Camera-domain adapter from the retained selection runtime."""
import math
from typing import Any
import numpy as np

def camera_domain_from_view(view: Any):
    from gdmgs.mesh_index import CameraDomain

    w2c = np.ascontiguousarray(
        view.world_view_transform.transpose(0, 1).detach().cpu().numpy(), dtype=np.float64
    )
    tx = math.tan(float(view.FoVx) * 0.5)
    ty = math.tan(float(view.FoVy) * 0.5)
    return CameraDomain.parse(
        {
            "w2c": w2c,
            "angular_domain": (-tx, tx, -ty, ty),
            "near": float(view.znear),
            "far": float(view.zfar),
            "camera_id": str(view.image_name),
        }
    )
