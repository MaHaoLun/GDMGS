"""Explicit camera identity, without content digests or mutable camera references."""

from dataclasses import dataclass
import math
from numbers import Integral

import torch


def appearance_key(ape_code):
    if isinstance(ape_code, Integral) and not isinstance(ape_code, bool):
        return int(ape_code)
    if isinstance(ape_code, (tuple, list)) and len(ape_code) == 1:
        return appearance_key(ape_code[0])
    if isinstance(ape_code, torch.Tensor) and ape_code.numel() == 1:
        if ape_code.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("Appearance identity must be an integer.")
        return int(ape_code.item())
    raise TypeError("Appearance identity must be an integer or a one-element integer sequence.")


def integer_field(value, name):
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    return int(value)


@dataclass(frozen=True)
class CameraQueryKey:
    world_to_camera: tuple
    camera_center: tuple
    fov_x: float
    fov_y: float
    width: int
    height: int
    near: float
    far: float
    resolution_scale: float
    uid: int
    appearance: int
    iteration: int
    projection_near: float = 0.01
    projection_far: float = 1e10

    @classmethod
    def from_camera(cls, camera, iteration, ape_code=-1):
        matrix = camera.world_view_transform
        center = camera.camera_center
        if tuple(matrix.shape) != (4, 4) or tuple(center.shape) != (3,):
            raise ValueError("Camera requires a 4x4 world-view transform and a 3D center.")
        values = tuple(float(x) for x in matrix.detach().T.cpu().reshape(-1).tolist())
        center_values = tuple(float(x) for x in center.detach().cpu().tolist())
        fx, fy = float(camera.FoVx), float(camera.FoVy)
        scale = float(camera.resolution_scale)
        near, far = float(getattr(camera, "znear", 0.01)), float(getattr(camera, "zfar", 100.0))
        if not all(math.isfinite(x) for x in values + center_values + (fx, fy, scale, near, far)):
            raise ValueError("Camera contains non-finite values.")
        if not (0 < fx < math.pi and 0 < fy < math.pi and 0 < near < far and scale > 0):
            raise ValueError("Invalid camera FoV, clipping interval, or resolution scale.")
        width = integer_field(camera.image_width, "Camera width")
        height = integer_field(camera.image_height, "Camera height")
        if width <= 0 or height <= 0:
            raise ValueError("Camera dimensions must be positive.")
        uid = appearance_key(camera.uid)
        iteration = integer_field(iteration, "Checkpoint iteration")
        if iteration < 0:
            raise ValueError("Checkpoint iteration must be resolved and nonnegative.")
        return cls(values, center_values, fx, fy, width, height, near, far, scale,
                   uid, appearance_key(ape_code), iteration)
