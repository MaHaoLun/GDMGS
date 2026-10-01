"""Camera convention: x_camera = R @ x_world + t; center = -R.T @ t."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclass(frozen=True)
class Camera:
    w2c: np.ndarray
    width: int
    height: int
    fx: float
    fy: float
    near: float = .01
    far: float = 1000.

    def __post_init__(self):
        matrix = np.array(self.w2c, dtype=np.float64, copy=True)
        if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[3], [0, 0, 0, 1])
                or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-6)):
            raise ValueError('expected a rigid world-to-camera transform')
        if not (self.width > 0 and self.height > 0 and self.fx > 0 and self.fy > 0
                and 0 < self.near < self.far):
            raise ValueError('invalid pinhole camera')
        matrix.setflags(write=False)
        object.__setattr__(self, 'w2c', matrix)

    @property
    def center(self):
        return -self.w2c[:3, :3].T @ self.w2c[:3, 3]

    @property
    def planes(self):
        x, y = self.width / (2 * self.fx), self.height / (2 * self.fy)
        local = np.array([[1, 0, x, 0], [-1, 0, x, 0], [0, 1, y, 0],
                          [0, -1, y, 0], [0, 0, 1, -self.near], [0, 0, -1, self.far]])
        planes = local @ self.w2c
        return planes / np.linalg.norm(planes[:, :3], axis=1)[:, None]


def source_camera(cameras):
    """Actual-group midpoint, with center interpolation and rotation SLERP.

    Adapted from dense_math.interpolate_pose and epoch_cache.source_view.
    """
    if not cameras:
        raise ValueError('empty group')
    position = (len(cameras) - 1) / 2
    a, b = cameras[int(np.floor(position))], cameras[int(np.ceil(position))]
    if a is b:
        return a
    if (a.width, a.height) != (b.width, b.height):
        raise ValueError('group cameras must have matching image sizes')
    t = position % 1
    rotation = Slerp([0., 1.], Rotation.from_matrix([a.w2c[:3, :3], b.w2c[:3, :3]]))([t]).as_matrix()[0]
    center = (1 - t) * a.center + t * b.center
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, -rotation @ center
    return Camera(matrix, a.width, a.height, (1-t)*a.fx+t*b.fx, (1-t)*a.fy+t*b.fy,
                  (1-t)*a.near+t*b.near, (1-t)*a.far+t*b.far)
