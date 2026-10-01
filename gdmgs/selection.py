"""Read-only indices, private query state, and canonical original anchor IDs."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CPUSelector:
    anchors: object
    occluders: object = None
    eligibility: object = None

    def __post_init__(self):
        if self.occluders is not None and not (
                np.array_equal(self.anchors.domain_lo, self.occluders.lo)
                and np.array_equal(self.anchors.domain_hi, self.occluders.hi)):
            raise ValueError('anchor and occluder indices must share the same domain')

    def __call__(self, camera):
        cells = () if self.occluders is None else self.occluders.query(
            camera.planes, camera.w2c, camera.near)
        eligible = None if self.eligibility is None else self.eligibility(camera)
        return self.anchors.query(camera.planes, cells, camera.center, eligible)[0]
