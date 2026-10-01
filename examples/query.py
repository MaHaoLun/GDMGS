"""Small geometry example; no model, GPU, or archived artifacts needed."""
import numpy as np
from gdmgs.camera import Camera
from gdmgs.index import AnchorIndex
from gdmgs.selection import CPUSelector


def main():
    camera = Camera(np.eye(4), 64, 64, 32, 32)
    bounds = np.array([[0, 0, 2, .2, .2, 3], [10, 10, 2, 11, 11, 3.]])
    index = AnchorIndex.build(bounds)
    print('Selected original anchor IDs:', CPUSelector(index)(camera).tolist())


if __name__ == '__main__':
    main()
