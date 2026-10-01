from pathlib import Path

import numpy as np
import torch

from fast_geometry import fast_hole_planes


def native():
    from torch.utils.cpp_extension import load
    return load(name='occluder_hole_filter_20260929',
                sources=[str(Path(__file__).with_name('hole_filter.cu'))],
                extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--fmad=false'],
                with_cuda=True,verbose=False)


def filter_ids(extension,bounds,ids,cells,eye):
    if not len(cells):
        return ids
    array=np.zeros((len(cells),12,4),np.float64)
    counts=np.empty(len(cells),np.int32)
    for i,cell in enumerate(cells):
        planes=fast_hole_planes(cell,eye)
        if len(planes)>12:
            raise ValueError('too many hole planes')
        array[i,:len(planes)]=planes
        counts[i]=len(planes)
    gpu_planes=torch.tensor(array,device=bounds.device,dtype=torch.float64)
    gpu_counts=torch.tensor(counts,device=bounds.device,dtype=torch.int32)
    return ids[extension.hole_keep(bounds,ids,gpu_planes,gpu_counts)]
