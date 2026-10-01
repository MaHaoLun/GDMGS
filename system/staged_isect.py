"""Staged gsplat current-depth intersections, float32 packed=False only.

The only intentional D2H synchronization is read_totals, once per group. Count
and finish run on the current CUDA stream. For multiple streams, callers must
join count events before read_totals and protect tensor allocator lifetimes.
Inputs and CountState must remain immutable until finish completes. total_count
must be the exact entry returned by read_totals for the corresponding state.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import time
import torch

_EXTENSION = None
LOAD_SECONDS = None


def load_extension():
    global _EXTENSION, LOAD_SECONDS
    if _EXTENSION is None:
        import gsplat
        from torch.utils.cpp_extension import load
        root = Path(__file__).resolve().parent / 'staged_isect_src'
        native = Path(gsplat.__file__).resolve().parent / 'cuda' / 'csrc'
        sources = [root/'binding.cpp', root/'staged_isect.cu']
        digest = hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest()[:12]
        start = time.perf_counter()
        _EXTENSION = load(name='vista_staged_isect_'+digest,
            sources=[str(p) for p in sources],
            extra_include_paths=[str(native), str(native/'third_party'/'glm')],
            extra_cflags=['-O3'], extra_cuda_cflags=['-O3', '--use_fast_math'],
            verbose=False)
        LOAD_SECONDS = time.perf_counter() - start
    return _EXTENSION


@dataclass(frozen=True)
class CountState:
    means2d: torch.Tensor
    radii: torch.Tensor
    depths: torch.Tensor
    tiles: torch.Tensor
    prefix: torch.Tensor
    total: torch.Tensor
    tile_size: int
    tile_width: int
    tile_height: int

    def tensors(self):
        return (self.means2d,self.radii,self.depths,self.tiles,self.prefix,self.total)


def count_async(means2d, radii, depths, tile_size, tile_width, tile_height):
    tiles, prefix, total = load_extension().count_async(
        means2d,radii,depths,tile_size,tile_width,tile_height)
    return CountState(means2d,radii,depths,tiles,prefix,total,
                      tile_size,tile_width,tile_height)


def read_totals(states):
    """Exactly one stacked D2H copy. Join producer events before calling."""
    if not states:
        raise ValueError('nonempty count-state list required')
    return torch.stack([s.total for s in states]).cpu().tolist()


def finish(state, total_count, sort=True, double_buffer=False):
    if not isinstance(total_count,int):
        raise TypeError('total_count must be host int from read_totals')
    return load_extension().finish(state.means2d,state.radii,state.depths,
        state.tiles,state.prefix,total_count,state.tile_size,state.tile_width,
        state.tile_height,sort,double_buffer)
