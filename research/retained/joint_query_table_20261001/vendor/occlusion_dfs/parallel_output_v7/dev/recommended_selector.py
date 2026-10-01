"""Validated selection entry point; default chosen by the final paired run.

`single` uses optimized DFS followed by one complete depth/filter pass.
`incremental` uses four appends into the same native depth/color buffers.
Initialize outside the frame timing region; use one instance per concurrent frame.
The caller supplies the same resident geometry/camera inputs as the benchmark.
"""
from hybrid import Hybrid
from incremental_depth import IncrementalDepth


class RecommendedSelector:
    def __init__(self,joint,online_rasterizer,*,mode='single'):
        if mode not in ('single','incremental'):
            raise ValueError('mode must be single or incremental')
        self.mode=mode
        self.index=Hybrid(joint,cut=20)
        self.rasterizer=online_rasterizer
        self.incremental=IncrementalDepth(online_rasterizer) if mode=='incremental' else None

    def query(self,camera_tensor,planes,eye,anchor_ids,camera_domain):
        return self.index.query(camera_tensor,planes,eye,anchor_ids,
            raster=self.rasterizer,domain=camera_domain,
            epochs=4 if self.mode=='incremental' else 1,
            occlusion=True,incremental=self.incremental)
