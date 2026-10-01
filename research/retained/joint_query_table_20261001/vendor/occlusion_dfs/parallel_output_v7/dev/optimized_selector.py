"""Selection entry point: query -> current mesh depth -> anchor occlusion filter.

Each instance owns its current-frame buffers. Use a separate instance for each
concurrently processed frame. The four-epoch incremental path is opt-in; the
single-epoch path avoids batching overhead while reusing the native buffers.
"""
from hybrid import Hybrid
from incremental_depth import IncrementalDepth


class OptimizedSelector:
    def __init__(self,joint,online_rasterizer,*,cut=20):
        self.index=Hybrid(joint,cut=cut)
        self.depth=IncrementalDepth(online_rasterizer)
        self.rasterizer=online_rasterizer

    def query(self,camera_tensor,planes,eye,anchor_ids,camera_domain,*,epochs=1):
        if epochs not in (1,4):
            raise ValueError('Only 1 and 4 epochs have full-inventory validation')
        return self.index.query(camera_tensor,planes,eye,anchor_ids,
            raster=self.rasterizer,domain=camera_domain,epochs=epochs,
            occlusion=True,incremental=self.depth)
