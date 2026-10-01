"""Target-specific gsplat projection/sort with zero opacity for nonmembers."""
import math
from dataclasses import replace
from types import SimpleNamespace
import torch


class GSplatRenderer:
    def __init__(self, background=(0., 0., 0.), mode='RGB+ED'):
        self.background, self.mode = background, mode

    def __call__(self, camera, shared, target):
        from .gsplat import render_gdmgs_backend
        batch = shared.batch
        device = batch.xyz.device
        mask = shared.row_mask(target)
        # The large attribute arrays remain shared; only target opacity is private.
        opacity = batch.opacity * mask.reshape(batch.opacity.shape).to(batch.opacity.dtype)
        target_batch = replace(batch, opacity=opacity)
        view = SimpleNamespace(image_width=camera.width, image_height=camera.height,
                               FoVx=2*math.atan(camera.width/(2*camera.fx)),
                               FoVy=2*math.atan(camera.height/(2*camera.fy)),
                               znear=camera.near, zfar=camera.far,
                               world_view_transform=torch.tensor(camera.w2c.copy(), device=device,
                                                                 dtype=torch.float32).T)
        result = render_gdmgs_backend(view, target_batch,
                                     torch.tensor(self.background, device=device, dtype=torch.float32), self.mode)
        # Host images have no references to the shared GPU attributes.
        return {name: result[name].detach().cpu() for name in ('render', 'render_alpha', 'render_depth')
                if result[name] is not None}
