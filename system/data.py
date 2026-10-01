"""Actual ProxyGS checkpoint/camera loader, with explicit portable input paths."""
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from dense_math import interpolate_pose
from render_gdmgs_backend import _load_cfg, _new_model, _ordered_views, _frozen_camera_names
from scene import Scene
from step4_runtime import camera_domain_from_view
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


def load(config):
    model_path = Path(config['model_path']).resolve()
    cfg = _load_cfg(model_path)
    if config.get('source_path'):
        cfg.source_path = str(Path(config['source_path']).resolve())
    cfg.data_device = 'cpu'
    model = _new_model(cfg)
    scene = Scene(cfg, model, load_iteration=int(config.get('iteration', 40000)),
                  shuffle=False, resolution_scales=cfg.resolution_scales)
    model.eval()
    views = _ordered_views(scene, _frozen_camera_names(model_path))
    if config.get('camera_ids'):
        names = config['camera_ids']
        if len(names) != len(set(names)):
            raise ValueError('duplicate camera IDs')
        by_name = {v.image_name: v for v in views}
        views = [by_name[name] for name in names]
    density = config.get('interpolation', 1)
    if isinstance(density, bool) or not isinstance(density, int) or density < 1:
        raise ValueError('interpolation must be a positive integer')
    if density > 1:
        dense = []
        for j, (a, b) in enumerate(zip(views[:-1], views[1:])):
            if (a.image_width, a.image_height) != (b.image_width, b.image_height):
                raise ValueError('camera size changes across interpolation')
            for q in range(density):
                if q == 0:
                    dense.append(a)
                    continue
                alpha = q / density
                v = copy.copy(a)
                v.R, v.T = interpolate_pose(a.R, a.T, b.R, b.T, alpha)
                v.Fx, v.Fy = (1-alpha)*a.Fx+alpha*b.Fx, (1-alpha)*a.Fy+alpha*b.Fy
                v.FoVx = 2*math.atan(v.image_width/(2*v.Fx))
                v.FoVy = 2*math.atan(v.image_height/(2*v.Fy))
                v.world_view_transform = torch.tensor(getWorld2View2(v.R, v.T)).T.cuda()
                v.projection_matrix = getProjectionMatrix(znear=v.znear, zfar=v.zfar,
                                                          fovX=v.FoVx, fovY=v.FoVy).T.cuda()
                v.full_proj_transform = v.world_view_transform @ v.projection_matrix
                v.camera_center = v.world_view_transform.inverse()[3, :3]
                v.image_name = f'dense_{j:03d}_{q}of{density}'
                v.original_image = None
                dense.append(v)
        views = dense + [views[-1]]
    expected = config.get('expected_targets')
    if expected is not None and len(views) != expected:
        raise ValueError(f'complete trajectory has {len(views)} targets, expected {expected}')
    background = torch.tensor([1., 1., 1.] if cfg.white_background else [0., 0., 0.], device='cuda')
    rt = SimpleNamespace(model=model, levels=model.get_level.detach().reshape(-1).long().contiguous(),
                         background=background, views=views,
                         domains=[camera_domain_from_view(v) for v in views])
    return SimpleNamespace(rt=rt, views=views, cfg=cfg,
                           base=SimpleNamespace(CAPACITY_ROWS=config.get('capacity_rows', 6826846)))
