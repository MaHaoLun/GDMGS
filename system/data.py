"""Explicit CacheGS or historical ProxyGS loading, preserving model formats."""
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from dense_math import interpolate_pose
from camera_domain import camera_domain_from_view
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


def load(config):
    model_path = Path(config['model_path']).resolve()
    backend = config.get('model_backend', 'cachegs')
    iteration = int(config.get('iteration', 40000))
    if backend == 'cachegs':
        from utils.render_workflow import load_model_config, load_inference_scene, enumerate_cameras
        if not (model_path/'config.yaml').is_file():
            raise ValueError('CacheGS requires its original config.yaml; no automatic model conversion')
        cfg, opt, pipe = load_model_config(model_path)
        if config.get('source_path'):
            cfg.source_path = str(Path(config['source_path']).resolve())
        cfg.data_device = 'cpu'
        model, scene = load_inference_scene(cfg, opt, iteration, 'merged')
        views = enumerate_cameras(cfg, scene, 'merged')
        background = scene.background
        if getattr(cfg, 'random_background', False):
            raise ValueError('shared inference requires a fixed background')
    elif backend == 'proxygs':
        from render_gdmgs_backend import _load_cfg, _new_model, _ordered_views, _frozen_camera_names
        from scene import Scene
        if not (model_path/'cfg_args').is_file():
            raise ValueError('ProxyGS compatibility requires its original cfg_args')
        cfg = _load_cfg(model_path)
        if config.get('source_path'):
            cfg.source_path = str(Path(config['source_path']).resolve())
        cfg.data_device = 'cpu'
        model = _new_model(cfg)
        scene = Scene(cfg, model, load_iteration=iteration, shuffle=False, resolution_scales=cfg.resolution_scales)
        model.eval()
        views = _ordered_views(scene, _frozen_camera_names(model_path))
        background = torch.tensor([1.,1.,1.] if cfg.white_background else [0.,0.,0.], device='cuda')
    else:
        raise ValueError(f'unknown model backend: {backend}')
    model._gdmgs_model_backend = backend
    model._gdmgs_iteration = int(scene.loaded_iter)
    if not views:
        raise ValueError('checkpoint has no rendering cameras')
    if config.get('camera_ids'):
        names = config['camera_ids']
        if len(names) != len(set(names)):
            raise ValueError('duplicate camera IDs')
        by_name = {v.image_name: v for v in views}
        if len(by_name) != len(views):
            raise ValueError('camera_ids is ambiguous across resolution scales; supply one scale in the model config')
        views = [by_name[name] for name in names]
    density = config.get('interpolation', 1)
    if isinstance(density, bool) or not isinstance(density, int) or density < 1:
        raise ValueError('interpolation must be a positive integer')
    if density > 1:
        dense = []
        for j, (a, b) in enumerate(zip(views[:-1], views[1:])):
            if (a.image_width, a.image_height) != (b.image_width, b.image_height):
                raise ValueError('camera size changes across interpolation')
            if not (np.all(np.asarray(a.trans) == 0) and a.scale == 1 and np.all(np.asarray(b.trans) == 0) and b.scale == 1):
                raise ValueError('interpolation requires the retained zero-translation, unit-scale camera convention')
            for q in range(density):
                if q == 0:
                    dense.append(a)
                    continue
                alpha = q / density
                v = copy.copy(a)
                v.R, v.T = interpolate_pose(a.R, a.T, b.R, b.T, alpha)
                afx = getattr(a, 'Fx', a.image_width/(2*math.tan(a.FoVx/2)))
                afy = getattr(a, 'Fy', a.image_height/(2*math.tan(a.FoVy/2)))
                bfx = getattr(b, 'Fx', b.image_width/(2*math.tan(b.FoVx/2)))
                bfy = getattr(b, 'Fy', b.image_height/(2*math.tan(b.FoVy/2)))
                v.Fx, v.Fy = (1-alpha)*afx+alpha*bfx, (1-alpha)*afy+alpha*bfy
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
    rt = SimpleNamespace(model=model, levels=model.get_level.detach().reshape(-1).long().contiguous(),
                         background=background, views=views,
                         domains=[camera_domain_from_view(v) for v in views])
    return SimpleNamespace(rt=rt, views=views, cfg=cfg,
                           base=SimpleNamespace(CAPACITY_ROWS=config.get('capacity_rows', 6826846)))
