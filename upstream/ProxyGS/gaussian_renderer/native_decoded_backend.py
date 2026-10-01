"""ProxyGS native diagnostic rasterizer for an already-decoded batch."""

import math

import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def render_native_decoded(viewpoint_camera, gaussian_batch, pipe, bg_color, scaling_modifier=1.0):
    """Rasterize the exact tensor objects handed to the gsplat backend."""
    gaussian_batch.validate_contract()
    screenspace_points = torch.zeros_like(
        gaussian_batch.xyz,
        dtype=gaussian_batch.xyz.dtype,
        requires_grad=False,
        device=gaussian_batch.xyz.device,
    )
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=math.tan(viewpoint_camera.FoVx * 0.5),
        tanfovy=math.tan(viewpoint_camera.FoVy * 0.5),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    rendered_image, radii = rasterizer(
        means3D=gaussian_batch.xyz,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=gaussian_batch.color,
        opacities=gaussian_batch.opacity,
        scales=gaussian_batch.scaling,
        rotations=gaussian_batch.rotation,
        cov3D_precomp=None,
    )
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "backend": "proxygs-native-decoded-diagnostic",
    }
