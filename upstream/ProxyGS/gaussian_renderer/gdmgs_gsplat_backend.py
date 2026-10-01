"""GDM-GS gsplat raster core ported without behavioral changes.

The bodies of ``validate_raster_inputs``, ``_normalize_raster_inputs`` and
``FvdbNativeRenderer.render_jagged`` are kept source-parity equivalent to
GDMGS_Codebase/gaussian_renderer/fvdb_native_renderer.py.  A validator in
``tools/validate_gdmgs_backend.py`` enforces that claim.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import math

import gsplat
import torch

from gaussian_renderer.raster_batch import NeuralGaussianBatch


def validate_raster_inputs(
    xyz: torch.Tensor,
    color: torch.Tensor,
    opacity: torch.Tensor,
    scaling: torch.Tensor,
    rot: torch.Tensor,
    sh_degree: Optional[int] = None,
) -> None:
    """Validate every row, including empty selections, without copying tensors.

    Opacity keeps the legacy ``[R]``/``[R, 1]`` alternatives. Other attributes
    must have explicit row dimensions; mismatches are never truncated.
    """
    attributes = {"xyz": xyz, "color": color, "opacity": opacity, "scaling": scaling, "rotation": rot}
    for name, value in attributes.items():
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise ValueError(f"Raster attribute {name} must be a floating-point tensor")
    for name, value, width in (("xyz", xyz, 3), ("scaling", scaling, 3), ("rotation", rot, 4)):
        if value.ndim != 2 or value.shape[1] != width:
            raise ValueError(f"Raster attribute {name} must have shape [R, {width}]")
    valid_color = color.ndim == 2 and color.shape[1] == 3
    if sh_degree is not None:
        valid_color = valid_color or (
            color.ndim == 3 and color.shape[2] == 3 and color.shape[1] >= (sh_degree + 1) ** 2
        )
    if not valid_color:
        raise ValueError("Raster color must have shape [R, 3] or compatible [R, SH, 3] coefficients")
    if not (opacity.ndim == 1 or (opacity.ndim == 2 and opacity.shape[1] == 1)):
        raise ValueError("Raster opacity must have shape [R] or [R, 1]")
    rows = xyz.shape[0]
    for name, value in attributes.items():
        if value.shape[0] != rows:
            raise ValueError(f"Raster attribute {name} has {value.shape[0]} rows; expected {rows}")


def _normalize_raster_inputs(
    xyz: torch.Tensor,
    color: torch.Tensor,
    opacity: torch.Tensor,
    scaling: torch.Tensor,
    rot: torch.Tensor,
    device: torch.device,
    sh_degree: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate shapes, then normalize the rasterizer dtype/device contiguously."""
    validate_raster_inputs(xyz, color, opacity, scaling, rot, sh_degree)

    def to32(value: torch.Tensor) -> torch.Tensor:
        return value.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()

    return to32(xyz), to32(color), to32(opacity.reshape(-1)), to32(scaling), to32(rot)


class FvdbNativeRenderer:
    """Renderer implementation used as the configured fvdb-native backend."""

    def render_jagged(
        self,
        *,
        viewpoint_camera: Any,
        gaussian_batch: NeuralGaussianBatch,
        bg_color: torch.Tensor,
        render_mode: str,
    ) -> Dict[str, Any]:
        xyz, color, opacity, scaling, rot, sh_degree, _ = gaussian_batch.materialize()
        device = bg_color.device

        xyz_n, color_n, opacity_n, scaling_n, rot_n = _normalize_raster_inputs(
            xyz, color, opacity, scaling, rot, device=device, sh_degree=sh_degree
        )
        if gaussian_batch.bundle_metadata is not None:
            gaussian_batch.bundle_metadata.validate(row_count=xyz_n.shape[0])

        if xyz_n.shape[0] == 0:
            h, w = int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)
            rendered_image = bg_color.view(-1, 1, 1).expand(3, h, w).contiguous().clone().requires_grad_(True)
            return {
                "render": rendered_image,
                "viewspace_points": torch.zeros(1, 0, 2, device=device),
                "visibility_filter": torch.zeros(0, dtype=torch.bool, device=device),
                "radii": torch.zeros(0, device=device),
                "render_depth": torch.zeros(1, h, w, device=device) if "+" in render_mode else None,
                "render_alpha": torch.zeros(1, h, w, device=device),
                "scaling": scaling,
                "rotation": rot,
                "opacity": opacity,
                "xyz": xyz,
                "color": color,
            }

        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
        focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
        k = torch.tensor(
            [
                [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
                [0, focal_length_y, viewpoint_camera.image_height / 2.0],
                [0, 0, 1],
            ],
            device=device,
        )
        viewmat = viewpoint_camera.world_view_transform.transpose(0, 1)

        render_colors, render_alphas, info = gsplat.rasterization(
            means=xyz_n,
            quats=rot_n,
            scales=scaling_n,
            opacities=opacity_n,
            colors=color_n,
            viewmats=viewmat[None],
            Ks=k[None],
            backgrounds=bg_color[None],
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            packed=False,
            sh_degree=sh_degree,
            render_mode=render_mode,
        )

        if render_colors.shape[-1] == 4:
            colors, depths = render_colors[..., 0:3], render_colors[..., 3:4]
            depth = depths[0].permute(2, 0, 1)
        else:
            colors = render_colors
            depth = None
        rendered_image = colors[0].permute(2, 0, 1)
        radii = info.get("radii", torch.zeros(0, device=device)).squeeze(0)
        means2d = info.get("means2d", torch.zeros(1, 0, 2, device=device))
        return {
            "render": rendered_image,
            "viewspace_points": means2d,
            "visibility_filter": radii > 0,
            "radii": radii,
            "render_depth": depth,
            "render_alpha": render_alphas[0].permute(2, 0, 1),
            "scaling": scaling,
            "rotation": rot,
            "opacity": opacity,
            "xyz": xyz,
            "color": color,
        }


def render_gdmgs_backend(
    viewpoint_camera: Any,
    gaussian_batch: NeuralGaussianBatch,
    bg_color: torch.Tensor,
    render_mode: str = "RGB",
) -> Dict[str, Any]:
    """Fail-closed public entrypoint; it has no native-renderer fallback."""
    gaussian_batch.validate_contract()
    result = FvdbNativeRenderer().render_jagged(
        viewpoint_camera=viewpoint_camera,
        gaussian_batch=gaussian_batch,
        bg_color=bg_color,
        render_mode=render_mode,
    )
    result.update(
        selection_mask=gaussian_batch.selection_mask,
        selected_anchor_ids=gaussian_batch.anchor_indices,
        bundle_metadata=gaussian_batch.bundle_metadata,
        backend="gdmgs-gsplat-v1",
    )
    return result


def camera_backend_settings(viewpoint_camera: Any, bg_color: torch.Tensor, render_mode: str) -> Dict[str, Any]:
    """Serializable settings record built by the same formulas as the backend."""
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
    k = torch.tensor(
        [
            [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
            [0, focal_length_y, viewpoint_camera.image_height / 2.0],
            [0, 0, 1],
        ],
        device=bg_color.device,
    )
    return {
        "K": k.detach().cpu().tolist(),
        "viewmat": viewpoint_camera.world_view_transform.transpose(0, 1).detach().cpu().tolist(),
        "width": int(viewpoint_camera.image_width),
        "height": int(viewpoint_camera.image_height),
        "background": bg_color.detach().cpu().tolist(),
        "packed": False,
        "sh_degree": None,
        "render_mode": render_mode,
    }
