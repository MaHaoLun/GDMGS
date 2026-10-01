"""
Shared helpers for sampling view-dependent visibility and building fvdb descriptors.

The main renderer, legacy render path, and offline precompute jobs all rely on the
same projection code, so this module centralizes the logic that used to live
inside ``render.py``'s ``prefilter_*`` helpers.  Each sample returns the list of
visible anchor indices plus an optional ``JaggedVisibilityDescriptor`` so cache
and precompute flows can stay in jagged form end-to-end.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from gsplat.cuda._wrapper import fully_fused_projection, fully_fused_projection_2dgs

from utils.fvdb_visibility import JaggedVisibilityDescriptor


@dataclass
class VisibilitySample:
    """Result of a single frustum sample."""

    indices: torch.Tensor
    descriptor: Optional[JaggedVisibilityDescriptor]
    mask: Optional[torch.Tensor] = None


def attach_descriptor_lookup(pc, descriptor: Optional[JaggedVisibilityDescriptor]) -> None:
    """Attach precomputed lookup tables to the descriptor when the model exposes them."""
    if descriptor is None or pc is None:
        return
    attach_fn = getattr(pc, "attach_descriptor_lookup", None)
    if not callable(attach_fn):
        attach_fn = getattr(pc, "_attach_descriptor_lookup", None)
    if callable(attach_fn):
        attach_fn(descriptor)


def _build_descriptor_from_indices(
    pc,
    indices: torch.Tensor,
    coords_table: Optional[torch.Tensor] = None,
    level_table: Optional[torch.Tensor] = None,
) -> JaggedVisibilityDescriptor:
    fvdb_grid = getattr(pc, "fvdb_grid", None)
    if fvdb_grid is None:
        raise RuntimeError("Model is missing fvdb_grid; cannot build JaggedVisibilityDescriptor.")
    total_voxels = int(pc.get_anchor.shape[0])
    if coords_table is None or level_table is None:
        fetch_tables = getattr(pc, "fvdb_anchor_tables", None)
        if not callable(fetch_tables):
            raise RuntimeError("Model does not expose fvdb_anchor_tables(); retrain with fvdb metadata.")
        coords_table, level_table = fetch_tables()
    descriptor = JaggedVisibilityDescriptor.from_indices(
        fvdb_grid,
        indices,
        total_voxels=total_voxels,
        coords_table=coords_table,
        level_table=level_table,
        legacy_indices=indices,
    )
    attach_descriptor_lookup(pc, descriptor)
    return descriptor


def _project_visibility_mask(viewpoint_camera, pc, *, anchor_mask=None) -> torch.Tensor:
    """Project Gaussians via fully_fused_projection and return a boolean visibility mask."""
    if anchor_mask is None:
        anchor_mask = getattr(pc, "_anchor_mask", None)
    if anchor_mask is None:
        anchor_mask = torch.ones(int(pc.get_anchor.shape[0]), dtype=torch.bool, device=pc.get_anchor.device)
    means = pc.get_anchor[anchor_mask]
    scales = pc.get_scaling[anchor_mask][:, :3]
    quats = pc.get_rotation[anchor_mask]

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)

    Ks = torch.tensor(
        [
            [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
            [0, focal_length_y, viewpoint_camera.image_height / 2.0],
            [0, 0, 1],
        ],
        device=means.device,
    )[None]
    viewmats = viewpoint_camera.world_view_transform.transpose(0, 1)[None]

    proj_results = fully_fused_projection(
        means,
        None,
        quats,
        scales,
        viewmats,
        Ks,
        int(viewpoint_camera.image_width),
        int(viewpoint_camera.image_height),
        eps2d=0.3,
        packed=False,
        near_plane=0.01,
        far_plane=1e10,
        radius_clip=0.0,
        sparse_grad=False,
        calc_compensations=False,
    )
    radii = proj_results[0]
    visible_mask = anchor_mask.clone()
    if visible_mask.numel() > 0:
        visible_mask[anchor_mask] = radii.squeeze(0) > 0
    return visible_mask


def sample_visibility_pose_local(
    viewpoint_camera,
    pc,
    pipe,
    bg_color,
    *,
    pose_state,
    return_mask: bool = False,
) -> VisibilitySample:
    """Sample FoV using only this pose's LoD mask, without changing model state.

    Explicit anchor IDs are the identity boundary for the new rendering path;
    no GridBatch descriptor or legacy last-visibility field is needed here.
    """
    del pipe, bg_color
    anchor_mask = pose_state.anchor_mask
    anchor_count = int(pc.get_anchor.shape[0])
    if anchor_mask.dtype != torch.bool or anchor_mask.shape != (anchor_count,):
        raise ValueError(f"Pose anchor_mask must be boolean with shape [{anchor_count}].")
    if anchor_mask.device != pc.get_anchor.device:
        raise ValueError("Pose anchor_mask and model anchors must share a device.")
    if not bool(anchor_mask.any()):
        visible_mask = anchor_mask.clone()
    else:
        visible_mask = _project_visibility_mask(viewpoint_camera, pc, anchor_mask=anchor_mask)
    indices = torch.nonzero(visible_mask, as_tuple=False).flatten()
    return VisibilitySample(indices=indices, descriptor=None, mask=visible_mask if return_mask else None)


def sample_visibility(
    viewpoint_camera,
    pc,
    pipe,
    bg_color,
    *,
    coords_table: Optional[torch.Tensor] = None,
    level_table: Optional[torch.Tensor] = None,
    return_mask: bool = False,
) -> VisibilitySample:
    """Return visible anchor indices and their fvdb descriptor for 3D Gaussians."""
    del pipe, bg_color  # Unused but kept for signature compatibility
    visible_mask = _project_visibility_mask(viewpoint_camera, pc)
    indices = torch.nonzero(visible_mask, as_tuple=False).flatten()
    descriptor = _build_descriptor_from_indices(pc, indices, coords_table=coords_table, level_table=level_table)
    return VisibilitySample(
        indices=indices,
        descriptor=descriptor,
        mask=visible_mask if return_mask else None,
    )


def sample_visibility_2dgs(
    viewpoint_camera,
    pc,
    pipe,
    bg_color,
    *,
    return_mask: bool = False,
) -> VisibilitySample:
    """Return visible anchor indices for 2D Gaussian scenes (no fvdb descriptor)."""
    del pipe, bg_color  # Signature parity
    anchor_mask = getattr(pc, "_anchor_mask", None)
    if anchor_mask is None:
        anchor_mask = torch.ones(int(pc.get_anchor.shape[0]), dtype=torch.bool, device=pc.get_anchor.device)
    means = pc.get_anchor[anchor_mask]
    scales = pc.get_scaling[anchor_mask][:, :3]
    quats = pc.get_rotation[anchor_mask]

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)

    Ks = torch.tensor(
        [
            [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
            [0, focal_length_y, viewpoint_camera.image_height / 2.0],
            [0, 0, 1],
        ],
        device=means.device,
    )[None]
    viewmats = viewpoint_camera.world_view_transform.transpose(0, 1)[None]
    densifications = torch.zeros((viewmats.shape[0], means.shape[0], 2), dtype=means.dtype, device=means.device)

    proj_results = fully_fused_projection_2dgs(
        means,
        quats,
        scales,
        viewmats,
        densifications,
        Ks,
        int(viewpoint_camera.image_width),
        int(viewpoint_camera.image_height),
        eps2d=0.3,
        packed=False,
        near_plane=0.01,
        far_plane=1e10,
        radius_clip=0.0,
        sparse_grad=False,
    )
    radii = proj_results[0]
    visible_mask = anchor_mask.clone()
    if visible_mask.numel() > 0:
        visible_mask[anchor_mask] = radii.squeeze(0) > 0
    indices = torch.nonzero(visible_mask, as_tuple=False).flatten()
    return VisibilitySample(
        indices=indices,
        descriptor=None,
        mask=visible_mask if return_mask else None,
    )
