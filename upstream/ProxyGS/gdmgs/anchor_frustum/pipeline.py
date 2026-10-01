"""Live-depth decoder/render handoff for the frozen gsplat inference backend."""
import torch


@torch.no_grad()
def render_with_anchor_frustum(view, model, spatial_index, depth_index,
                               candidate_ids, depth, background,
                               spatial_mode="gpu_tree"):
    """Cull before both depth projection and neural decode; never query mesh.

    The caller generates the usual LoD candidates and supplies its existing
    CUDA depth image. Ranges in `depth_selection` use `spatial_candidates`.
    Returned original anchor IDs are the cache/request identity; callers must
    not reuse old candidate-ordinal ranges with the reduced candidate domain.
    """
    if spatial_index.model is not model or model.get_color_mlp.training:
        raise ValueError("index must belong to this frozen inference model")
    from gaussian_renderer import generate_neural_gaussians
    from gaussian_renderer.raster_batch import batch_from_proxygs_decode
    from gaussian_renderer.gdmgs_gsplat_backend import render_gdmgs_backend
    candidates, _, counters = spatial_index.query(view, candidate_ids, spatial_mode)
    selection = depth_index.query(candidates, depth, view.world_view_transform.contiguous(),
                                  view.full_proj_transform.contiguous(), mode="fused",
                                  camera=view.image_name)
    ids = selection.selected_anchor_ids
    decoded = generate_neural_gaussians(view, model, anchor_indices=ids, is_training=False)
    batch = batch_from_proxygs_decode(anchor_ids=ids, decoded=decoded,
                                     n_offsets=model.n_offsets,
                                     request_level_ids=model.get_level[ids].reshape(-1).long())
    result = render_gdmgs_backend(view, batch, background)
    return dict(render_result=result, depth_selection=selection,
                spatial_candidates=candidates, spatial_counters=counters,
                decoded_anchor_count=ids.numel(), decoded_gaussian_count=batch.xyz.shape[0])
