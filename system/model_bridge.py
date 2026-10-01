"""Explicit model-family handoff; preserve each decoder's tensor/precision policy."""
from dataclasses import replace
import torch
from batch import BundleMetadata, NeuralGaussianBatch, batch_from_proxygs_decode


def family(model):
    value = getattr(model, '_gdmgs_model_backend', None)
    if value not in ('cachegs', 'proxygs'):
        raise ValueError('loader must explicitly bind a model backend')
    return value


def source_state(view, model, iteration=40000):
    if family(model) == 'proxygs':
        model.set_anchor_mask(view.camera_center, iteration, view.resolution_scale)
    # CacheGS uses compute_pose_local_state inside its original decoder.
    # Calling set_anchor_mask would mutate shared fVDB attribute buffers.


def from_cachegs_batch(source):
    xyz, color, opacity, scaling, rotation, sh_degree, mask = source.materialize()
    old = source.bundle_metadata
    if old is None:
        raise ValueError('CacheGS decoder must return explicit row ownership')
    meta = BundleMetadata(old.request_anchor_ids, old.row_owner_ids, old.row_offset_slots,
                          old.counts, old.offsets, old.request_level_ids, old.row_owner_levels)
    return NeuralGaussianBatch(source.anchor_indices, xyz, color, opacity, scaling,
                              rotation, mask, sh_degree, meta)


def decode_batch(view, model, ids, levels):
    if family(model) == 'cachegs':
        state = model.compute_pose_local_state(view, getattr(model, '_gdmgs_iteration', 40000))
        original = model.generate_neural_gaussians(view, ids, -1, build_descriptor=False,
                                                   pose_state=state, return_bundle_metadata=True)
        result = from_cachegs_batch(original)
    else:
        from gaussian_renderer import generate_neural_gaussians as original_decode
        values = original_decode(view, model, is_training=False, anchor_indices=ids)
        result = batch_from_proxygs_decode(anchor_ids=ids, decoded=values,
                                          n_offsets=model.n_offsets, request_level_ids=levels)
    if not torch.equal(result.bundle_metadata.request_level_ids, levels):
        raise ValueError('decoder changed requested levels')
    result.validate_contract()
    return result


def generate_neural_gaussians(view, model, *, is_training=False, anchor_indices):
    if is_training:
        raise ValueError('shared materialization is inference-only; use original training')
    batch = decode_batch(view, model, anchor_indices, model.get_level[anchor_indices].reshape(-1).long())
    return batch.xyz, batch.color, batch.opacity, batch.scaling, batch.rotation, batch.selection_mask


class DecoderInterface:
    """Expose absent ProxyGS-only options as false for the CacheGS head layout."""
    def __init__(self, model):
        self.model = model

    def __getattr__(self, name):
        if name in ('add_level', 'add_opacity_dist', 'add_color_dist', 'add_cov_dist'):
            return getattr(self.model, name, False)
        return getattr(self.model, name)


def prepare_decode(view, model, ids):
    from fullblock_timed import prepare_decode as retained_prepare
    return retained_prepare(view, DecoderInterface(model), ids)


def finish_decode(state, model, ids, levels):
    from fullblock_timed import finish_decode as retained_finish
    batch = retained_finish(state, DecoderInterface(model), ids, levels)
    if family(model) == 'cachegs':
        # NeuralGaussianBatch.__post_init__ stores these two fields in FP16;
        # materialize() converts back to FP32. Preserve that rounding exactly.
        batch = replace(batch, scaling=batch.scaling.half().float(),
                        rotation=batch.rotation.half().float())
    batch.validate_contract()
    return batch
