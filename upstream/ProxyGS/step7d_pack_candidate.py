"""Independent complete-bundle pack candidate using index_select gathers."""
import torch

from gaussian_renderer.raster_batch import BundleMetadata, NeuralGaussianBatch
from gdmgs.cache.full_bundle_cache import _segment_rows


FIELDS = ("xyz", "color", "opacity", "scaling", "rotation")


def take_requests_index_select(batch, positions, ids, levels):
    meta = batch.bundle_metadata
    counts = meta.counts.index_select(0, positions)
    rows = _segment_rows(meta.offsets[:-1].index_select(0, positions), counts)
    return NeuralGaussianBatch(
        anchor_indices=ids,
        **{name: getattr(batch, name).index_select(0, rows) for name in FIELDS},
        selection_mask=torch.ones(rows.numel(), dtype=torch.bool, device=ids.device),
        sh_degree=batch.sh_degree,
        bundle_metadata=BundleMetadata(
            request_anchor_ids=ids, request_level_ids=levels,
            counts=counts, offsets=torch.cat((counts.new_zeros(1), counts.cumsum(0))),
            row_owner_ids=ids.repeat_interleave(counts),
            row_owner_levels=levels.repeat_interleave(counts),
            row_offset_slots=meta.row_offset_slots.index_select(0, rows),
        ),
    )
