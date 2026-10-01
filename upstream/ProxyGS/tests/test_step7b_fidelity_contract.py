from __future__ import annotations

import numpy as np
import torch

from gaussian_renderer.raster_batch import BundleMetadata, NeuralGaussianBatch
from gdmgs.cache import CacheIdentity, FullBundleCacheCore
from render_step7b_fidelity import (
    FORMAL_CAPACITY_ROWS,
    THRESHOLDS,
    capacity_ceiling,
    image_metrics,
    request_diagnostics,
)


LEVELS = torch.tensor([0, 1, 0], dtype=torch.long)


class FakeLpips:
    def __call__(self, left, right, normalize=True):
        assert normalize is True
        return torch.mean(torch.abs(left - right)).reshape(1)


def make_batch(ids, counts_by_id):
    anchor_ids = torch.tensor(ids, dtype=torch.long)
    level_ids = LEVELS[anchor_ids]
    counts = torch.tensor([counts_by_id[index] for index in ids], dtype=torch.long)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    owners = anchor_ids.repeat_interleave(counts)
    owner_levels = level_ids.repeat_interleave(counts)
    slots = []
    values = []
    mask = torch.zeros(anchor_ids.numel() * 10, dtype=torch.bool)
    for request, (anchor, count) in enumerate(zip(ids, counts.tolist())):
        slots.extend(range(count))
        values.extend(float(anchor * 100 + slot) for slot in range(count))
        mask[request * 10 : request * 10 + count] = True
    base = torch.tensor(values, dtype=torch.float32).reshape(-1, 1)
    metadata = BundleMetadata(
        request_anchor_ids=anchor_ids,
        row_owner_ids=owners,
        row_offset_slots=torch.tensor(slots, dtype=torch.long),
        counts=counts,
        offsets=offsets,
        request_level_ids=level_ids,
        row_owner_levels=owner_levels,
    )
    return NeuralGaussianBatch(
        anchor_indices=anchor_ids,
        xyz=base.repeat(1, 3),
        color=(base + 0.1).repeat(1, 3),
        opacity=base + 0.2,
        scaling=(base + 0.3).repeat(1, 3),
        rotation=(base + 0.4).repeat(1, 4),
        selection_mask=mask,
        sh_degree=None,
        bundle_metadata=metadata,
    )


def cache():
    return FullBundleCacheCore(
        identity=CacheIdentity(
            scene="fixture",
            model="iteration-40000",
            backend="gdmgs-gsplat-v1",
            anchor_table="fixture-table",
            trace="fixture-trace",
        ),
        anchor_levels=LEVELS,
        capacity_rows=10,
        n_offsets=10,
    )


def test_preregistered_thresholds_and_capacity_are_frozen():
    assert FORMAL_CAPACITY_ROWS == 6_826_846
    assert THRESHOLDS == {
        "direct_psnr_min_db": 40.0,
        "direct_ssim_min": 0.99,
        "direct_lpips_max": 0.01,
        "direct_rgb_mae_max": 0.005,
        "direct_rgb_p99_max": 0.02,
        "direct_rgb_max_max": 0.10,
        "gt_psnr_drop_max_db": 0.10,
        "gt_ssim_drop_max": 0.002,
        "gt_lpips_increase_max": 0.005,
    }


def test_exact_images_are_a_finite_perfect_pass():
    image = torch.full((3, 16, 16), 0.25)
    metrics = image_metrics(image, image, image, FakeLpips())
    assert metrics["direct"]["psnr"] == 120.0
    assert metrics["pass"] is True
    assert all(metrics["gates"].values())


def test_large_single_pixel_error_fails_direct_gate():
    fresh = torch.zeros(3, 16, 16)
    reuse = fresh.clone()
    reuse[0, 0, 0] = 1.0
    metrics = image_metrics(reuse, fresh, fresh, FakeLpips())
    assert metrics["pass"] is False
    assert metrics["gates"]["direct_rgb_max"] is False


def test_owner_offset_diagnostics_preserve_every_hit_request(tmp_path):
    item = cache()
    source = make_batch([0, 1], {0: 2, 1: 1})
    generation = item.build_generation(
        source,
        request_level_ids=LEVELS[source.anchor_indices],
        source_camera_id="source",
    )
    target = make_batch([0, 1, 2], {0: 1, 1: 1, 2: 1})
    evidence = tmp_path / "requests.npz"
    record = request_diagnostics(
        generation,
        target,
        LEVELS[target.anchor_indices],
        evidence,
    )
    assert record["hit_anchors"] == 2
    assert record["miss_anchors"] == 1
    assert record["length_mismatch_anchors"] == 1
    assert record["source_only_slots"] == 1
    assert record["target_only_slots"] == 0
    with np.load(evidence, allow_pickle=False) as data:
        assert data["anchor_ids"].tolist() == [0, 1]
        assert data["source_lengths"].tolist() == [2, 1]
        assert data["target_lengths"].tolist() == [1, 1]
        assert data["matched_offset_slots"].tolist() == [1, 1]
        assert data["attribute_max_abs"].shape == (2, 5)


def test_capacity_ceiling_is_exact_for_unit_value_bundles():
    item = cache()
    source = make_batch([0, 1], {0: 2, 1: 1})
    generation = item.build_generation(
        source,
        request_level_ids=LEVELS[source.anchor_indices],
        source_camera_id="source",
    )
    target_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    result = capacity_ceiling(generation, target_ids, LEVELS[target_ids], [1, 2, 3])
    assert [item["oracle_hit_anchor_ceiling"] for item in result] == [1, 1, 2]
    assert result[-1]["available_overlap_anchors"] == 2
