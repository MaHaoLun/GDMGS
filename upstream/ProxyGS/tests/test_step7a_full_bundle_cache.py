from __future__ import annotations

from typing import Dict, Iterable, List

import pytest
import torch

from gaussian_renderer.raster_batch import BundleMetadata, NeuralGaussianBatch
from gdmgs.cache import CacheIdentity, FullBundleCacheCore


N_OFFSETS = 10
ANCHOR_LEVELS = torch.tensor([1, 0, 2, 1, 3], dtype=torch.long)


def identity(trace: str = "step6-window-fixture") -> CacheIdentity:
    return CacheIdentity(
        scene="fixture",
        model="iteration-40000",
        backend="gdmgs-gsplat-v1",
        anchor_table="fixture-ply-row-order",
        trace=trace,
    )


def cache(capacity_rows: int = 32) -> FullBundleCacheCore:
    return FullBundleCacheCore(
        identity=identity(),
        anchor_levels=ANCHOR_LEVELS,
        capacity_rows=capacity_rows,
        n_offsets=N_OFFSETS,
    )


def make_batch(anchor_ids: Iterable[int], counts_by_id: Dict[int, int]) -> NeuralGaussianBatch:
    ids = torch.tensor(list(anchor_ids), dtype=torch.long)
    levels = ANCHOR_LEVELS.index_select(0, ids)
    counts = torch.tensor([counts_by_id[int(value)] for value in ids], dtype=torch.long)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    owners = ids.repeat_interleave(counts)
    owner_levels = levels.repeat_interleave(counts)
    slots: List[int] = []
    values: List[float] = []
    selection_mask = torch.zeros(ids.numel() * N_OFFSETS, dtype=torch.bool)
    for request, (anchor, count) in enumerate(zip(ids.tolist(), counts.tolist())):
        slots.extend(range(count))
        values.extend(float(anchor * 100 + slot) for slot in range(count))
        if count:
            selection_mask[request * N_OFFSETS : request * N_OFFSETS + count] = True
    base = torch.tensor(values, dtype=torch.float32).reshape(-1, 1)
    metadata = BundleMetadata(
        request_anchor_ids=ids,
        row_owner_ids=owners,
        row_offset_slots=torch.tensor(slots, dtype=torch.long),
        counts=counts,
        offsets=offsets,
        request_level_ids=levels,
        row_owner_levels=owner_levels,
    )
    batch = NeuralGaussianBatch(
        anchor_indices=ids,
        xyz=base.repeat(1, 3),
        color=(base + 0.1).repeat(1, 3),
        opacity=base + 0.2,
        scaling=(base + 0.3).repeat(1, 3),
        rotation=(base + 0.4).repeat(1, 4),
        selection_mask=selection_mask,
        sh_degree=None,
        bundle_metadata=metadata,
    )
    batch.validate_contract()
    return batch


def assert_payload_equal(left: NeuralGaussianBatch, right: NeuralGaussianBatch) -> None:
    for name in ("xyz", "color", "opacity", "scaling", "rotation"):
        assert torch.equal(getattr(left, name), getattr(right, name)), name
    assert torch.equal(left.anchor_indices, right.anchor_indices)
    for name in (
        "request_anchor_ids",
        "request_level_ids",
        "row_owner_ids",
        "row_owner_levels",
        "row_offset_slots",
        "counts",
        "offsets",
    ):
        assert torch.equal(
            getattr(left.bundle_metadata, name),
            getattr(right.bundle_metadata, name),
        ), name


def decoder(counts_by_id: Dict[int, int], calls: List[List[int]]):
    def decode(anchor_ids: torch.Tensor, level_ids: torch.Tensor) -> NeuralGaussianBatch:
        calls.append(anchor_ids.tolist())
        batch = make_batch(anchor_ids.tolist(), counts_by_id)
        assert torch.equal(batch.bundle_metadata.request_level_ids, level_ids)
        return batch

    return decode


def test_empty_generation_all_miss_is_exact_and_decodes_once():
    item = cache()
    ids = torch.tensor([2, 0, 3], dtype=torch.long)
    levels = ANCHOR_LEVELS[ids]
    counts = {0: 1, 2: 3, 3: 2}
    calls: List[List[int]] = []
    result = item.resolve(
        anchor_ids=ids,
        level_ids=levels,
        decode_misses=decoder(counts, calls),
    )
    assert calls == [[2, 0, 3]]
    assert result.hit_mask.tolist() == [False, False, False]
    assert result.hit_rows == 0
    assert result.fresh_rows == 6
    assert_payload_equal(result.batch, make_batch(ids.tolist(), counts))


def test_same_pose_all_hit_restores_complete_bundles_exactly():
    item = cache()
    ids = torch.tensor([2, 0, 3], dtype=torch.long)
    levels = ANCHOR_LEVELS[ids]
    counts = {0: 1, 2: 3, 3: 2}
    fresh = make_batch(ids.tolist(), counts)
    generation = item.build_generation(
        fresh,
        request_level_ids=levels,
        source_camera_id="camera-7",
    )
    item.publish_generation(generation)

    def forbidden_decode(*_args):
        raise AssertionError("all-hit resolution must not call the decoder")

    result = item.resolve(
        anchor_ids=ids,
        level_ids=levels,
        decode_misses=forbidden_decode,
    )
    assert result.hit_mask.tolist() == [True, True, True]
    assert result.hit_rows == 6
    assert result.fresh_rows == 0
    assert_payload_equal(result.batch, fresh)
    assert result.batch.xyz.data_ptr() != generation.xyz.data_ptr()
    assert generation.source_camera_ids == ("camera-7", "camera-7", "camera-7")


def test_mixed_hit_miss_reconstructs_original_request_order_exactly():
    item = cache()
    counts = {0: 2, 1: 1, 2: 3}
    seed_ids = torch.tensor([2, 0], dtype=torch.long)
    seed = make_batch(seed_ids.tolist(), counts)
    item.publish_generation(
        item.build_generation(
            seed,
            request_level_ids=ANCHOR_LEVELS[seed_ids],
            source_camera_id="source",
        )
    )
    request_ids = torch.tensor([2, 1, 0], dtype=torch.long)
    calls: List[List[int]] = []
    result = item.resolve(
        anchor_ids=request_ids,
        level_ids=ANCHOR_LEVELS[request_ids],
        decode_misses=decoder(counts, calls),
    )
    assert calls == [[1]]
    assert result.hit_mask.tolist() == [True, False, True]
    assert result.batch.bundle_metadata.counts.tolist() == [3, 1, 2]
    assert_payload_equal(result.batch, make_batch(request_ids.tolist(), counts))


def test_zero_row_bundle_is_never_admitted_and_remains_a_miss():
    item = cache()
    counts = {0: 0, 1: 1}
    ids = torch.tensor([0, 1], dtype=torch.long)
    generation = item.build_generation(
        make_batch(ids.tolist(), counts),
        request_level_ids=ANCHOR_LEVELS[ids],
        source_camera_id="zero-test",
    )
    assert generation.anchor_ids.tolist() == [1]
    item.publish_generation(generation)
    calls: List[List[int]] = []
    result = item.resolve(
        anchor_ids=ids,
        level_ids=ANCHOR_LEVELS[ids],
        decode_misses=decoder(counts, calls),
    )
    assert result.hit_mask.tolist() == [False, True]
    assert calls == [[0]]
    assert result.batch.bundle_metadata.counts.tolist() == [0, 1]


def test_directory_is_sorted_by_composite_level_anchor_key():
    item = cache()
    ids = torch.tensor([4, 0, 2, 1], dtype=torch.long)
    counts = {0: 1, 1: 1, 2: 1, 4: 1}
    generation = item.build_generation(
        make_batch(ids.tolist(), counts),
        request_level_ids=ANCHOR_LEVELS[ids],
        source_camera_id=None,
    )
    assert generation.packed_keys.tolist() == sorted(generation.packed_keys.tolist())
    assert list(zip(generation.level_ids.tolist(), generation.anchor_ids.tolist())) == [
        (0, 1),
        (1, 0),
        (2, 2),
        (3, 4),
    ]


def test_capacity_is_rows_and_never_partially_admits_a_bundle():
    exact = cache(capacity_rows=10)
    ids = torch.tensor([2], dtype=torch.long)
    generation = exact.build_generation(
        make_batch([2], {2: 10}),
        request_level_ids=ANCHOR_LEVELS[ids],
        source_camera_id="full-bundle",
    )
    assert generation.occupied_rows == 10
    assert generation.memory_bytes()["payload"] == 10 * 56

    overflow = cache(capacity_rows=10)
    ids = torch.tensor([0, 1], dtype=torch.long)
    with pytest.raises(ValueError, match="complete bundles require 11 rows"):
        overflow.build_generation(
            make_batch([0, 1], {0: 6, 1: 5}),
            request_level_ids=ANCHOR_LEVELS[ids],
            source_camera_id="overflow",
        )


def test_invalid_requests_and_identity_fail_closed():
    item = cache()
    with pytest.raises(ValueError, match="duplicates"):
        item.resolve(
            anchor_ids=torch.tensor([0, 0]),
            level_ids=torch.tensor([1, 1]),
            decode_misses=lambda *_: None,
        )
    with pytest.raises(ValueError, match="out-of-range"):
        item.resolve(
            anchor_ids=torch.tensor([5]),
            level_ids=torch.tensor([0]),
            decode_misses=lambda *_: None,
        )
    with pytest.raises(ValueError, match="frozen anchor table"):
        item.resolve(
            anchor_ids=torch.tensor([0]),
            level_ids=torch.tensor([0]),
            decode_misses=lambda *_: None,
        )
    other = FullBundleCacheCore(
        identity=identity(trace="other-trace"),
        anchor_levels=ANCHOR_LEVELS,
        capacity_rows=32,
        n_offsets=N_OFFSETS,
    )
    foreign = other.build_generation(
        make_batch([0], {0: 1}),
        request_level_ids=ANCHOR_LEVELS[torch.tensor([0])],
        source_camera_id=None,
    )
    with pytest.raises(ValueError, match="lifecycle identity"):
        item.publish_generation(foreign)


def test_reset_and_new_generation_do_not_mutate_sealed_old_output():
    item = cache()
    ids = torch.tensor([0], dtype=torch.long)
    counts = {0: 2}
    first = item.build_generation(
        make_batch([0], counts),
        request_level_ids=ANCHOR_LEVELS[ids],
        source_camera_id="first",
    )
    item.publish_generation(first)
    before = item.resolve(
        anchor_ids=ids,
        level_ids=ANCHOR_LEVELS[ids],
        decode_misses=lambda *_: (_ for _ in ()).throw(AssertionError()),
    )
    saved = before.batch.xyz.clone()
    item.reset()
    assert item.generation.descriptor_count == 0
    assert torch.equal(before.batch.xyz, saved)
    calls: List[List[int]] = []
    after = item.resolve(
        anchor_ids=ids,
        level_ids=ANCHOR_LEVELS[ids],
        decode_misses=decoder(counts, calls),
    )
    assert calls == [[0]]
    assert after.generation_id > before.generation_id


def test_generation_publication_is_forbidden_during_resolution():
    item = cache()
    ids = torch.tensor([0], dtype=torch.long)
    next_generation = item.build_generation(
        make_batch([0], {0: 1}),
        request_level_ids=ANCHOR_LEVELS[ids],
        source_camera_id="next",
    )
    publication_errors = []

    def decode(anchor_ids: torch.Tensor, level_ids: torch.Tensor) -> NeuralGaussianBatch:
        with pytest.raises(RuntimeError, match="during current-frame resolution") as error:
            item.publish_generation(next_generation)
        publication_errors.append(str(error.value))
        return make_batch(anchor_ids.tolist(), {0: 1})

    result = item.resolve(
        anchor_ids=ids,
        level_ids=ANCHOR_LEVELS[ids],
        decode_misses=decode,
    )
    assert publication_errors
    assert result.miss_mask.tolist() == [True]


def test_empty_request_set_is_a_valid_background_batch():
    item = cache()
    calls = []
    result = item.resolve(
        anchor_ids=torch.empty(0, dtype=torch.long),
        level_ids=torch.empty(0, dtype=torch.long),
        decode_misses=lambda *_: calls.append(True),
    )
    assert not calls
    assert result.batch.xyz.shape == (0, 3)
    assert result.batch.bundle_metadata.offsets.tolist() == [0]
    assert result.hit_mask.numel() == 0
