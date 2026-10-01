"""CPU contract tests; the raster kernel is replaced only at its public boundary."""

import importlib
from types import SimpleNamespace

import pytest
import torch

from gaussian_renderer.fvdb_native_renderer import FvdbNativeRenderer, _normalize_raster_inputs
from gaussian_renderer.neural_gaussians import BundleMetadata, NeuralGaussianBatch, ensure_gaussian_batch


def _metadata(mask=None):
    if mask is None:
        mask = [True, False, True, False, False, False, False, True, True]
    return BundleMetadata.from_selection(
        torch.tensor([9, 2, 5]), torch.tensor([2, 0, 1]), torch.tensor(mask, dtype=torch.bool), 3
    )


def _batch(rows=4, metadata=None, requires_grad=False):
    def values(width):
        return torch.ones(rows, width, requires_grad=requires_grad)

    return NeuralGaussianBatch(
        descriptor=None,
        anchor_indices=torch.tensor([9, 2, 5]),
        xyz=values(3), color=values(3), opacity=values(1),
        scaling=values(3), rotation=values(4),
        selection_mask=torch.ones(rows, dtype=torch.bool), sh_degree=None,
        bundle_metadata=metadata,
    )


def _camera():
    return SimpleNamespace(
        image_height=3, image_width=5, FoVx=1.0, FoVy=0.8,
        world_view_transform=torch.eye(4), camera_center=torch.zeros(3), resolution_scale=1.0,
    )


def test_bundle_keeps_request_order_zero_owner_and_exact_offset_slots():
    bundle = _metadata()
    assert bundle.request_anchor_ids.tolist() == [9, 2, 5]
    assert bundle.counts.tolist() == [2, 0, 2]
    assert bundle.offsets.tolist() == [0, 2, 2, 4]
    assert bundle.row_owner_ids.tolist() == [9, 9, 5, 5]
    assert bundle.row_owner_levels.tolist() == [2, 2, 1, 1]
    assert bundle.row_offset_slots.tolist() == [0, 2, 1, 2]


def test_bundle_all_filtered_and_empty_request_are_different():
    all_filtered = _metadata([False] * 9)
    assert all_filtered.counts.tolist() == [0, 0, 0]
    assert all_filtered.offsets.tolist() == [0, 0, 0, 0]
    empty = BundleMetadata.from_selection(
        torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.bool), 3,
    )
    assert empty.counts.tolist() == []
    assert empty.offsets.tolist() == [0]


@pytest.mark.parametrize("field,value,match", [
    ("counts", [2, 1, 1], "prefix sum"),
    ("offsets", [0, 2, 3, 4], "prefix sum"),
    ("row_owner_ids", [9, 5, 9, 5], "row owners"),
    ("row_owner_levels", [2, 1, 2, 1], "owner levels"),
    ("row_offset_slots", [0, 0, 1, 2], "offset slots"),
])
def test_malformed_bundle_is_rejected(field, value, match):
    fields = dict(vars(_metadata()))
    fields[field] = torch.tensor(value)
    with pytest.raises(ValueError, match=match):
        BundleMetadata(**fields)


def test_bundle_rejects_wrong_mask_and_wrong_decoded_count():
    with pytest.raises(ValueError, match="request_count"):
        _metadata([True] * 8)
    with pytest.raises(ValueError, match="row count"):
        _batch(rows=3, metadata=_metadata())
    fields = dict(vars(_metadata()))
    fields["counts"] = fields["counts"].to(torch.float32)
    with pytest.raises(ValueError, match="int64"):
        BundleMetadata(**fields)


def test_batch_retains_seven_item_tuple_storage_and_gradient_contract():
    batch = _batch(metadata=_metadata(), requires_grad=True)
    assert batch.scaling.dtype == batch.rotation.dtype == torch.float16
    payload = tuple(batch)
    assert len(payload) == 7
    assert payload[3].dtype == payload[4].dtype == torch.float32
    assert payload[0] is batch.xyz
    assert payload[6] is batch.selection_mask
    assert batch.materialize(jagged=True)[-1].keys() == {"anchor_indices", "level_ids"}
    loss = sum(value.sum() for value in payload[:5])
    loss.backward()
    assert batch.xyz.grad is not None
    assert batch.color.grad is not None
    assert batch.scaling.grad_fn is not None


def test_central_adapter_retains_legacy_tuple_behavior():
    batch = _batch()
    payload = (*batch.as_tuple()[:6], None)
    pc = SimpleNamespace(get_anchor=torch.zeros(10, 3))
    converted = ensure_gaussian_batch(pc, payload, anchor_reference=torch.tensor([9, 2, 5]))
    assert converted.anchor_indices.tolist() == [9, 2, 5]
    assert converted.selection_mask.tolist() == [True] * 4
    assert converted.bundle_metadata is None
    assert ensure_gaussian_batch(pc, converted) is converted


@pytest.mark.parametrize("index", range(5))
def test_raster_rejects_each_mismatched_attribute_instead_of_truncating(index):
    widths = [3, 3, 1, 3, 4]
    values = [torch.ones(2, width) for width in widths]
    values[index] = torch.ones(1, widths[index])
    with pytest.raises(ValueError, match="rows"):
        _normalize_raster_inputs(*values, device=torch.device("cpu"))


@pytest.mark.parametrize("index,bad_shape", [(0, (6,)), (1, (2, 4)), (2, (2, 2)), (3, (2, 4)), (4, (2, 3))])
def test_raster_rejects_malformed_shapes(index, bad_shape):
    values = [torch.ones(2, width) for width in [3, 3, 1, 3, 4]]
    values[index] = torch.ones(bad_shape)
    with pytest.raises(ValueError, match="shape"):
        _normalize_raster_inputs(*values, device=torch.device("cpu"))


def test_empty_xyz_does_not_hide_malformed_other_attributes():
    batch = _batch(rows=0)
    batch.color = torch.ones(1, 3)
    with pytest.raises(ValueError, match="color.*rows"):
        FvdbNativeRenderer().render_jagged(
            viewpoint_camera=_camera(), gaussian_batch=batch,
            bg_color=torch.zeros(3), render_mode="RGB",
        )


@pytest.mark.parametrize("color_rows", [0, 1])
def test_legacy_cache_empty_return_validates_all_attributes(monkeypatch, color_rows):
    render_module = importlib.import_module("gaussian_renderer.render")
    ids = torch.tensor([0])
    descriptor = SimpleNamespace(anchor_indices=lambda: ids, level_ids=torch.tensor([0]))
    payload = {
        "xyz": torch.empty(0, 3), "color": torch.empty(color_rows, 3),
        "opacity": torch.empty(0, 1), "scaling": torch.empty(0, 3), "rotation": torch.empty(0, 4),
    }
    cache = SimpleNamespace(
        process_frame=lambda *args, **kwargs: (torch.tensor([True]), torch.tensor([False]), payload, None),
        get_cache_statistics=lambda: {},
    )
    monkeypatch.setattr(render_module, "_CACHE_ENABLE", True)
    monkeypatch.setattr(render_module, "_KEYFRAME_ENABLE", False)
    monkeypatch.setattr(render_module, "_CACHE_LOG_STATS", False)
    monkeypatch.setattr(render_module, "_PROFILE_RENDER", False)
    monkeypatch.setattr(render_module, "_PRECOMP_INDICES_PATH", "")
    monkeypatch.setattr(render_module, "_get_rendering_cache", lambda: cache)
    monkeypatch.setattr(render_module, "_VISIBLE_MASK_STATE", {})
    monkeypatch.setattr(render_module, "sample_visibility", lambda *args, **kwargs: SimpleNamespace(descriptor=descriptor, indices=ids))
    pc = SimpleNamespace(get_anchor=torch.ones(1, 3), set_anchor_mask=lambda *args: None)

    def invoke():
        return render_module.render(_camera(), pc, None, torch.zeros(3), 0, "RGB", disable_cache=False)

    if color_rows:
        with pytest.raises(ValueError, match="color.*rows"):
            invoke()
    else:
        out = invoke()
        assert out["render"].shape == (3, 3, 5)
        assert out["render_depth"] is None
        assert out["visible_mask"].tolist() == [True]
        assert out["selection_mask"].tolist() == [False]


@pytest.mark.parametrize("mode", ["RGB", "RGB+ED"])
def test_empty_raster_has_background_alpha_and_optional_depth(mode):
    batch = _batch(rows=0, metadata=_metadata([False] * 9))
    background = torch.tensor([0.1, 0.2, 0.3])
    out = FvdbNativeRenderer().render_jagged(
        viewpoint_camera=_camera(), gaussian_batch=batch,
        bg_color=background, render_mode=mode,
    )
    torch.testing.assert_close(out["render"], background[:, None, None].expand(3, 3, 5))
    assert not out["render_alpha"].any()
    assert out["render_alpha"].shape == (1, 3, 5)
    assert (out["render_depth"] is None) == (mode == "RGB")
    if mode == "RGB+ED":
        assert out["render_depth"].shape == (1, 3, 5)


def test_public_raster_keeps_training_fields_gradients_and_rgb_default(monkeypatch):
    render_module = importlib.import_module("gaussian_renderer.render")
    native = importlib.import_module("gaussian_renderer.fvdb_native_renderer")
    calls = []

    def fake_rasterization(**kwargs):
        calls.append(kwargs)
        for name in ("means", "quats", "scales", "opacities", "colors"):
            assert kwargs[name].dtype == torch.float32
            assert kwargs[name].is_contiguous()
        means2d = kwargs["means"][:, :2].unsqueeze(0)
        intensity = means2d.sum() + sum(kwargs[name].sum() for name in ("quats", "scales", "opacities", "colors"))
        channels = 4 if kwargs["render_mode"] == "RGB+ED" else 3
        rgb = intensity.expand(1, kwargs["height"], kwargs["width"], channels)
        alpha = intensity.expand(1, kwargs["height"], kwargs["width"], 1)
        return rgb, alpha, {"radii": torch.ones(1, 4), "means2d": means2d}

    monkeypatch.setattr(native.gsplat, "rasterization", fake_rasterization)
    batch = _batch(metadata=_metadata(), requires_grad=True)
    mask = torch.tensor([True, False, True])
    pc = SimpleNamespace(fvdb_renderer=FvdbNativeRenderer())
    out = render_module.rasterize_batch(_camera(), pc, batch, torch.zeros(3), "RGB", visible_mask=mask)
    assert out["visible_mask"] is mask
    assert out["selection_mask"] is batch.selection_mask
    assert out["opacity"] is batch.opacity
    assert out["render_depth"] is None
    assert out["render"].shape == (3, 3, 5)
    assert out["render_alpha"].shape == (1, 3, 5)
    assert out["visibility_filter"].shape == (4,)
    out["render"].sum().backward()
    assert batch.xyz.grad is not None and bool((batch.xyz.grad[:, :2] != 0).all())
    assert batch.color.grad is not None
    assert out["viewspace_points"].grad is not None
    depth_out = render_module.rasterize_batch(_camera(), pc, batch, torch.zeros(3), "RGB+ED", visible_mask=mask)
    torch.testing.assert_close(depth_out["render"], out["render"])
    assert depth_out["render_depth"].shape == (1, 3, 5)
    assert [call["render_mode"] for call in calls] == ["RGB", "RGB+ED"]


def test_explicit_reset_clears_all_legacy_scene_state(monkeypatch):
    render_module = importlib.import_module("gaussian_renderer.render")
    monkeypatch.setattr(render_module, "_PRECOMP_CACHE", {"loaded": True, "cursor": 17, "indices": torch.tensor([4])})
    monkeypatch.setattr(render_module, "_VISIBLE_MASK_STATE", {"buf": torch.ones(3, dtype=torch.bool), "last_idx": torch.tensor([2])})
    monkeypatch.setattr(render_module, "_RENDERING_CACHE", object())
    monkeypatch.setattr(render_module, "_KEYFRAME_SAMPLER", object())
    monkeypatch.setattr(render_module, "_PROFILE_FRAME", 8)
    render_module.reset_render_context()
    assert render_module._PRECOMP_CACHE == render_module._PRECOMP_CACHE_DEFAULTS
    assert render_module._VISIBLE_MASK_STATE == {"buf": None, "last_idx": None, "size": 0, "device": None}
    assert render_module._RENDERING_CACHE is None
    assert render_module._KEYFRAME_SAMPLER is None
    assert render_module._PROFILE_FRAME == 0


def test_scene_reset_refreshes_precompute_path_and_environment_without_per_frame_changes(monkeypatch):
    render_module = importlib.import_module("gaussian_renderer.render")
    try:
        with monkeypatch.context() as environment:
            environment.setenv("PRECOMP_INDICES_PATH", "/scene_a/precomputed.pt")
            environment.setenv("PRECOMP_START_AT", "3")
            environment.setenv("PRECOMP_LOOP", "0")
            environment.setenv("PRECOMP_STRICT", "1")
            environment.setenv("CACHE_ENABLE", "1")
            render_module.reset_render_context()
            assert render_module._PRECOMP_INDICES_PATH == "/scene_a/precomputed.pt"
            assert render_module._PRECOMP_START_AT == 3
            assert render_module._CACHE_ENABLE
            render_module._PRECOMP_CACHE.update(loaded=True, cursor=13, indices=torch.tensor([9]))

            environment.setenv("PRECOMP_INDICES_PATH", "/scene_b/precomputed.pt")
            environment.setenv("PRECOMP_START_AT", "7")
            environment.setenv("PRECOMP_LOOP", "1")
            environment.setenv("PRECOMP_STRICT", "0")
            environment.setenv("CACHE_ENABLE", "0")
            environment.setenv("VIS_SUMMARY_SAMPLES", "2")
            # Environment edits do not mutate a running scene between frames.
            assert render_module._PRECOMP_INDICES_PATH == "/scene_a/precomputed.pt"
            render_module.reset_render_context()
            assert render_module._PRECOMP_INDICES_PATH == "/scene_b/precomputed.pt"
            assert render_module._PRECOMP_START_AT == 7
            assert render_module._PRECOMP_LOOP
            assert not render_module._PRECOMP_STRICT
            assert not render_module._CACHE_ENABLE
            assert render_module._VIS_SUMMARY_SAMPLES == 2
            assert render_module._PRECOMP_CACHE["loaded"] is False
            assert render_module._PRECOMP_CACHE["cursor"] == 0
            assert render_module._PRECOMP_CACHE["indices"] is None
    finally:
        render_module.reset_render_context()


def test_reset_can_keep_settings_and_environment_defaults_remain_compatible(monkeypatch):
    render_module = importlib.import_module("gaussian_renderer.render")
    try:
        with monkeypatch.context() as environment:
            environment.setenv("PRECOMP_INDICES_PATH", "/scene_a/precomputed.pt")
            render_module.reset_render_context()
            for name in ("PRECOMP_INDICES_PATH", "PRECOMP_START_AT", "PRECOMP_LOOP", "PRECOMP_STRICT", "CACHE_ENABLE"):
                environment.delenv(name, raising=False)
            render_module.reset_render_context(refresh_environment=False)
            assert render_module._PRECOMP_INDICES_PATH == "/scene_a/precomputed.pt"
            render_module.reset_render_context()
            assert render_module._PRECOMP_INDICES_PATH == ""
            assert render_module._PRECOMP_START_AT == 0
            assert render_module._PRECOMP_LOOP is False
            assert render_module._PRECOMP_STRICT is True
            assert render_module._CACHE_ENABLE is False
    finally:
        render_module.reset_render_context()


@pytest.mark.parametrize("batch_size,valid", [
    (1.5, False), (True, False), (0, False), (-1, False), ("2", False), (None, False),
    (1, True), (2, True),
])
def test_precompute_header_validates_batch_size_before_integer_coercion(monkeypatch, tmp_path, caplog, batch_size, valid):
    render_module = importlib.import_module("gaussian_renderer.render")
    precompute_path = tmp_path / "precomputed.pt"
    torch.save({
        "version": 2, "num_frames": 1, "total_anchors": 1,
        "grid_meta": {"batch_size": batch_size},
        "frames": [{"indices": [0], "ijk_jdata": [[0, 0, 0]], "ijk_jidx": [0]}],
    }, precompute_path)
    monkeypatch.setattr(render_module, "_PRECOMP_INDICES_PATH", str(precompute_path))
    monkeypatch.setattr(render_module, "_PRECOMP_CACHE", dict(render_module._PRECOMP_CACHE_DEFAULTS))
    render_module._maybe_load_precomputed(num_anchors=1, device=torch.device("cpu"))
    assert render_module._PRECOMP_CACHE["loaded"] is valid
    if valid:
        assert render_module._PRECOMP_CACHE["batch_size"] == batch_size
        assert type(render_module._PRECOMP_CACHE["batch_size"]) is int
    else:
        assert "grid_meta.batch_size must be a positive integer" in caplog.text
        assert render_module._PRECOMP_CACHE["indices"] is None
