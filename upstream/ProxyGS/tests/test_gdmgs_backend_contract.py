from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from gaussian_renderer.gdmgs_gsplat_backend import FvdbNativeRenderer, validate_raster_inputs
from gaussian_renderer.raster_batch import BundleMetadata, NeuralGaussianBatch


def camera():
    return SimpleNamespace(
        image_height=3,
        image_width=5,
        FoVx=1.0,
        FoVy=0.8,
        world_view_transform=torch.eye(4),
    )


def batch(rows=2):
    ids = torch.arange(rows, dtype=torch.long)
    mask = torch.ones(rows, dtype=torch.bool)
    metadata = BundleMetadata(
        request_anchor_ids=ids,
        row_owner_ids=ids,
        row_offset_slots=torch.zeros(rows, dtype=torch.long),
        counts=torch.ones(rows, dtype=torch.long),
        offsets=torch.arange(rows + 1, dtype=torch.long),
    )
    return NeuralGaussianBatch(
        anchor_indices=ids,
        xyz=torch.ones(rows, 3),
        color=torch.ones(rows, 3),
        opacity=torch.ones(rows, 1),
        scaling=torch.ones(rows, 3),
        rotation=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(rows, 1),
        selection_mask=mask,
        sh_degree=None,
        bundle_metadata=metadata,
    )


@pytest.mark.parametrize("field,width", [("xyz", 2), ("color", 4), ("opacity", 2), ("scaling", 4), ("rotation", 3)])
def test_malformed_shapes_fail_closed(field, width):
    item = batch()
    values = {name: getattr(item, name) for name in ("xyz", "color", "opacity", "scaling", "rotation")}
    values[field] = torch.ones(2, width)
    with pytest.raises(ValueError):
        validate_raster_inputs(
            values["xyz"], values["color"], values["opacity"], values["scaling"], values["rotation"]
        )


def test_nonfinite_handoff_fails_before_kernel():
    item = batch()
    item.xyz[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        item.validate_contract()


def test_empty_batch_uses_gdmgs_background_contract():
    item = batch(rows=0)
    background = torch.tensor([0.1, 0.2, 0.3])
    output = FvdbNativeRenderer().render_jagged(
        viewpoint_camera=camera(), gaussian_batch=item, bg_color=background, render_mode="RGB+ED"
    )
    torch.testing.assert_close(output["render"], background[:, None, None].expand(3, 3, 5))
    assert output["render_alpha"].shape == (1, 3, 5)
    assert output["render_depth"].shape == (1, 3, 5)


def test_kernel_call_is_float32_contiguous_and_packed_false():
    item = batch()

    def fake_rasterization(**kwargs):
        for name in ("means", "quats", "scales", "opacities", "colors"):
            assert kwargs[name].dtype == torch.float32
            assert kwargs[name].is_contiguous()
        assert kwargs["packed"] is False
        return (
            torch.zeros(1, 3, 5, 3),
            torch.zeros(1, 3, 5, 1),
            {"radii": torch.ones(1, 2), "means2d": torch.zeros(1, 2, 2)},
        )

    with mock.patch("gaussian_renderer.gdmgs_gsplat_backend.gsplat.rasterization", fake_rasterization):
        output = FvdbNativeRenderer().render_jagged(
            viewpoint_camera=camera(),
            gaussian_batch=item,
            bg_color=torch.zeros(3),
            render_mode="RGB",
        )
    assert output["render"].shape == (3, 3, 5)
    assert output["visibility_filter"].tolist() == [True, True]
