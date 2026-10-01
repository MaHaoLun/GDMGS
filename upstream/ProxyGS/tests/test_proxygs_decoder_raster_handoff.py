from types import SimpleNamespace

import pytest
import torch

from gaussian_renderer import generate_neural_gaussians
from gaussian_renderer.raster_batch import batch_from_proxygs_decode, validate_ordered_anchor_ids


class ConstantMLP:
    def __init__(self, output):
        self.output = output
        self.training = False

    def __call__(self, value):
        return self.output.to(value.device).expand(value.shape[0], -1)


class FakeModel:
    def __init__(self):
        self.get_anchor = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0]])
        self.get_anchor_feat = torch.ones(3, 2)
        self.get_level = torch.zeros(3, 1)
        self._offset = torch.zeros(3, 1, 3)
        self.get_scaling = torch.ones(3, 6)
        self.n_offsets = 1
        self.use_feat_bank = False
        self.add_level = False
        self.appearance_dim = 0
        self.add_opacity_dist = False
        self.add_color_dist = False
        self.add_cov_dist = False
        self.dist2level = "round"
        self.get_opacity_mlp = ConstantMLP(torch.tensor([[1.0]]))
        self.get_color_mlp = ConstantMLP(torch.tensor([[0.2, 0.3, 0.4]]))
        self.get_cov_mlp = ConstantMLP(torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]))

    @staticmethod
    def rotation_activation(value):
        return torch.nn.functional.normalize(value, dim=-1)


def test_ordered_ids_survive_decoder_and_row_ownership():
    model = FakeModel()
    view = SimpleNamespace(camera_center=torch.zeros(3), uid=0)
    ids = torch.tensor([2, 0], dtype=torch.long)
    decoded = generate_neural_gaussians(view, model, is_training=False, anchor_indices=ids)
    gaussian_batch = batch_from_proxygs_decode(anchor_ids=ids, decoded=decoded, n_offsets=1)
    assert gaussian_batch.xyz[:, 0].tolist() == [2.0, 0.0]
    assert gaussian_batch.bundle_metadata.row_owner_ids.tolist() == [2, 0]
    assert gaussian_batch.anchor_indices.data_ptr() == ids.data_ptr()


def test_explicit_ids_reject_duplicates_out_of_range_and_wrong_dtype():
    device = torch.device("cpu")
    with pytest.raises(ValueError, match="duplicates"):
        validate_ordered_anchor_ids(torch.tensor([1, 1]), anchor_count=3, device=device)
    with pytest.raises(ValueError, match="out-of-range"):
        validate_ordered_anchor_ids(torch.tensor([3]), anchor_count=3, device=device)
    with pytest.raises(ValueError, match="int64"):
        validate_ordered_anchor_ids(torch.tensor([1.0]), anchor_count=3, device=device)


def test_legacy_mask_and_explicit_ids_are_mutually_exclusive():
    model = FakeModel()
    view = SimpleNamespace(camera_center=torch.zeros(3), uid=0)
    with pytest.raises(ValueError, match="mutually exclusive"):
        generate_neural_gaussians(
            view,
            model,
            visible_mask=torch.ones(3, dtype=torch.bool),
            anchor_indices=torch.tensor([0]),
        )
