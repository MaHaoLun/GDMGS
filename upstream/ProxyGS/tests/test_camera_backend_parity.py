import math
from types import SimpleNamespace

import pytest
import torch

from gaussian_renderer.gdmgs_gsplat_backend import camera_backend_settings


def test_camera_contract_uses_gdmgs_fov_K_and_transposed_viewmat():
    viewmat = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    camera = SimpleNamespace(
        image_width=1600,
        image_height=900,
        FoVx=1.1,
        FoVy=0.7,
        world_view_transform=viewmat,
    )
    settings = camera_backend_settings(camera, torch.zeros(3), "RGB")
    expected_fx = 1600 / (2 * math.tan(1.1 / 2))
    expected_fy = 900 / (2 * math.tan(0.7 / 2))
    assert settings["K"][0] == pytest.approx([expected_fx, 0.0, 800.0])
    assert settings["K"][1] == pytest.approx([0.0, expected_fy, 450.0])
    assert settings["K"][2] == [0.0, 0.0, 1.0]
    assert settings["viewmat"] == viewmat.transpose(0, 1).tolist()
    assert settings["packed"] is False
    assert settings["sh_degree"] is None
