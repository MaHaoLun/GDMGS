"""Validate source, environment, camera, tensor, and output parity gates."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import torch

from gaussian_renderer.gdmgs_gsplat_backend import FvdbNativeRenderer, camera_backend_settings
from gaussian_renderer.raster_batch import BundleMetadata, NeuralGaussianBatch


def _definitions(path: Path) -> Dict[str, str]:
    tree = ast.parse(path.read_text())
    definitions: Dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions[node.name] = ast.dump(ast.Module(body=node.body, type_ignores=[]), include_attributes=False)
        elif isinstance(node, ast.ClassDef) and node.name == "FvdbNativeRenderer":
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name == "render_jagged":
                    definitions["FvdbNativeRenderer.render_jagged"] = ast.dump(
                        ast.Module(body=member.body, type_ignores=[]), include_attributes=False
                    )
    return definitions


def source_parity(gdmgs_source: Path, proxy_source: Path) -> Dict[str, Any]:
    left = _definitions(gdmgs_source)
    right = _definitions(proxy_source)
    names = ["validate_raster_inputs", "_normalize_raster_inputs", "FvdbNativeRenderer.render_jagged"]
    matches = {name: left.get(name) == right.get(name) for name in names}
    return {"status": "pass" if all(matches.values()) else "fail", "definition_body_matches": matches}


def fixture_parity() -> Dict[str, Any]:
    camera = SimpleNamespace(
        image_height=3,
        image_width=5,
        FoVx=1.0,
        FoVy=0.8,
        world_view_transform=torch.eye(4),
    )
    ids = torch.tensor([2, 0], dtype=torch.long)
    mask = torch.tensor([True, False, True, True], dtype=torch.bool)
    counts = torch.tensor([1, 2], dtype=torch.long)
    metadata = BundleMetadata(
        request_anchor_ids=ids,
        row_owner_ids=torch.tensor([2, 0, 0], dtype=torch.long),
        row_offset_slots=torch.tensor([0, 0, 1], dtype=torch.long),
        counts=counts,
        offsets=torch.tensor([0, 1, 3], dtype=torch.long),
    )
    batch = NeuralGaussianBatch(
        anchor_indices=ids,
        xyz=torch.ones(3, 3),
        color=torch.full((3, 3), 0.5),
        opacity=torch.ones(3, 1),
        scaling=torch.ones(3, 3),
        rotation=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1),
        selection_mask=mask,
        sh_degree=None,
        bundle_metadata=metadata,
    )
    batch.validate_contract()
    background = torch.tensor([0.1, 0.2, 0.3])
    calls = []

    def fake_rasterization(**kwargs):
        calls.append(kwargs)
        channels = 4 if kwargs["render_mode"] == "RGB+ED" else 3
        colors = torch.zeros(1, kwargs["height"], kwargs["width"], channels)
        alpha = torch.zeros(1, kwargs["height"], kwargs["width"], 1)
        info = {"radii": torch.ones(1, 3), "means2d": torch.zeros(1, 3, 2)}
        return colors, alpha, info

    import gaussian_renderer.gdmgs_gsplat_backend as backend

    original = backend.gsplat.rasterization
    backend.gsplat.rasterization = fake_rasterization
    try:
        output = FvdbNativeRenderer().render_jagged(
            viewpoint_camera=camera,
            gaussian_batch=batch,
            bg_color=background,
            render_mode="RGB+ED",
        )
    finally:
        backend.gsplat.rasterization = original
    settings = camera_backend_settings(camera, background, "RGB+ED")
    call = calls[0]
    checks = {
        "float32": all(call[name].dtype == torch.float32 for name in ("means", "quats", "scales", "opacities", "colors")),
        "contiguous": all(call[name].is_contiguous() for name in ("means", "quats", "scales", "opacities", "colors")),
        "packed_false": call["packed"] is False,
        "camera_K": call["Ks"][0].tolist() == settings["K"],
        "camera_viewmat": call["viewmats"][0].tolist() == settings["viewmat"],
        "rgb_shape": list(output["render"].shape) == [3, 3, 5],
        "depth_shape": list(output["render_depth"].shape) == [1, 3, 5],
        "alpha_shape": list(output["render_alpha"].shape) == [1, 3, 5],
    }
    return {"status": "pass" if all(checks.values()) else "fail", "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gdmgs-source", type=Path, required=True)
    parser.add_argument("--proxy-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "schema": "proxygs_step3_backend_parity_v1",
        "source_parity": source_parity(args.gdmgs_source, args.proxy_source),
        "fixture_parity": fixture_parity(),
        "torch": torch.__version__,
    }
    report["status"] = "pass" if all(
        section["status"] == "pass" for section in (report["source_parity"], report["fixture_parity"])
    ) else "fail"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
