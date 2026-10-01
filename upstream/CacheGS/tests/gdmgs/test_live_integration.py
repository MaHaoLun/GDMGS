"""Opt-in independent live comparison; baseline must come from a separate source."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_live_fresh_matches_clean_baseline_and_records_pose_probe(tmp_path):
    if os.environ.get("GDMGS_RUN_CUDA_TESTS") != "1":
        pytest.skip("Set GDMGS_RUN_CUDA_TESTS=1 to run real CUDA integration")
    model = os.environ.get("GDMGS_LIVE_MODEL")
    baseline = os.environ.get("GDMGS_LIVE_BASELINE")
    assert model and baseline, "Set GDMGS_LIVE_MODEL and GDMGS_LIVE_BASELINE to immutable comparison inputs"
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    root = Path(__file__).resolve().parents[2]
    baseline_report = json.loads((Path(baseline) / "fresh_report.json").read_text())
    assert Path(baseline_report["source_root"]).resolve() != root
    assert baseline_report["status"] == "pass"
    environment = dict(os.environ, CACHE_ENABLE="0")
    environment.pop("PRECOMP_INDICES_PATH", None)
    completed = subprocess.run(
        [sys.executable, str(root / "tools/gdmgs/validate_ab.py"),
         "--source-root", str(root), "--model-path", model, "--output", str(tmp_path),
         "--run-label", "pytest_live_fresh", "--mode", "gdmgs", "--compare", baseline],
        env=environment, capture_output=True, text=True, timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads((tmp_path / "gdmgs_report.json").read_text())
    assert report["status"] == "pass"
    assert report["frames"] == baseline_report["frames"]
    assert report["checkpoint_stats_unchanged"]
    assert all(row["baseline_comparison"]["torch_equal"] for row in report["results"])
    assert len(report["cross_pose_precheck"]) == 2
    assert all(row["c2_status"] == "not_frozen; measured_only" for row in report["cross_pose_precheck"])
    actual_counts = {row["frame_index"]: row["selected_anchor_count"] for row in report["results"]}
    for probe in report["cross_pose_precheck"]:
        assert probe["source_frame"] == report["results"][0]["frame_index"]
        assert probe["selection_source"] == "target_pose_actual_fov_selection"
        assert probe["same_target_ids_as_fresh_render"]
        assert probe["target_actual_selection_count"] == actual_counts[probe["target_frame"]]
