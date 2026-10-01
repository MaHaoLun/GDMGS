"""Real production training smoke; deliberately uses no existing checkpoint."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_real_forward_backward_optimizer_statistics_save_reload(tmp_path):
    if os.environ.get("GDMGS_RUN_CUDA_TESTS") != "1":
        pytest.skip("Set GDMGS_RUN_CUDA_TESTS=1 with an explicitly reserved CUDA device")
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "Lease a CUDA device before running this test"
    root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ, CACHE_ENABLE="0")
    environment.pop("PRECOMP_INDICES_PATH", None)
    completed = subprocess.run(
        [sys.executable, str(root / "tools/gdmgs/validate_ab.py"),
         "--source-root", str(root), "--output", str(tmp_path),
         "--run-label", "pytest_temporary_training", "--mode", "training"],
        env=environment, capture_output=True, text=True, timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads((tmp_path / "training_report.json").read_text())
    assert report["status"] == "pass"
    assert all(report["optimizer_changed"].values())
    assert report["training_stats"]["offset_denom"] > 0
    assert report["reload_render"]["torch_equal"]
    assert all(report["saved_tensor_equal"].values())
