"""Pure contract tests for Step 4 dense selection and parity reporting."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from step4_runtime import dense_anchor_filter_cpu, parity_stats


def camera_matrices():
    view = np.eye(4, dtype=np.float32).T
    projection = np.eye(4, dtype=np.float32).T
    return view, projection


def test_dense_filter_contract_and_original_order():
    anchors = np.array(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [3.0, 0.0, 2.0], [0.0, 0.0, -1.0]],
        dtype=np.float32,
    )
    candidates = np.arange(4, dtype=np.int64)
    depth = np.full((4, 4), np.inf, dtype=np.float32)
    depth[2, 2] = 1.5
    view, projection = camera_matrices()
    selected, counters = dense_anchor_filter_cpu(candidates, anchors, view, projection, depth)
    np.testing.assert_array_equal(selected, [0, 2])
    assert counters["culled_nonpositive_z"] == 1
    assert counters["culled_finite_depth"] == 1
    assert counters["kept_out_of_image"] == 1


def test_infinite_and_out_of_image_are_kept_but_nonpositive_z_is_not():
    anchors = np.array([[0, 0, 2], [4, 0, 2], [0, 0, 0]], dtype=np.float32)
    candidates = np.arange(3, dtype=np.int64)
    depth = np.full((2, 2), np.inf, dtype=np.float32)
    view, projection = camera_matrices()
    selected, _ = dense_anchor_filter_cpu(candidates, anchors, view, projection, depth)
    np.testing.assert_array_equal(selected, [0, 1])


def test_rejects_duplicate_candidate_ids_and_bad_depth():
    anchors = np.ones((3, 3), dtype=np.float32)
    depth = np.ones((2, 2), dtype=np.float32)
    view, projection = camera_matrices()
    with pytest.raises(ValueError):
        dense_anchor_filter_cpu(np.array([0, 0], dtype=np.int64), anchors, view, projection, depth)
    depth[0, 0] = np.nan
    with pytest.raises(ValueError):
        dense_anchor_filter_cpu(np.array([0], dtype=np.int64), anchors, view, projection, depth)


def test_depth_parity_requires_exact_coverage_and_frozen_tolerance():
    reference = np.array([[1.0, np.inf]], dtype=np.float32)
    actual = np.array([[1.00001, np.inf]], dtype=np.float32)
    assert parity_stats(actual, reference, atol=1e-3, rtol=2e-4)["pass"]
    actual[0, 1] = 2.0
    assert not parity_stats(actual, reference, atol=1e-3, rtol=2e-4)["pass"]
    assert parity_stats(
        actual,
        reference,
        atol=1e-3,
        rtol=2e-4,
        max_coverage_mismatch_fraction=0.5,
    )["pass"]


def test_oracle_aggregate_policy_allows_rare_edge_tie_but_not_global_drift():
    reference = np.ones((200, 200), dtype=np.float32)
    actual = reference.copy()
    actual[0, 0] += 0.2
    report = parity_stats(
        actual,
        reference,
        atol=1e-3,
        rtol=2e-4,
        mean_limit=1e-4,
        p99_limit=5e-4,
        allow_local_edge_ties=True,
        max_outlier_fraction=1e-4,
    )
    assert not report["all_pixels_within_atol_rtol"]
    assert report["pass"]
    actual.fill(1.001)
    assert not parity_stats(
        actual,
        reference,
        atol=1e-3,
        rtol=2e-4,
        mean_limit=1e-4,
        p99_limit=5e-4,
        allow_local_edge_ties=True,
        max_outlier_fraction=1e-4,
    )["pass"]


def test_normalized_depth_policy_is_scale_aware():
    reference = np.full((100, 100), 100.0, dtype=np.float32)
    actual = reference + np.float32(1e-3)
    report = parity_stats(
        actual,
        reference,
        atol=1e-3,
        rtol=2e-4,
        normalized_mean_limit=1.1e-5,
        normalized_p99_limit=1.1e-5,
    )
    assert report["pass"]


def test_protocol_v2_keeps_saved_depth_diagnostic_only():
    root = Path(__file__).resolve().parents[1]
    render_source = (root / "render_g1.py").read_text()
    review_source = (root / "tools" / "final_step4_review.py").read_text()
    assert 'PROTOCOL_ID = "proxygs-step4-g1-v2"' in render_source
    assert 'oracle_parity["hard_gate"] = False' in render_source
    assert 'if not oracle_parity["pass"]' not in render_source
    assert 'parity.get("full_online_vs_step2_oracle", {}).get("pass")' not in review_source
