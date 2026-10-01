"""Independent analytic geometry and raw-result audit tests.

Oracles deliberately do not call production clipping, coverage, projection,
support or range helpers. Native query APIs are the system under test.
"""
import importlib.util
import math
from pathlib import Path
import json
import copy
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/gdmgs"))
import audit_index_results as AUDIT


def camera(near=0.01, far=math.inf, matrix=None):
    return {"w2c": np.eye(4, dtype=np.float64) if matrix is None else matrix,
            "angular_domain": (-1., 1., -1., 1.), "near": near, "far": far,
            "camera_id": "independent-camera"}


def clip_polygon_halfspace(polygon, normal, constant):
    """Sutherland-Hodgman in camera xyz, separate from native plane rejection."""
    if not len(polygon):
        return polygon
    result = []
    previous = polygon[-1]
    previous_distance = float(previous @ normal + constant)
    for point in polygon:
        distance = float(point @ normal + constant)
        if (distance >= 0) != (previous_distance >= 0):
            ratio = previous_distance / (previous_distance - distance)
            result.append(previous + ratio * (point - previous))
        if distance >= 0:
            result.append(point)
        previous, previous_distance = point, distance
    return np.asarray(result, dtype=np.float64).reshape(-1, 3)


def exact_intersection_oracle(vertices, triangles, domain):
    matrix = domain["w2c"]
    positions = vertices @ matrix[:3, :3].T + matrix[:3, 3]
    xmin, xmax, ymin, ymax = domain["angular_domain"]
    planes = [((1, 0, -xmin), 0), ((-1, 0, xmax), 0),
              ((0, 1, -ymin), 0), ((0, -1, ymax), 0),
              ((0, 0, 1), -domain["near"])]
    if np.isfinite(domain["far"]):
        planes.append(((0, 0, -1), domain["far"]))
    accepted = []
    for index, indices in enumerate(triangles):
        polygon = positions[indices]
        for normal, constant in planes:
            polygon = clip_polygon_halfspace(polygon, np.asarray(normal), constant)
            if not len(polygon):
                break
        if len(polygon):
            accepted.append(index)
    return np.asarray(accepted, dtype=np.int64)


def rectangles_mesh(rectangles, depth=2.):
    vertices, triangles = [], []
    for left, right, bottom, top in rectangles:
        start = len(vertices)
        vertices.extend([(left * depth, bottom * depth, depth),
                         (right * depth, bottom * depth, depth),
                         (right * depth, top * depth, depth),
                         (left * depth, top * depth, depth)])
        triangles.extend([(start, start + 1, start + 2), (start, start + 2, start + 3)])
    return np.asarray(vertices, dtype=np.float64).reshape(-1, 3), np.asarray(triangles, dtype=np.int64).reshape(-1, 3)


def mesh_query(vertices, triangles, domain, shape=(4, 4)):
    from gdmgs.mesh_index import MeshIndex
    from gdmgs.query.ori import build_ori
    mesh = MeshIndex(vertices, triangles, mesh_token="independent-mesh", leaf_size=2)
    result = mesh.query(domain)
    return mesh, result, build_ori(mesh, result, domain, ori_shape=shape)


class TriangleRetrievalOracleTests(unittest.TestCase):
    def test_random_complete_triangles_against_independent_clipping(self):
        from gdmgs.mesh_index import MeshIndex
        rng = np.random.default_rng(1889)
        vertices = rng.uniform(-12, 12, (300, 3)).astype(np.float64)
        triangles = np.arange(300, dtype=np.int64).reshape(-1, 3)
        for method in ("median", "binned_sah"):
            mesh = MeshIndex(vertices, triangles, method=method, leaf_size=3)
            for shift in ((0., 0., 0.), (3., -1., 2.), (-4., 2., -3.)):
                transform = np.eye(4, dtype=np.float64)
                transform[:3, 3] = shift
                domain = camera(near=.25, far=8., matrix=transform)
                independent = exact_intersection_oracle(vertices, triangles, domain)
                indexed = mesh.query(domain, backend="bvh").triangle_ids
                scanned = mesh.query(domain, backend="brute_force").triangle_ids
                np.testing.assert_array_equal(indexed, scanned)
                self.assertTrue(np.isin(independent, indexed).all(), "Lost exact intersecting triangle")
                self.assertEqual(len(indexed), len(np.unique(indexed)))

    def test_long_triangle_near_crossing_root_external_and_empty(self):
        from gdmgs.mesh_index import MeshIndex
        # Centroids of the first two triangles are outside the frustum.
        triangles_xyz = np.array([
            [[-100., 0., 2.], [100., 0., 2.], [0., 100., 2.]],
            [[0., 0., -.5], [-1., -1., 2.], [1., 1., 2.]],
            [[30., 30., 2.], [31., 30., 2.], [30., 31., 2.]],
            [[0., 0., -3.], [1., 0., -2.], [0., 1., -2.]],
        ], dtype=np.float64)
        vertices = triangles_xyz.reshape(-1, 3)
        faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
        mesh = MeshIndex(vertices, faces, leaf_size=1)
        for backend in ("bvh", "brute_force"):
            np.testing.assert_array_equal(mesh.query(camera(), backend=backend).triangle_ids, [0, 1])
        empty = MeshIndex(np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.int64))
        self.assertEqual(len(empty.query(camera()).triangle_ids), 0)

    def test_layout_covers_whole_triangle_geometry_once(self):
        from gdmgs.mesh_index import MeshIndex
        rng = np.random.default_rng(404)
        vertices = rng.normal(size=(75, 3)).astype(np.float64)
        faces = np.arange(75, dtype=np.int64).reshape(-1, 3)
        mesh = MeshIndex(vertices, faces, leaf_size=2)
        layout = mesh.inspect_layout()
        refs, nodes, bounds = layout["triangle_refs"], layout["nodes"], layout["bounds"]
        np.testing.assert_array_equal(np.sort(refs), np.arange(len(faces)))
        self.assertEqual(refs.dtype, np.int64)
        # Native exposes left,right,begin,end. Leaf references directly prove
        # that triangle vertices, including long crossing edges, are enclosed.
        for node, box in zip(nodes, bounds):
            left, right, begin, end = node
            if left < 0:
                points = vertices[faces[refs[begin:end]]].reshape(-1, 3)
                if len(points):
                    self.assertTrue(np.all(points >= box[0]) and np.all(points <= box[1]))
            else:
                for child in (left, right):
                    self.assertTrue(np.all(bounds[child, 0] >= box[0]))
                    self.assertTrue(np.all(bounds[child, 1] <= box[1]))


class DepthGeometryOracleTests(unittest.TestCase):
    def test_camera_z_pixel_centers_and_world_translation(self):
        from gdmgs.geometry.depth_mesh import DepthFrame, backproject
        depth = np.array([[2., 4.], [6., 8.]], dtype=np.float64)
        intrinsics = np.array([[2., 0., .5], [0., 4., .5], [0., 0., 1.]], dtype=np.float64)
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, 3] = [-10., -20., -30.]
        frame = DepthFrame(depth, np.ones_like(depth), intrinsics, w2c, 0)
        rows = np.array([0, 0, 1, 1])
        columns = np.array([0, 1, 0, 1])
        points = backproject(frame, rows, columns)
        # Explicit arithmetic: x=(column+.5-.5)*z/2, y=row*z/4,
        # with world camera center (10,20,30). This is camera-z, not ray length.
        expected = np.array([[10., 20., 32.], [12., 20., 34.],
                             [10., 21.5, 36.], [14., 22., 38.]])
        np.testing.assert_allclose(points, expected, rtol=0, atol=1e-14)


class ContinuousCoverageOracleTests(unittest.TestCase):
    def test_shared_edge_union_covers_whole_cell(self):
        # Neither triangle contains the cell: their union is the exact square.
        vertices, faces = rectangles_mesh([(-1., 1., -1., 1.)])
        mesh, result, ori = mesh_query(vertices, faces, camera(), shape=(1, 1))
        self.assertTrue(np.isfinite(ori.depth_bounds).all())
        self.assertGreaterEqual(float(ori.depth_bounds[0, 0]), 2.)
        self.assertLess(float(ori.depth_bounds[0, 0]), 2.000001)
        from gdmgs.query.ori import build_ori
        for triangle in result.triangle_ids:
            partial = build_ori(mesh, np.array([triangle], dtype=np.int64), camera(), ori_shape=(1, 1))
            self.assertTrue(np.isinf(partial.depth_bounds).all())

    def test_arbitrarily_small_interior_hole_stays_unknown(self):
        # Four closed rectangles leave a real open square hole. Its position
        # avoids cell centers/corners and its width is not an area tolerance.
        center_x, center_y, radius = .1234567, -.2345678, 1e-7
        left, right = center_x - radius, center_x + radius
        bottom, top = center_y - radius, center_y + radius
        rectangles = [(-1., left, -1., 1.), (right, 1., -1., 1.),
                      (left, right, -1., bottom), (left, right, top, 1.)]
        vertices, faces = rectangles_mesh(rectangles)
        _, _, ori = mesh_query(vertices, faces, camera(), shape=(1, 1))
        self.assertTrue(np.isinf(ori.depth_bounds).all(), "A genuine microhole was filled")

    def test_partial_strip_and_two_sided_winding(self):
        vertices, faces = rectangles_mesh([(-1., 0., -1., 1.)])
        _, _, first = mesh_query(vertices, faces, camera())
        _, _, reversed_ori = mesh_query(vertices, faces[:, ::-1].copy(), camera())
        self.assertTrue(np.isfinite(first.depth_bounds[:, :2]).all())
        self.assertTrue(np.isinf(first.depth_bounds[:, 2:]).all())
        np.testing.assert_array_equal(first.depth_bounds, reversed_ori.depth_bounds)

    def test_geometry_before_near_plane_cannot_occlude(self):
        vertices, faces = rectangles_mesh([(-2., 2., -2., 2.)], depth=.005)
        _, result, ori = mesh_query(vertices, faces, camera(near=.01))
        self.assertEqual(len(result.triangle_ids), 0)
        self.assertTrue(np.isinf(ori.depth_bounds).all())

    def test_near_crossing_plane_is_clipped_continuously(self):
        # Plane inverse depth is 100*(1-.9*u). At u<0 it lies before
        # near=.01; at u>=0 it covers the right angular half continuously.
        vertices = []
        for u, v in ((-1., -1.), (1., -1.), (1., 1.), (-1., 1.)):
            z = .01 / (1 - .9 * u)
            vertices.append((u * z, v * z, z))
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        _, _, ori = mesh_query(np.asarray(vertices, dtype=np.float64), faces, camera())
        self.assertTrue(np.isinf(ori.depth_bounds[:, :2]).all())
        # The seam at u=0 can differ by a rounding ulp; the strictly interior
        # far-right column has no seam ambiguity and must receive coverage.
        self.assertTrue(np.isfinite(ori.depth_bounds[:, -1]).all())

    def test_depth_bound_is_upper_bound_for_slanted_plane(self):
        vertices = []
        for u, v in ((-1., -1.), (1., -1.), (1., 1.), (-1., 1.)):
            z = 2. / (1 - .3 * u)
            vertices.append((u * z, v * z, z))
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        _, _, ori = mesh_query(np.asarray(vertices, dtype=np.float64), faces, camera(), shape=(1, 1))
        self.assertTrue(np.isfinite(ori.depth_bounds).all())
        self.assertGreaterEqual(float(ori.depth_bounds[0, 0]), 2. / .7)
        self.assertLess(float(ori.depth_bounds[0, 0]), 2. / .7 + 1e-6)


class AnchorOracleTests(unittest.TestCase):
    def make_index(self):
        from gdmgs.unified_index import AnchorIndex
        # Rows 0/1 share coordinates and must retain separate PLY identities.
        positions = np.array([[0., 0., 5.], [0., 0., 5.], [0., 0., 1.],
                              [0., 0., 5.], [0., 0., .02], [6., 0., 5.]], dtype=np.float64)
        offsets = np.zeros((6, 2, 3), dtype=np.float64)
        offsets[3, 1, 2] = -4.  # A second decoded center is in front of wall.
        scales = np.full((6, 6), .01, dtype=np.float64)
        scales[:, :3] = 1.
        return AnchorIndex.from_arrays(positions, offsets, scales, leaf_capacity=1)

    def test_wall_removes_known_full_support_and_retains_crossings(self):
        index = self.make_index()
        fov = np.array([5, 3, 1, 4, 0, 2], dtype=np.int64)
        wall = np.full((16, 16), 2., dtype=np.float64)
        results = {}
        for mode in ("linear", "tree"):
            result = index.query(fov, wall, np.eye(4, dtype=np.float64),
                                 (-1., 1., -1., 1.), (512, 512), mode=mode)
            results[mode] = result
            np.testing.assert_array_equal(result.selected_anchor_ids, [5, 3, 4, 2])
            for ranges in (result.raw_ranges, result.formal_ranges):
                expanded = AUDIT.expand_ranges(ranges, index.dfs_to_row, mode)
                np.testing.assert_array_equal(expanded, np.sort(result.selected_anchor_ids))
        self.assertTrue(np.isin(results["linear"].selected_anchor_ids,
                               results["tree"].selected_anchor_ids).all())
        np.testing.assert_array_equal(index.dfs_to_row[index.rank_of_row], np.arange(6))

    def test_unknown_cell_retains_anchor_and_empty_fov_is_empty(self):
        index = self.make_index()
        unknown = np.full((16, 16), np.inf, dtype=np.float64)
        fov = np.array([2, 0, 1, 5, 3, 4], dtype=np.int64)
        for mode in ("linear", "tree"):
            result = index.query(fov, unknown, np.eye(4, dtype=np.float64),
                                 (-1., 1., -1., 1.), (512, 512), mode=mode)
            np.testing.assert_array_equal(result.selected_anchor_ids, fov)
            empty = index.query(np.empty(0, dtype=np.int64), unknown, np.eye(4, dtype=np.float64),
                                (-1., 1., -1., 1.), (512, 512), mode=mode)
            self.assertEqual(len(empty.selected_anchor_ids), 0)

    def test_support_encloses_all_offset_centers_and_rotated_axis_extremes(self):
        index = self.make_index()
        boxes = index.layout()["anchor_support_bounds"]
        centers = [[(0., 0., 5.)], [(0., 0., 5.)], [(0., 0., 1.)],
                   [(0., 0., 5.), (0., 0., 1.)], [(0., 0., .02)], [(6., 0., 5.)]]
        radius = math.sqrt(2 * math.log(255)) * .01
        # Every possible rotated covariance principal axis has norm <= .01.
        # Coordinate extrema of that sphere prove whole ellipsoid containment.
        for box, row_centers in zip(boxes, centers):
            for center in row_centers:
                center = np.asarray(center, dtype=np.float64)
                self.assertTrue(np.all(box[:3] <= center - radius))
                self.assertTrue(np.all(box[3:] >= center + radius))


class ResultAuditOracleTests(unittest.TestCase):
    def gpu_counts(self, mode):
        profile, definition, _ = AUDIT.SUPPORT_CONTRACTS["decoder_all_view"]
        counts = {"query_device": "gpu", "scene_token": "scene-owner", "index_token": "index",
                  "mesh_query_index_token": "gpu-mesh-index", "camera_token": "camera-0", "mesh_token": "mesh",
                  "anchor_support_profile": profile, "anchor_support_definition": definition,
                  "payload_transfer_bytes": {key: 0 for key in AUDIT.PAYLOAD_TRANSFERS},
                  "fov": 1, "selected": 1, "decoded": 1, "culled": 0, "triangles": 0}
        counts["mesh"] = {
            "actual_device": "cuda:0", "mesh_token": "mesh", "index_token": "gpu-mesh-index", "camera_id": "camera-0",
            "triangle_table_identity": {"mesh_token": "mesh", "vertices": 3, "triangles": 1, "face_id_policy": "stable_triangle_table_row"},
            "index_structure": "gpu_parallel_triangle_scan" if mode == "R1" else "gpu_bvh_leaf_cluster_scan",
            "backend": "brute_force" if mode == "R1" else "bvh", "visited_nodes": 0,
            "mesh_triangles": 1, "returned_triangles": 0, "tested_triangles": 1 if mode == "R1" else 0,
            "tested_leaf_clusters": 0 if mode == "R1" else 1, "active_leaf_clusters": 0,
            "triangle_id_download_bytes": 0, "camera_upload_bytes": 384}
        counts["ori"] = {
            "ori_definition": AUDIT.ORI_DEFINITION, "continuous_coverage_certificate": False, "tile_size": 8,
            "original_image_size": [12, 12], "padded_image_size": [16, 16],
            "actual_device": "cuda:0", "triangle_query_device": "cuda:0", "mesh_token": "mesh", "camera_id": "camera-0",
            "source_query_index_token": "gpu-mesh-index", "source_triangle_id_policy": "sorted_unique_global_triangle_rows",
            "selected_triangles": 0, "covered_pixels": 0, "unknown_pixels": 256,
            "covered_cells": 0, "unknown_cells": 4, "invalid_raster_hits": 0,
            "upload_bytes": 160, "triangle_id_upload_bytes": 0, "download_bytes": 0}
        counts["anchor"] = {
            "actual_device": "cuda:0", "query_device": "gpu", "support_profile": profile, "support_definition": definition,
            "scene_token": "scene-owner", "index_token": "index", "camera_token": "camera-0", "mesh_token": "mesh",
            "strategy": "parallel_per_fov_anchor" if mode == "R1" else "batch_all_nonempty_nodes_then_uncertified_anchors",
            "range_definition": "canonical_selected_DFS_runs", "fov_count": 1, "selected_count": 1,
            "certified_anchors": 0, "visited_nodes": 0 if mode == "R1" else 1,
            "rectangle_queries": 1, "sparse_table_rectangle_reads": 4}
        counts["anchor"].update({key: 0 for key in (
            "fov_id_upload_bytes", "fov_id_download_bytes", "selected_id_upload_bytes", "selected_id_download_bytes",
            "range_upload_bytes", "range_download_bytes", "ori_payload_upload_bytes", "ori_payload_download_bytes")})
        return counts

    def timing_fixture(self):
        profile, definition, rounding = AUDIT.SUPPORT_CONTRACTS["decoder_all_view"]
        manifest = {"run_id": "fixture", "timing_repeats": 3,
                    "timing_mode_order": [["R0", "R1", "R2"], ["R1", "R2", "R0"], ["R2", "R0", "R1"]],
                    "scenes": [{"scene": "fixture", "settings_token": "native-original",
                                "scene_token": "scene-owner", "mesh_token": "mesh", "index_token": "index",
                                "mesh_query_index_token": "gpu-mesh-index", "anchor_nodes": 1, "triangles": 1, "vertices": 3,
                                "settings": {"ori_backend": "pixel_depth", "ori_definition": AUDIT.ORI_DEFINITION, "tile_size": 8,
                                             "ori_image_size_source": AUDIT.ORI_IMAGE_SIZE_SOURCE,
                                             "query_device": "gpu", "materialization": "fresh-no-cache-metadata-v1",
                                             "support_profile": "decoder_all_view",
                                             "support": {"profile": profile, "definition": definition, "scale_rounding": rounding}},
                                "frames": [{"frame_index": 0, "camera_token": "camera-0", "render_dimensions": [12, 12]}]}]}
        rows = []
        for repeat, order in enumerate(manifest["timing_mode_order"]):
            for mode in order:
                timings = {k: 1. for k in AUDIT.TIMINGS}
                timings.update(total=20., query=6., decode_raster=14.)
                if mode == "R0":
                    timings = {k: None for k in AUDIT.TIMINGS}
                    timings["total"] = 15.
                rows.append({"run_id": "fixture", "scene": "fixture", "frame_index": 0,
                             "camera_token": "camera-0", "settings_token": "native-original",
                             "repeat": repeat, "mode": mode, "status": "complete", "timings_ms": timings,
                             "counts": {"selected": 1} if mode == "R0" else self.gpu_counts(mode)})
        return manifest, rows

    def write_complete_audit_fixture(self, root):
        manifest, rows = self.timing_fixture()
        frame = manifest["scenes"][0]["frames"][0]
        frame.update(uid=0, colmap_id=0, image_name="fixture", split="train", R=np.eye(3).tolist(),
                     T=[0., 0., 0.], FoVx=1., FoVy=1., resolution_scale=1., pose_scale=1.,
                     source_dimensions=[12, 12], render_dimensions=[12, 12],
                     image={"path": "/inputs/fixture/image.png", "size": 42, "mtime_ns": 123})
        scene = manifest["scenes"][0]
        scene.update(scope="full_scene", model_path="/models/fixture", iteration=40000,
                     checkpoint_stats=[{"path": "/models/fixture/model.ply", "size": 42, "mtime_ns": 123}],
                     mesh_token="mesh", index_token="index", dfs_to_row_path="dfs.npy")
        manifest.update(schema_version=1, scope="full_setting", excluded_scenes=["small_city"],
                        required_modes=["R0", "R1", "R2"], reference_survey_path="reference.json",
                        machine={"hostname": "cpu-test", "cpu_model": "test-cpu", "gpu_uuid": "test-device", "gpu_name": "test-gpu"},
                        settings={"cpu_threads": 1, "torch_threads": 1, "cache_enabled": False,
                                  "precompute_enabled": False, "scheduling_enabled": False,
                                  "query_device": "gpu", "thread_environment": {
                                      "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
                                  "quality": AUDIT.QUALITY_SETTINGS})
        (root / "manifest.json").write_text(json.dumps(manifest))
        (root / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        reference_scene = {k: scene[k] for k in ("model_path", "iteration", "checkpoint_stats")}
        reference_scene.update(source_path="/inputs/fixture", merged_render_frames=[frame])
        (root / "reference.json").write_text(json.dumps({"scenes": [reference_scene]}))
        np.save(root / "dfs.npy", np.array([0], dtype=np.int64))
        arrays = {k: np.full((3, 12, 12), .4, dtype=np.float32) for k in ("gt", "r0", "r0_repeat", "r1", "r2")}
        arrays.update(run_id=np.array("fixture"), scene=np.array("fixture"), frame_index=np.int64(0),
                      camera_token=np.array("camera-0"), mesh_token=np.array("mesh"),
                      fov_ids=np.array([0], dtype=np.int64), scan_triangle_ids=np.empty(0, dtype=np.int64),
                      bvh_triangle_ids=np.empty(0, dtype=np.int64))
        for mode in ("r1", "r2"):
            for suffix in ("fov_ids", "selected_ids", "decoded_anchor_ids"):
                arrays[f"{mode}_{suffix}"] = np.array([0], dtype=np.int64)
            for kind in ("raw", "formal"):
                arrays[f"{mode}_{kind}_ranges"] = np.array([[0, 1]], dtype=np.int64)
        (root / "quality" / "fixture").mkdir(parents=True)
        (root / "scenes" / "fixture").mkdir(parents=True)
        (root / "scenes" / "fixture" / "manifest.json").write_text(json.dumps({**manifest, "scope": "full_scene"}))
        (root / "scenes" / "fixture" / "status.json").write_text(json.dumps({
            "status": "complete", "run_id": "fixture", "scene": "fixture", "scope": "full_scene",
            "checkpoint_stats_unchanged": True, "completed_frames": 1, "expected_frames": 1,
            "full_trajectory_frames": 1}))
        return manifest, arrays

    def test_identical_images_have_zero_drop_and_exact_direct_difference(self):
        rng = np.random.default_rng(291)
        image = rng.random((3, 16, 20), dtype=np.float32)
        result = AUDIT.audit_images({k: image.copy() for k in ("gt", "r0", "r0_repeat", "r1", "r2")}, (20, 16))
        self.assertTrue(result["quality"]["r2"]["pass"])
        self.assertEqual(result["quality"]["r2"]["psnr_drop_db"], 0)
        self.assertTrue(result["direct"]["r2"]["exact_equal"])
        self.assertEqual(result["direct"]["r2"]["max_abs"], 0)
        self.assertAlmostEqual(result["metrics"]["r0"]["ssim"], 1.)

    def test_raw_difference_survives_equal_quantized_images(self):
        first = np.full((3, 15, 15), .25, dtype=np.float32)
        second = first + np.float32(1e-5)
        np.testing.assert_array_equal(AUDIT.png_equivalent(first), AUDIT.png_equivalent(second))
        self.assertGreater(AUDIT.direct_metrics(first, second)["max_abs"], 0.)

    def test_quality_gate_rejects_degradation_and_nonfinite_pixels(self):
        gt = np.full((3, 12, 12), .4, dtype=np.float32)
        arrays = {k: gt.copy() for k in ("gt", "r0", "r0_repeat", "r1", "r2")}
        arrays["r2"] += .3
        result = AUDIT.audit_images(arrays, (12, 12))
        self.assertFalse(result["quality"]["r2"]["pass"])
        self.assertEqual(result["quality"]["r2"]["psnr_drop_db"], math.inf)
        arrays["r2"][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            AUDIT.audit_images(arrays, (12, 12))

    def test_ssim_independent_separable_matches_direct_2d_definition(self):
        rng = np.random.default_rng(11)
        x, y = rng.random((2, 3, 13, 17))
        one = np.exp(-np.arange(-5, 6, dtype=np.float64) ** 2 / 4.5)
        one /= one.sum()
        kernel = one[:, None] * one[None, :]

        def direct_blur(value):
            windows = np.lib.stride_tricks.sliding_window_view(np.pad(value, ((0, 0), (5, 5), (5, 5))), (11, 11), axis=(1, 2))
            return np.einsum("...ij,ij->...", windows, kernel)

        a, b = direct_blur(x), direct_blur(y)
        variance_x, variance_y = direct_blur(x*x)-a*a, direct_blur(y*y)-b*b
        cov = direct_blur(x*y)-a*b
        score = np.mean(((2*a*b+.0001)*(2*cov+.0009))/((a*a+b*b+.0001)*(variance_x+variance_y+.0009)))
        self.assertAlmostEqual(AUDIT.independent_ssim(x, y), score, places=13)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "Torch compatibility comparison needs render environment")
    def test_independent_metric_matches_existing_torch_definition(self):
        import torch
        from utils.loss_utils import ssim
        from utils.image_utils import psnr
        rng = np.random.default_rng(2026)
        x = rng.random((3, 23, 19), dtype=np.float32)
        y = np.clip(x + rng.normal(0, .07, x.shape).astype(np.float32), 0., 1.)
        score = AUDIT.image_metrics(x, y)
        a = torch.from_numpy(AUDIT.png_equivalent(x).astype(np.float32))[None]
        b = torch.from_numpy(AUDIT.png_equivalent(y).astype(np.float32))[None]
        self.assertLess(abs(score["ssim"] - float(ssim(a, b))), 2e-6)
        self.assertLess(abs(score["psnr_db"] - float(psnr(a, b).mean())), 2e-5)

    def test_ranges_are_halfopen_exact_and_reject_duplicate_denotation(self):
        order = np.array([4, 0, 3, 1, 2], dtype=np.int64)
        np.testing.assert_array_equal(AUDIT.expand_ranges(np.array([[0, 2], [3, 5]], dtype=np.int64), order, "test"), [0, 1, 2, 4])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            AUDIT.expand_ranges(np.array([[0, 3], [2, 4]], dtype=np.int64), order, "test")

    def test_elapsed_sum_ratio_differs_from_average_per_frame_ratio(self):
        grouped = {}
        for mode, values in (("R0", [1., 100.]), ("R1", [2., 80.]), ("R2", [2., 20.])):
            grouped[("fixture", mode)] = {"query": values, "total": values, "retrieval": values}
        ratios = AUDIT.summarize_timings(grouped)["combined"]["ratios"]
        self.assertAlmostEqual(ratios["r0_total_over_r2_total"], 101./22.)
        self.assertNotAlmostEqual(ratios["r0_total_over_r2_total"], (.5+5.)/2.)

    def test_timing_audit_accepts_null_uninstrumented_r0_components(self):
        manifest, rows = self.timing_fixture()
        grouped = AUDIT.audit_timing_rows(manifest, rows)
        self.assertEqual(grouped[("fixture", "R0")]["total"], [15.] * 3)
        result = AUDIT.summarize_timings(grouped)
        self.assertEqual(result["combined"]["ratios"]["r0_total_over_r2_total"], .75)

    def test_timing_audit_rejects_missing_duplicate_unpaired_and_null_index_measurements(self):
        manifest, rows = self.timing_fixture()
        with self.assertRaisesRegex(ValueError, "Missing 1"):
            AUDIT.audit_timing_rows(manifest, rows[:-1])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            AUDIT.audit_timing_rows(manifest, rows + rows[:1])
        rows[0]["camera_token"] = "unrelated-camera"
        with self.assertRaisesRegex(ValueError, "Unpaired"):
            AUDIT.audit_timing_rows(manifest, rows)
        manifest, rows = self.timing_fixture()
        rows[1]["timings_ms"]["anchor"] = None
        with self.assertRaisesRegex(ValueError, "Missing/nonfinite"):
            AUDIT.audit_timing_rows(manifest, rows)
        manifest, rows = self.timing_fixture()
        rows[1]["timings_ms"]["fov"] = 5000.
        with self.assertRaisesRegex(ValueError, "Impossible query"):
            AUDIT.audit_timing_rows(manifest, rows)
        manifest, rows = self.timing_fixture()
        rows[1]["timings_ms"]["total"] = 19.
        with self.assertRaisesRegex(ValueError, "Impossible total"):
            AUDIT.audit_timing_rows(manifest, rows)

    def test_selection_audit_rejects_extra_r2_deletion_and_changed_fov_order(self):
        order = np.array([0, 1, 2], dtype=np.int64)
        arrays = {"fov_ids": np.array([2, 0, 1], dtype=np.int64),
                  "r1_selected_ids": np.array([2, 0], dtype=np.int64),
                  "r2_selected_ids": np.array([2, 0], dtype=np.int64),
                  "scan_triangle_ids": np.array([1, 4], dtype=np.int64),
                  "bvh_triangle_ids": np.array([1, 4], dtype=np.int64)}
        for mode in ("r1", "r2"):
            arrays[mode + "_fov_ids"] = arrays["fov_ids"].copy()
            arrays[mode + "_decoded_anchor_ids"] = arrays[mode + "_selected_ids"].copy()
            for kind in ("raw", "formal"):
                arrays[f"{mode}_{kind}_ranges"] = np.array([[0, 1], [2, 3]], dtype=np.int64)
        AUDIT.audit_selection(arrays, order)
        arrays["r2_selected_ids"] = np.array([0, 2], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "FoV order"):
            AUDIT.audit_selection(arrays, order)
        arrays["r2_selected_ids"] = np.array([2], dtype=np.int64)
        arrays["r2_decoded_anchor_ids"] = arrays["r2_selected_ids"].copy()
        for kind in ("raw", "formal"):
            arrays[f"r2_{kind}_ranges"] = np.array([[2, 3]], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "R2 deletes"):
            AUDIT.audit_selection(arrays, order)

    def test_final_audit_fails_baseline_repeat_drift_despite_identical_indexed_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, arrays = self.write_complete_audit_fixture(root)
            arrays["r0_repeat"][:] = .8
            np.savez(root / "quality" / "fixture" / "000000.npz", **arrays)
            # This test exercises the full final-audit path on one synthetic
            # frame. The standalone production scene contract is unchanged.
            with patch.dict(AUDIT.EXPECTED_SCENES, {"fixture": 1}, clear=True):
                report = AUDIT.audit_run(root, root / "audit", audit_workers=1)
            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["errors"], [])
            self.assertEqual(report["scenes"]["fixture"]["r0_repeat_failed_frames"], [0])
            self.assertEqual(report["scenes"]["fixture"]["r1_failed_frames"], [])
            self.assertEqual(report["scenes"]["fixture"]["r2_failed_frames"], [])

    def test_final_audit_rejects_development_scope_even_with_complete_frame_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, _ = self.write_complete_audit_fixture(Path(temporary))
            with patch.dict(AUDIT.EXPECTED_SCENES, {"fixture": 1}, clear=True):
                manifest["scope"] = "development"
                with self.assertRaisesRegex(ValueError, "full_setting"):
                    AUDIT.audit_manifest(manifest)
                manifest["scope"] = "full_setting"
                manifest["scenes"][0]["scope"] = "development"
                with self.assertRaisesRegex(ValueError, "development scene"):
                    AUDIT.audit_manifest(manifest)

    def test_final_audit_rejects_old_ori_definition_and_unbound_native_image_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, _ = self.write_complete_audit_fixture(Path(temporary))
            with patch.dict(AUDIT.EXPECTED_SCENES, {"fixture": 1}, clear=True):
                manifest["scenes"][0]["settings"]["ori_definition"] = "continuous_angular_v0"
                with self.assertRaisesRegex(ValueError, "native-pixel"):
                    AUDIT.audit_manifest(manifest)
                manifest["scenes"][0]["settings"]["ori_definition"] = AUDIT.ORI_DEFINITION
                manifest["scenes"][0]["settings"]["ori_image_size_source"] = "thumbnail"
                with self.assertRaisesRegex(ValueError, "native render frame"):
                    AUDIT.audit_manifest(manifest)

    def test_timing_audit_rejects_changed_pixel_grid_and_missing_ori_transfers(self):
        manifest, rows = self.timing_fixture()
        rows[1]["counts"]["ori"]["original_image_size"] = [6, 6]
        with self.assertRaisesRegex(ValueError, "ORI dimensions"):
            AUDIT.audit_timing_rows(manifest, rows)
        manifest, rows = self.timing_fixture()
        rows[1]["counts"]["ori"]["upload_bytes"] = 0
        with self.assertRaisesRegex(ValueError, "camera uploads"):
            AUDIT.audit_timing_rows(manifest, rows)

    def test_final_audit_rejects_failed_terminal_status_or_changed_source_with_complete_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, arrays = self.write_complete_audit_fixture(root)
            np.savez(root / "quality" / "fixture" / "000000.npz", **arrays)
            status_path = root / "scenes" / "fixture" / "status.json"
            status = json.loads(status_path.read_text())
            status["status"] = "failed"
            status_path.write_text(json.dumps(status))
            with patch.dict(AUDIT.EXPECTED_SCENES, {"fixture": 1}, clear=True):
                with self.assertRaisesRegex(ValueError, "did not finish"):
                    AUDIT.audit_run(root, root / "audit", audit_workers=1)
                status.update(status="complete", checkpoint_stats_unchanged=False)
                status_path.write_text(json.dumps(status))
                with self.assertRaisesRegex(ValueError, "preservation"):
                    AUDIT.audit_run(root, root / "audit", audit_workers=1)

    def test_final_audit_rejects_timing_using_different_selection_from_quality_capture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, arrays = self.write_complete_audit_fixture(root)
            np.savez(root / "quality" / "fixture" / "000000.npz", **arrays)
            rows = [json.loads(line) for line in (root / "records.jsonl").read_text().splitlines()]
            rows[1]["counts"].update(selected=0, culled=1)
            rows[1]["counts"]["decoded"] = 0
            rows[1]["counts"]["anchor"].update(selected_count=0, certified_anchors=1)
            (root / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            with patch.dict(AUDIT.EXPECTED_SCENES, {"fixture": 1}, clear=True):
                report = AUDIT.audit_run(root, root / "audit", audit_workers=1)
            self.assertEqual(report["status"], "fail")
            self.assertIn("formal execution counts differ", report["errors"][0]["error"])

    def test_parallel_scene_workers_match_serial_math_and_preserve_scene_frame_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, arrays = self.write_complete_audit_fixture(root)
            rng = np.random.default_rng(44)
            arrays["gt"] = rng.random((3, 12, 12), dtype=np.float32)
            for name in ("r0", "r0_repeat", "r1", "r2"):
                arrays[name] = np.clip(arrays["gt"] + np.float32(.01), 0., 1.)
            np.savez(root / "quality" / "fixture" / "000000.npz", **arrays)
            second = copy.deepcopy(manifest["scenes"][0])
            second["scene"] = "fixture_b"
            manifest["scenes"].append(second)
            (root / "manifest.json").write_text(json.dumps(manifest))
            reference = json.loads((root / "reference.json").read_text())
            reference["scenes"].append({**reference["scenes"][0], "source_path": "/inputs/fixture_b"})
            (root / "reference.json").write_text(json.dumps(reference))
            records = [json.loads(line) for line in (root / "records.jsonl").read_text().splitlines()]
            records += [{**row, "scene": "fixture_b"} for row in records]
            (root / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
            second_root = root / "scenes" / "fixture_b"
            second_root.mkdir()
            original = json.loads((root / "scenes" / "fixture" / "manifest.json").read_text())
            original["scenes"] = [second]
            (second_root / "manifest.json").write_text(json.dumps(original))
            status = json.loads((root / "scenes" / "fixture" / "status.json").read_text())
            status["scene"] = "fixture_b"
            (second_root / "status.json").write_text(json.dumps(status))
            (root / "quality" / "fixture_b").mkdir()
            arrays["scene"] = np.array("fixture_b")
            np.savez(root / "quality" / "fixture_b" / "000000.npz", **arrays)
            with patch.dict(AUDIT.EXPECTED_SCENES, {"fixture": 1, "fixture_b": 1}, clear=True):
                serial = AUDIT.audit_run(root, root / "serial", audit_workers=1)
                parallel = AUDIT.audit_run(root, root / "parallel", audit_workers=2)
            self.assertEqual(serial["status"], "pass")
            self.assertEqual(parallel["status"], "pass")
            for field in ("scenes", "errors", "timing", "actual_scene_statuses", "input_inventory"):
                self.assertEqual(AUDIT.json_safe(serial[field]), AUDIT.json_safe(parallel[field]))
            first_rows = [json.loads(line) for line in (root / "serial" / "independent_per_frame.jsonl").read_text().splitlines()]
            second_rows = [json.loads(line) for line in (root / "parallel" / "independent_per_frame.jsonl").read_text().splitlines()]
            self.assertEqual(first_rows, second_rows)
            self.assertEqual([row["scene"] for row in second_rows], ["fixture", "fixture_b"])
            self.assertEqual(parallel["audit_execution"]["actual_worker_limit"], 2)
            for worker in parallel["audit_execution"]["workers"]:
                self.assertTrue(all(value == "1" for value in worker["thread_environment"].values()))

    def test_audit_worker_resource_limit_is_bounded(self):
        for workers in (0, 9, True):
            with self.assertRaisesRegex(ValueError, "from 1 to 8"):
                AUDIT.audit_run(Path("unused"), Path("unused"), audit_workers=workers)

    def test_gpu_audit_rejects_cpu_baseline_cross_device_source_masks_and_payload_roundtrip(self):
        manifest, rows = self.timing_fixture()
        rows[1]["counts"]["mesh"]["index_structure"] = "cpu_brute_force"
        with self.assertRaisesRegex(ValueError, "GPU mesh strategy"):
            AUDIT.audit_timing_rows(manifest, rows)
        manifest, rows = self.timing_fixture()
        for key in ("mesh", "ori", "anchor"):
            rows[2]["counts"][key]["actual_device"] = "cuda:1"
        rows[2]["counts"]["ori"]["triangle_query_device"] = "cuda:1"
        with self.assertRaisesRegex(ValueError, "different CUDA"):
            AUDIT.audit_timing_rows(manifest, rows)
        manifest, rows = self.timing_fixture()
        rows[2]["counts"]["ori"]["source_query_index_token"] = "unrelated-mask"
        with self.assertRaisesRegex(ValueError, "ORI source mask"):
            AUDIT.audit_timing_rows(manifest, rows)
        manifest, rows = self.timing_fixture()
        rows[2]["counts"]["payload_transfer_bytes"]["fov_d2h"] = 8
        with self.assertRaisesRegex(ValueError, "CPU round trip"):
            AUDIT.audit_timing_rows(manifest, rows)

    def test_gpu_audit_rejects_wrong_support_definition_and_false_coverage_counts(self):
        manifest, rows = self.timing_fixture()
        rows[2]["counts"]["anchor"]["support_definition"] = AUDIT.SUPPORT_CONTRACTS["native_anchor_proxy"][1]
        with self.assertRaisesRegex(ValueError, "proxy/support definition"):
            AUDIT.audit_timing_rows(manifest, rows)
        manifest, rows = self.timing_fixture()
        rows[2]["counts"]["ori"].update(covered_cells=5, unknown_cells=0)
        with self.assertRaisesRegex(ValueError, "coverage counts"):
            AUDIT.audit_timing_rows(manifest, rows)


if __name__ == "__main__":
    unittest.main()
