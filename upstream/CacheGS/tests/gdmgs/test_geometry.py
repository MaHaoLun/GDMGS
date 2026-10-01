"""Analytic geometry fixtures, independent of model/CUDA loading."""
import numpy as np
import pytest

from gdmgs.geometry.depth_mesh import (DepthFrame, TriangleTable, backproject,
    assert_boundary_preserved, compact_triangles, cross_view_support, grid_triangles, load_triangle_table,
    mesh_topology, validate_depth_source, weld_same_source_vertices,
    save_triangle_table)


def plane(height=9, width=9, depth=3.):
    return DepthFrame(np.full((height, width), depth, dtype=np.float32),
        np.ones((height, width), dtype=np.float32),
        np.array([[6., 0., width / 2], [0., 6., height / 2], [0., 0., 1.]]),
        np.eye(4), 4)


def test_camera_z_not_ray_distance_and_pixel_center():
    frame = plane()
    point = backproject(frame, np.array([0, 4]), np.array([0, 4]))
    np.testing.assert_allclose(point, [[-2., -2., 3.], [0., 0., 3.]])
    assert np.linalg.norm(point[0]) > frame.depth[0, 0]


def test_world_to_camera_rotation_and_translation():
    original = plane()
    w2c = np.array([[0., 0., 1., 2.], [0., 1., 0., -1.], [-1., 0., 0., 4.], [0., 0., 0., 1.]])
    frame = DepthFrame(original.depth, original.alpha, original.intrinsics, w2c, 0)
    points = backproject(frame, np.array([4]), np.array([4]))
    np.testing.assert_allclose(points, [[1., 1., -2.]])
    agrees, conflict = cross_view_support(points, frame)
    assert agrees.tolist() == [True] and conflict.tolist() == [False]


@pytest.mark.parametrize("bad", ["projective_camera", "skew_intrinsics"])
def test_malformed_calibration_is_rejected_exactly(bad):
    frame = plane()
    w2c, intrinsics = frame.world_to_camera.copy(), frame.intrinsics.copy()
    if bad == "projective_camera":
        w2c[3, 0] = 1e-10
    else:
        intrinsics[0, 1] = 1e-10
    with pytest.raises(ValueError):
        DepthFrame(frame.depth, frame.alpha, intrinsics, w2c, 0)


@pytest.mark.parametrize("stride, expected", [(1, 128), (2, 32), (4, 8)])
def test_entire_plane_grid_preserves_border_cells(stride, expected):
    vertices, faces, confidence = grid_triangles(plane(), stride=stride)
    assert faces.shape == (expected, 3)
    assert len(confidence) == len(faces)
    np.testing.assert_allclose(vertices[:, 2], 3.)
    normal = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    assert (normal[:, 2] > 0).all()


@pytest.mark.parametrize("failure", ["hole", "depth_jump", "nan"])
def test_interior_pixel_invalidates_quad_even_when_four_corners_valid(failure):
    frame = plane()
    if failure == "hole":
        frame.alpha[2, 2] = 0
    elif failure == "depth_jump":
        frame.depth[2, 2] = 4
    else:
        frame.depth[2, 2] = np.nan
    vertices, faces, _ = grid_triangles(frame, stride=4)
    assert len(faces) == 6
    centers = vertices[faces].mean(axis=1)
    assert not ((centers[:, 0] < 0) & (centers[:, 1] < 0)).any()


def test_visibility_distinguishes_missing_from_free_space_conflict():
    frame = plane()
    points = np.array([[0., 0., 3.], [0., 0., 1.], [0., 0., 5.], [0., 0., -3.], [100., 0., 3.]])
    agrees, conflict = cross_view_support(points, frame)
    assert agrees.tolist() == [True, False, False, False, False]
    assert conflict.tolist() == [False, True, False, False, False]


def test_compaction_does_not_add_or_reorder_surviving_faces():
    vertices = np.array([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.], [100., 100., 100.]])
    faces = np.array([[0, 1, 2], [1, 1, 1]], dtype=np.int64)
    compact, indices, keep = compact_triangles(vertices, faces)
    assert keep.tolist() == [True, False]
    np.testing.assert_array_equal(compact[indices], vertices[faces[:1]])


def test_boundary_audit_detects_hole_filling_and_motion():
    vertices = np.array([[0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.]])
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    stats, edges = mesh_topology(vertices, faces)
    assert stats == {"components": 1, "euler": 1, "boundary_edges": 4, "nonmanifold_edges": 0}
    assert_boundary_preserved(vertices, edges, vertices, edges[::-1])
    moved = vertices.copy()
    moved[0, 0] = 1e-10
    with pytest.raises(ValueError):
        assert_boundary_preserved(vertices, edges, moved, edges)
    with pytest.raises(ValueError):
        assert_boundary_preserved(vertices, edges, vertices, edges[:2])


def test_boundary_audit_accepts_unchanged_coincident_components():
    triangle = np.array([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.]])
    vertices = np.concatenate([triangle, triangle])
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    _, edges = mesh_topology(vertices, faces)
    assert_boundary_preserved(vertices, edges, vertices, edges[::-1])


def test_source_view_welding_keeps_every_triangle_and_separates_views():
    vertices, faces, _ = grid_triangles(plane(), stride=2)
    source = np.full(len(faces), 1, dtype=np.int64)
    first, indices = weld_same_source_vertices(vertices, faces, source)
    assert len(first) < len(vertices)
    np.testing.assert_array_equal(first[indices], vertices[faces])
    double_v = np.concatenate([vertices, vertices])
    double_f = np.concatenate([faces, faces + len(vertices)])
    double_source = np.concatenate([source, source + 1])
    welded, mapped = weld_same_source_vertices(double_v, double_f, double_source)
    assert len(welded) == 2 * len(first)
    assert not np.intersect1d(mapped[:len(faces)], mapped[len(faces):]).size


def test_triangle_table_save_load_identity(tmp_path):
    vertices, faces, _ = grid_triangles(plane(), stride=4)
    table = TriangleTable(vertices, faces, {"scene": "fixture"}, "fixture:explicit-run")
    path = tmp_path / "mesh.npz"
    save_triangle_table(path, table)
    loaded = load_triangle_table(path)
    np.testing.assert_array_equal(loaded.vertices, vertices)
    np.testing.assert_array_equal(loaded.faces, faces)
    assert loaded.mesh_token == table.mesh_token
    with pytest.raises(FileExistsError):
        save_triangle_table(path, table)


@pytest.mark.parametrize("field", ["model_path", "iteration", "run_token"])
def test_depth_source_binding_rejects_mixed_runs(field):
    vertices, faces, _ = grid_triangles(plane(), stride=4)
    table = TriangleTable(vertices, faces,
        {"model_path": "/frozen/model", "iteration": 40000, "depth_run_token": "depth-run"}, "mesh-run")
    manifest = {"model_path": "/frozen/model", "iteration": 40000, "run_token": "depth-run"}
    validate_depth_source(table, manifest)
    manifest[field] = "different"
    with pytest.raises(ValueError):
        validate_depth_source(table, manifest)


@pytest.mark.parametrize("bad", ["float_faces", "nan_vertices", "out_of_range", "wrong_shape", "empty_token"])
def test_triangle_table_rejects_invalid_boundary(bad):
    vertices, faces, _ = grid_triangles(plane(), stride=4)
    token = "fixture"
    if bad == "float_faces":
        faces = faces.astype(np.float64)
    elif bad == "nan_vertices":
        vertices[0, 0] = np.nan
    elif bad == "out_of_range":
        faces[0, 0] = len(vertices)
    elif bad == "wrong_shape":
        faces = faces[:, :2]
    else:
        token = ""
    with pytest.raises(ValueError):
        TriangleTable(vertices, faces, {}, token)


def test_native_pixel_interior_conflict_when_vertices_agree():
    pytest.importorskip("open3d")
    from gdmgs.geometry.pixel_consistency import NativePixelCaster
    frame = plane(height=9, width=9, depth=2.)
    # Pixel(4,4) is inside the quad and is absent from its corner observations.
    frame.depth[4, 4] = 4.
    vertices = backproject(frame, np.array([0, 0, 8, 8]), np.array([0, 8, 0, 8]), np.full(4, 2.))
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    agrees, conflicts = cross_view_support(vertices, frame)
    assert agrees.all() and not conflicts.any()
    ids, counts = NativePixelCaster(vertices, faces, cpu_threads=2).conflicts(frame, block_rows=4)
    assert len(ids) == 1 and counts["conflicting_pixels"] == 1
    assert counts["cast_pixels"] == counts["native_pixels"] == 81


def test_native_pixel_layered_deletion_requires_complete_zero_conflict_pass():
    pytest.importorskip("open3d")
    from gdmgs.geometry.pixel_consistency import NativePixelCaster
    frame = plane(height=9, width=9, depth=4.)
    vertices = np.concatenate([np.array([[-4., -4., z], [4., -4., z], [-4., 4., z], [4., 4., z]])
                               for z in (2., 3., 4.)])
    faces = np.concatenate([np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64) + base for base in (0, 4, 8)])
    active = np.arange(6)
    removed_per_pass = []
    for expected in (2, 2, 0):
        local, counts = NativePixelCaster(vertices, faces[active], cpu_threads=2).conflicts(frame, block_rows=4)
        assert len(local) == expected and counts["cast_pixels"] == 81
        removed_per_pass.append(active[local].tolist())
        active = np.delete(active, local)
    assert removed_per_pass == [[0, 1], [2, 3], []]
    assert active.tolist() == [4, 5]
    # Listing every below-ED intersection produces the same deletion fixed point
    # in one complete pass, including geometry hidden behind another bad layer.
    all_ids, counts = NativePixelCaster(vertices, faces, cpu_threads=2).conflicts(
        frame, block_rows=4, all_intersections=True)
    assert all_ids.tolist() == [0, 1, 2, 3]
    assert counts["cast_pixels"] == 81 and counts["conflicting_pixels"] == 81
    retained = np.delete(np.arange(6), all_ids)
    final_ids, _ = NativePixelCaster(vertices, faces[retained], cpu_threads=2).conflicts(
        frame, all_intersections=True)
    assert not len(final_ids)


def test_native_pixel_uncertain_alpha_and_two_percent_tolerance():
    pytest.importorskip("open3d")
    from gdmgs.geometry.pixel_consistency import NativePixelCaster
    frame = plane(height=9, width=9, depth=4.)
    vertices = np.array([[-4., -4., 3.96], [4., -4., 3.96], [-4., 4., 3.96], [4., 4., 3.96]])
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    ids, counts = NativePixelCaster(vertices, faces, cpu_threads=2).conflicts(frame)
    assert not len(ids) and counts["paired_pixels"] == 81
    vertices[:, 2] = 2.
    frame.alpha[:] = .994
    ids, counts = NativePixelCaster(vertices, faces, cpu_threads=2).conflicts(frame)
    assert not len(ids) and counts["observed_pixels"] == 0 and counts["cast_pixels"] == 81


def test_native_pixel_all_numeric_boundary_matches_nearest_fixed_point():
    pytest.importorskip("open3d")
    from gdmgs.geometry.pixel_consistency import NativePixelCaster
    frame = plane(height=9, width=9, depth=4.)
    threshold = np.float32(3.92)
    levels = [np.nextafter(threshold, np.float32(-np.inf)), threshold,
              np.nextafter(threshold, np.float32(np.inf)), np.float32(4.)]
    vertices = np.concatenate([np.array([[-4., -4., z], [4., -4., z], [-4., 4., z], [4., 4., z]], dtype=np.float64)
                               for z in levels])
    faces = np.concatenate([np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64) + 4 * i for i in range(len(levels))])
    fixed_points = []
    for use_all in (False, True):
        active = np.arange(len(faces))
        for _ in range(len(faces) + 1):
            local, counts = NativePixelCaster(vertices, faces[active], cpu_threads=2).conflicts(
                frame, block_rows=3, all_intersections=use_all)
            assert counts["cast_pixels"] == 81
            if not len(local):
                break
            active = np.delete(active, local)
        else:
            raise AssertionError("Finite layered fixture did not converge")
        fixed_points.append(active)
    np.testing.assert_array_equal(*fixed_points)
