"""Synchronous mesh/index-only inference, with measured explicit-ID consumption."""

from dataclasses import dataclass
import importlib
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from gdmgs.adapters import prepare_selection
from gdmgs.pipeline import validate_environment
from gdmgs.runtime.session import InferenceSession


@dataclass
class IndexFrame:
    package: dict
    timings_ms: dict
    fov_ids: np.ndarray
    selected_ids: np.ndarray
    triangle_ids: np.ndarray
    raw_ranges: np.ndarray
    formal_ranges: np.ndarray
    counters: dict


def camera_domain(camera, camera_token):
    tx, ty = math.tan(camera.FoVx / 2), math.tan(camera.FoVy / 2)
    return {
        "w2c": np.ascontiguousarray(camera.world_view_transform.T.detach().cpu().double().numpy()),
        "angular_domain": (-tx, tx, -ty, ty), "near": 0.01, "far": 1e10,
        "camera_id": str(camera_token),
    }


class IndexPipeline:
    """Own one frozen model session and immutable geometry/index artifacts.

    R0 executes the original renderer. R1 and R2 execute their own GPU FoV,
    D2H, CPU retrieval/ORI/anchor query, H2D and fresh decode/raster. No step
    is reused from another mode when recording a formal timing.
    """

    def __init__(self, model, checkpoint_path, iteration, mesh_index, anchor_index,
                 *, ori_shape=(128, 128), depth_margin=0.01, session=None, mesh_source_record=None,
                 ori_backend="pixel_depth", tile_size=8, query_device="cpu"):
        validate_environment()
        self.session = session or InferenceSession(model, checkpoint_path, iteration)
        if self.session.model is not model:
            raise ValueError("Session and renderer must own the same frozen model.")
        self.mesh_index = mesh_index
        self.anchor_index = anchor_index
        self.ori_shape = tuple(ori_shape)
        self.depth_margin = float(depth_margin)
        if not math.isfinite(self.depth_margin) or self.depth_margin < 0:
            raise ValueError("Depth margin must be finite and nonnegative.")
        if anchor_index.anchor_count != self.session.scene.anchor_count:
            raise ValueError("Anchor index count does not match the frozen model.")
        if anchor_index.scene_token != self.session.scene.token:
            raise ValueError("Anchor index belongs to a different frozen scene session.")
        if not mesh_source_record or not mesh_source_record.get("validated_depth_binding"):
            raise ValueError("Mesh requires validated checkpoint and complete depth-source binding.")
        if (str(Path(mesh_source_record["model_path"]).resolve()) != self.session.scene.checkpoint_path
                or mesh_source_record["iteration"] != self.session.scene.iteration
                or mesh_source_record["mesh_token"] != mesh_index.mesh_token):
            raise ValueError("Mesh belongs to a different checkpoint or iteration.")
        self.mesh_source_record = dict(mesh_source_record)
        if ori_backend not in ("pixel_depth", "exact_reference"):
            raise ValueError("Unknown ORI definition.")
        self.ori_backend = ori_backend
        self.tile_size = tile_size
        if query_device not in ("cpu", "gpu"):
            raise ValueError("Query device must be cpu or gpu.")
        if query_device == "gpu" and ori_backend != "pixel_depth":
            raise ValueError("GPU queries consume the pixel-depth definition.")
        self.query_device = query_device
        self.gpu_mesh_index = self.gpu_anchor_index = None
        self.depth_rasterizer = None
        initialization_start = perf_counter()
        if ori_backend == "pixel_depth":
            from gdmgs.query.depth_pyramid import MeshDepthRasterizer
            self.depth_rasterizer = MeshDepthRasterizer(mesh_index, device=model.get_anchor.device,
                                                       tile_size=tile_size)
        self.ori_initialization_ms = 1000 * (perf_counter() - initialization_start)
        initialization_start = perf_counter()
        if query_device == "gpu":
            from gdmgs.mesh_index.gpu_index import GPUMeshIndex
            from gdmgs.unified_index.gpu_index import GPUAnchorIndex
            self.gpu_mesh_index = GPUMeshIndex(mesh_index, device=model.get_anchor.device)
            self.gpu_anchor_index = GPUAnchorIndex.from_cpu(anchor_index, device=model.get_anchor.device)
        self.query_gpu_initialization_ms = 1000 * (perf_counter() - initialization_start)
        self.mesh_query_index_token = (self.gpu_mesh_index.index_token if self.gpu_mesh_index is not None else
                                       f"{mesh_index.mesh_token}:cpu:{mesh_index.build_settings['method']}:{mesh_index.build_settings['leaf_size']}")
        self._legacy = importlib.import_module("gaussian_renderer.render")
        self._legacy.reset_render_context()

    def render(self, camera, pipe, background, mode, *, camera_token="", diagnostics=False):
        validate_environment()
        self.session.validate()
        if mode not in ("R0", "R1", "R2"):
            raise ValueError("Index mode must be R0, R1 or R2.")
        if mode == "R0":
            torch.cuda.synchronize()
            start = perf_counter()
            with torch.no_grad():
                package = self._legacy.render(camera, self.session.model, pipe, background,
                                              self.session.scene.iteration, "RGB", disable_cache=True)
            torch.cuda.synchronize()
            elapsed = 1000 * (perf_counter() - start)
            # Collection follows the timed interval. R0 component timings are
            # deliberately unmeasured; its real enclosing renderer is measured.
            fov = (torch.nonzero(package["visible_mask"], as_tuple=False).flatten().cpu().numpy().copy()
                   if diagnostics else np.empty(0, dtype=np.int64))
            times = {name: None for name in ("fov", "d2h", "retrieval", "ori", "anchor",
                                             "h2d", "decode_raster", "query")}
            times["total"] = elapsed
            return IndexFrame(package, times, fov, fov, np.empty(0, dtype=np.int64),
                              np.empty((0, 2), dtype=np.int64), np.empty((0, 2), dtype=np.int64),
                              {"selected": int(package["visible_mask"].sum().item()),
                               "timing_scope": "original renderer; components not instrumented"})

        torch.cuda.synchronize()
        started = perf_counter()
        prepared = prepare_selection(self.session, camera, pipe, background)
        torch.cuda.synchronize()
        after_fov = perf_counter()
        fov_ids = (prepared.anchor_ids if self.query_device == "gpu" else
                   np.ascontiguousarray(prepared.anchor_ids.cpu().numpy(), dtype=np.int64))
        domain = camera_domain(camera, camera_token)
        after_transfer = perf_counter()
        mesh_backend = self.gpu_mesh_index if self.query_device == "gpu" else self.mesh_index
        mesh_query = mesh_backend.query(domain, backend="brute_force" if mode == "R1" else "bvh")
        if self.query_device == "gpu":
            torch.cuda.synchronize()
        after_retrieval = perf_counter()
        if self.ori_backend == "pixel_depth":
            ori = self.depth_rasterizer.build(mesh_query, domain,
                image_size=(int(camera.image_width), int(camera.image_height)),
                download_tiles=self.query_device != "gpu")
            query_image_size = ori.image_size
        else:
            from gdmgs.query.ori import build_ori
            ori = build_ori(self.mesh_index, mesh_query, domain, self.ori_shape)
            query_image_size = (int(camera.image_width), int(camera.image_height))
        if self.query_device == "gpu":
            torch.cuda.synchronize()
        after_ori = perf_counter()
        anchor_backend = self.gpu_anchor_index if self.query_device == "gpu" else self.anchor_index
        query = anchor_backend.query(
            fov_ids, ori.gpu_depth_bounds if self.query_device == "gpu" else ori.depth_bounds,
            domain["w2c"], ori.angular_domain,
            query_image_size,
            mode="linear" if mode == "R1" else "tree", camera_token=camera_token,
            mesh_token=self.mesh_index.mesh_token, depth_margin=self.depth_margin,
        )
        if self.query_device == "gpu":
            torch.cuda.synchronize()
        after_anchor = perf_counter()
        if self.query_device == "gpu":
            selected = query.selected_anchor_ids
        else:
            selected = torch.from_numpy(np.array(query.selected_anchor_ids, copy=True)).to(
                device=prepared.anchor_ids.device, dtype=torch.int64)
            torch.cuda.synchronize()
        after_upload = perf_counter()
        # No Gaussian cache participates in this experiment. Use the shared
        # decoder/raster boundaries directly, retaining its actual request IDs
        # without constructing and repeatedly validating future cache metadata.
        from gaussian_renderer.render import rasterize_batch
        prepared.validate()
        with torch.no_grad():
            batch = self.session.model.generate_neural_gaussians(
                camera, selected, -1, build_descriptor=False,
                pose_state=prepared.pose_state, return_bundle_metadata=False)
            visible = torch.zeros(self.session.scene.anchor_count, dtype=torch.bool, device=selected.device)
            visible[selected] = True
            package = rasterize_batch(camera, self.session.model, batch, background, "RGB",
                                      visible_mask=visible, visibility_summary={"total": int(selected.numel())})
            package.update(selected_anchor_ids=selected, decoded_anchor_ids=batch.anchor_indices,
                           scene_token=self.session.scene.token, pipeline="index-fresh")
        self.session.validate()
        torch.cuda.synchronize()
        finished = perf_counter()
        times = {
            "fov": 1000 * (after_fov - started),
            "d2h": 1000 * (after_transfer - after_fov),
            "retrieval": 1000 * (after_retrieval - after_transfer),
            "ori": 1000 * (after_ori - after_retrieval),
            "anchor": 1000 * (after_anchor - after_ori),
            "h2d": 1000 * (after_upload - after_anchor),
            "decode_raster": 1000 * (finished - after_upload),
            "query": 1000 * (after_upload - started),
            "total": 1000 * (finished - started),
        }
        return IndexFrame(
            package, times, fov_ids, query.selected_anchor_ids, mesh_query.triangle_ids,
            query.raw_ranges, query.formal_ranges,
            {"query_device": self.query_device,
             "scene_token": query.scene_token, "index_token": query.index_token,
             "mesh_query_index_token": self.mesh_query_index_token,
             "camera_token": query.camera_token, "mesh_token": query.mesh_token,
             "anchor_support_profile": query.support_profile, "anchor_support_definition": query.support_definition,
             "payload_transfer_bytes": {
                 "fov_d2h": 0 if self.query_device == "gpu" else prepared.anchor_ids.numel() * prepared.anchor_ids.element_size(),
                 "triangle_ids_d2h": 0,
                 "triangle_ids_h2d": 0 if self.query_device == "gpu" else mesh_query.triangle_ids.nbytes,
                 "ori_tiles_d2h": 0 if self.query_device == "gpu" else ori.depth_bounds.nbytes,
                 "selected_ids_h2d": 0 if self.query_device == "gpu" else selected.numel() * selected.element_size(),
             },
             "fov": len(fov_ids), "selected": len(query.selected_anchor_ids),
             "decoded": int(batch.anchor_indices.numel()),
             "culled": len(fov_ids) - len(query.selected_anchor_ids),
             "triangles": len(mesh_query.triangle_ids), "mesh": mesh_query.counters,
             "ori": ori.counters, "ori_timings": getattr(ori, "timings", {}),
             "anchor": {**query.counters, "support_profile": query.support_profile,
                        "support_definition": query.support_definition}},
        )

    def close(self):
        self.session.close()
        self.depth_rasterizer = None
        self.gpu_mesh_index = self.gpu_anchor_index = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
