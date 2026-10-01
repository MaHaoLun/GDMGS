"""GPU-resident anchor certificates and ranges with an exact 2D max table.

R2 certifies all nonempty nodes in parallel, then removes maximal certified
intervals and certifies remaining FoV rows. This is a batch GPU strategy, not
an early-exit CPU DFS traversal. No decoded Gaussian payload is retained.
"""
from dataclasses import dataclass, field
import importlib
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

_COUNTER_NAMES = ("visited_nodes", "empty_nodes", "certified_nodes", "anchor_checks",
                  "interval_removed_anchors", "certified_anchors", "unbounded", "near_plane",
                  "unknown_cells", "depth_fail", "outside", "early_unknown_cells",
                  "rectangle_queries", "maximal_certified_nodes")


def _native():
    directory = os.environ.get("GDMGS_GPU_NATIVE_DIR")
    if directory is None and os.environ.get("GDMGS_NATIVE_DIR"):
        directory = str(Path(os.environ["GDMGS_NATIVE_DIR"]) / "anchor_cuda")
    if directory and directory not in sys.path:
        sys.path.insert(0, directory)
    try:
        return importlib.import_module("_gdmgs_anchor_cuda")
    except ImportError as exc:
        raise RuntimeError("Build gdmgs/unified_index/cuda/build.py and set GDMGS_GPU_NATIVE_DIR") from exc


@dataclass(frozen=True)
class GPUAnchorQueryResult:
    selected_anchor_ids: torch.Tensor
    raw_ranges: torch.Tensor
    formal_ranges: torch.Tensor
    scene_token: str
    index_token: str
    camera_token: str
    mesh_token: str
    support_profile: str
    support_definition: str
    _counts: torch.Tensor = field(repr=False)
    _events: dict = field(repr=False)
    _metadata: dict = field(repr=False)
    _host_ms: float = field(repr=False)

    def collect_counters(self):
        """Download small diagnostics after the enclosing fresh frame completes."""
        values = self._counts.detach().cpu().tolist()
        result = dict(zip(_COUNTER_NAMES, values))
        result.update(self._metadata)
        result.update(support_profile=self.support_profile, support_definition=self.support_definition,
                      scene_token=self.scene_token, index_token=self.index_token,
                      camera_token=self.camera_token, mesh_token=self.mesh_token,
                      actual_device=str(self.selected_anchor_ids.device))
        result["selected_count"] = self.selected_anchor_ids.shape[0]
        result["raw_range_count"] = self.raw_ranges.shape[0]
        result["formal_range_count"] = self.formal_ranges.shape[0]
        result["sparse_table_rectangle_reads"] = 4 * result["rectangle_queries"]
        if result["certified_anchors"] != result["fov_count"] - result["selected_count"]:
            raise RuntimeError("GPU selection/certificate counters disagree")
        return result

    @property
    def counters(self):
        return self.collect_counters()

    @property
    def timings(self):
        """Collect event times only after all timed query work has completed."""
        if not self._events["finish"].query():
            raise RuntimeError("Collect GPU query timings after the enclosing frame synchronizes")
        pairs = (("start", "table", "sparse_table_ms"), ("table", "prepare", "prepare_ms"),
                 ("prepare", "nodes", "node_certification_ms"),
                 ("nodes", "intervals", "interval_prefix_ms"),
                 ("intervals", "anchors", "anchor_certification_ms"),
                 ("anchors", "finish", "materialization_ms"),
                 ("start", "finish", "gpu_total_ms"))
        result = {name: self._events[a].elapsed_time(self._events[b]) for a, b, name in pairs}
        result["host_launch_and_sync_ms"] = self._host_ms
        return result


def _tensor_guard(value):
    return (id(value), value.data_ptr(), value._version, tuple(value.shape), value.dtype, value.device)


class GPUAnchorIndex:
    """Static row binding/support on GPU; explicit fresh query outputs only."""

    @classmethod
    def from_cpu(cls, index, device="cuda:0"):
        index._assert_bound_definition()
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("GPUAnchorIndex requires a CUDA device")
        result = cls.__new__(cls)
        result._extension = _native()
        result.device = device
        result.scene_token, result.index_token = index.scene_token, index.index_token
        result.anchor_count = index.anchor_count
        result.support_settings = index.support_settings
        result._bound_settings = (index.support_settings.definition, index.support_settings.sigma,
                                  index.support_settings.scale_rounding)
        layout = index.layout()
        start = perf_counter()
        parents = np.full(len(layout["intervals"]), -1, dtype=np.int64)
        for node, children in enumerate(layout["children"]):
            parents[children[children >= 0]] = node
        result._tensors = {}
        with torch.cuda.device(device):
            for name in ("dfs_to_row", "rank_of_row", "intervals", "anchor_support_bounds",
                         "anchor_center_bounds", "anchor_radii", "support_bounds",
                         "node_center_bounds", "node_radii"):
                result._tensors[name] = torch.tensor(layout[name], device=device)
            result._tensors["parents"] = torch.tensor(parents, device=device)
            for prefix, bounds, centers, radii in (
                ("anchor", "anchor_support_bounds", "anchor_center_bounds", "anchor_radii"),
                ("node", "support_bounds", "node_center_bounds", "node_radii"),
            ):
                known = (np.isfinite(layout[bounds]).all(axis=1) & np.isfinite(layout[centers]).all(axis=1)
                         & np.isfinite(layout[radii]))
                result._tensors[prefix + "_known"] = torch.tensor(known.astype(np.uint8), device=device)
            torch.cuda.synchronize(device)
        result.initialization_ms = (perf_counter() - start) * 1000
        result.resident_bytes = sum(t.numel() * t.element_size() for t in result._tensors.values())
        result._guards = {name: _tensor_guard(tensor) for name, tensor in result._tensors.items()}
        return result

    def _validate(self):
        settings = self.support_settings
        if (settings.definition, settings.sigma, settings.scale_rounding) != self._bound_settings:
            raise ValueError("GPU support definition changed; rebuild/re-upload anchor geometry")
        if (self._tensors.keys() != self._guards.keys() or
                any(_tensor_guard(value) != self._guards[key] for key, value in self._tensors.items())):
            raise RuntimeError("GPU anchor index geometry or row binding was modified")

    def query(self, fov_ids, depth_bounds, w2c, angular_domain, image_size, *, mode="tree",
              camera_token="", mesh_token="", scene_token=None, depth_margin=.01):
        self._validate()
        if scene_token is not None and scene_token != self.scene_token:
            raise ValueError("GPU anchor index belongs to a different scene")
        if mode not in {"tree", "linear"}:
            raise ValueError("mode must be tree or linear")
        if not isinstance(fov_ids, torch.Tensor) or fov_ids.dtype != torch.int64 or fov_ids.ndim != 1:
            raise TypeError("fov_ids must be a one-dimensional CUDA int64 tensor")
        if (not isinstance(depth_bounds, torch.Tensor) or depth_bounds.dtype not in {torch.float32, torch.float64}
                or depth_bounds.ndim != 2):
            raise TypeError("depth_bounds must be a CUDA float32/64 [H,W] tensor")
        if fov_ids.device != self._tensors["rank_of_row"].device or depth_bounds.device != fov_ids.device:
            raise ValueError("FoV, ORI and anchor index must share the same CUDA device")
        if (not isinstance(w2c, np.ndarray) or w2c.dtype != np.float64 or w2c.shape != (4, 4)
                or not np.isfinite(w2c).all() or not np.array_equal(w2c[3], [0, 0, 0, 1])):
            raise ValueError("w2c must be a finite numpy float64 affine [4,4] matrix")
        started = perf_counter()
        with torch.cuda.device(fov_ids.device), torch.no_grad():
            events = {name: torch.cuda.Event(enable_timing=True) for name in
                      ("start", "table", "prepare", "nodes", "intervals", "anchors", "finish")}
            events["start"].record()
            ids, depth = fov_ids.contiguous(), depth_bounds.contiguous()
            torch._assert_async(torch.all((ids >= 0) & (ids < self.anchor_count)), "FoV IDs exceed anchor rows")
            torch._assert_async(torch.all((depth > 0) & ~torch.isnan(depth)), "ORI needs positive depth or +inf Unknown")
            table = self._extension.sparse_max(depth)
            events["table"].record()
            t = self._tensors
            ranks = t["rank_of_row"][ids]
            fov_marks = torch.zeros(self.anchor_count, dtype=torch.int32, device=ids.device)
            fov_marks.scatter_add_(0, ranks, torch.ones_like(ranks, dtype=torch.int32))
            torch._assert_async(torch.all(fov_marks <= 1), "FoV IDs must be unique")
            counts = torch.zeros(len(_COUNTER_NAMES), dtype=torch.int64, device=ids.device)
            cam = torch.from_numpy(np.array(w2c, copy=True, order="C"))
            common = (table, cam, tuple(angular_domain), tuple(image_size),
                      self.support_settings.pixel_pad, self.support_settings.near_z, float(depth_margin), counts)
            if mode == "tree":
                prefix = torch.cat((torch.zeros(1, dtype=torch.int64, device=ids.device),
                                    fov_marks.cumsum(0, dtype=torch.int64)))
            events["prepare"].record()
            if mode == "tree":
                flags = self._extension.certify_nodes(t["support_bounds"], t["node_center_bounds"],
                    t["node_radii"], t["node_known"], t["intervals"], prefix, *common)
            events["nodes"].record()
            if mode == "tree":
                delta = self._extension.maximal_intervals(flags, t["parents"], t["intervals"],
                                                          self.anchor_count, counts)
                removed_rank = delta[:-1].cumsum(0, dtype=torch.int32) > 0
            else:
                removed_rank = torch.empty(0, dtype=torch.bool, device=ids.device)
            events["intervals"].record()
            removed = self._extension.certify_anchors(t["anchor_support_bounds"], t["anchor_center_bounds"],
                t["anchor_radii"], t["anchor_known"], ids, removed_rank, t["rank_of_row"], *common)
            events["anchors"].record()
            keep = removed == 0
            selected = ids[keep]
            # Canonical selected DFS runs are both the raw and formal GPU
            # denotation. Construction (including necessary nonzero size sync)
            # occurs inside the measured query, never deferred to diagnostics.
            kept_ranks = torch.zeros(self.anchor_count, dtype=torch.bool, device=ids.device)
            kept_ranks[ranks] = keep
            edge = torch.cat((torch.zeros(1, dtype=torch.bool, device=ids.device), kept_ranks,
                              torch.zeros(1, dtype=torch.bool, device=ids.device)))
            starts = torch.nonzero(edge[1:] & ~edge[:-1], as_tuple=False).flatten()
            ends = torch.nonzero(edge[:-1] & ~edge[1:], as_tuple=False).flatten()
            formal = torch.stack((starts, ends), dim=1)
            events["finish"].record()
        metadata = {"fov_count": ids.shape[0], "anchor_count": self.anchor_count, "query_device": "gpu",
                    "strategy": "batch_all_nonempty_nodes_then_uncertified_anchors" if mode == "tree" else "parallel_per_fov_anchor",
                    "range_definition": "canonical_selected_DFS_runs", "sparse_table_elements": table.numel(),
                    "sparse_table_bytes": table.numel() * table.element_size(), "resident_index_bytes": self.resident_bytes,
                    "fov_id_upload_bytes": 0, "fov_id_download_bytes": 0,
                    "selected_id_upload_bytes": 0, "selected_id_download_bytes": 0,
                    "range_upload_bytes": 0, "range_download_bytes": 0,
                    "ori_payload_upload_bytes": 0, "ori_payload_download_bytes": 0,
                    "logging_download_bytes": counts.numel() * counts.element_size(),
                    "algorithm_shape_sync": "CUDA boolean selection and two nonzero output-size operations are timed",
                    "camera_input_host_bytes": 16 * 8 + 4 * 8 + 2 * 8 + 3 * 8,
                    "camera_kernel_parameter_bytes": 232 * (int(ids.shape[0] > 0) +
                        int(mode == "tree" and t["intervals"].shape[0] > 0))}
        return GPUAnchorQueryResult(selected, formal, formal, self.scene_token, self.index_token,
            str(camera_token), str(mesh_token), self.support_settings.profile, self.support_settings.definition,
            counts, events, metadata, (perf_counter() - started) * 1000)
