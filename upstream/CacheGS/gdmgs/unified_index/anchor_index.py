"""PLY-row-preserving anchor partition and conservative CPU occlusion queries.

Only index metadata is retained between frames. Decoder outputs, images, Gaussian
payloads, cache policy and scheduling are not part of this module.
"""
from dataclasses import asdict, dataclass
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np


@dataclass(frozen=True)
class SupportSettings:
    """Bounds for the installed gsplat 1.4 pinhole / classic raster profile.

    Alpha is at most one. Its pixel cutoff is 1/255, so sqrt(2 log(255))
    (3.329...) sigma contains every surviving sample. 3.5 leaves rounding
    slack. eps2d adds variance 0.3; the eigenvalue floor adds at most 0.1.
    Pixel support is determined by the alpha cutoff, not the tile evaluation
    domain. The v3 profile adds blur/eigenvalue-floor plus one rounding pixel;
    legacy artifacts retain their recorded extra 16-pixel tile padding.
    """
    sigma: float = 3.5
    eps2d: float = 0.3
    eigenvalue_floor: float = 0.1
    tile_size: int = 16
    rounding_pixels: float = 1.0
    near_z: float = 0.01
    scale_rounding: str = None
    profile: str = "gsplat-1.4-pinhole-classic-alpha-support-v3"
    definition: str = None

    def __post_init__(self):
        if not math.isfinite(self.sigma) or self.sigma < math.sqrt(2 * math.log(255)):
            raise ValueError("Support sigma must cover the 1/255 alpha cutoff")
        if self.eps2d < 0.3 or self.eigenvalue_floor < 0.1 or self.tile_size < 16:
            raise ValueError("Support settings cannot underbound the installed rasterizer")
        if not all(math.isfinite(v) for v in (self.eps2d, self.eigenvalue_floor, self.rounding_pixels, self.near_z)):
            raise ValueError("Support settings must be finite")
        if self.rounding_pixels < 1 or self.near_z < 0.01:
            raise ValueError("Support rounding/near-plane margin is too small")
        profiles = {
            "gsplat-1.4-pinhole-classic-fp32-fp16-batch":
                ("float16-outward", "all_decoder_offsets_max_last3_scale_full_bundle"),
            "gsplat-1.4-pinhole-classic-alpha-support-v3":
                ("float16-outward", "all_decoder_offsets_max_last3_scale_full_bundle"),
            "native_anchor_proxy_v1":
                ("float32-outward", "native_fov_anchor_first3_scale_image_space_proxy"),
        }
        if self.profile not in profiles:
            raise ValueError("Unsupported renderer support profile")
        expected_rounding, expected_definition = profiles[self.profile]
        if self.scale_rounding not in (None, expected_rounding) or self.definition not in (None, expected_definition):
            raise ValueError("Support profile, definition and scale rounding disagree")
        object.__setattr__(self, "scale_rounding", expected_rounding)
        object.__setattr__(self, "definition", expected_definition)

    @property
    def is_anchor_proxy(self):
        return self.profile == "native_anchor_proxy_v1"

    @property
    def pixel_pad(self):
        tile_padding = self.tile_size if self.profile == "gsplat-1.4-pinhole-classic-fp32-fp16-batch" else 0
        return self.sigma * math.sqrt(self.eps2d + self.eigenvalue_floor) + tile_padding + self.rounding_pixels


def _array(value, dtype, ndim, name):
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype) or value.ndim != ndim:
        raise TypeError(f"{name} requires a {np.dtype(dtype)} ndarray with {ndim} dimensions")
    return np.ascontiguousarray(value)


def _native(native_dir=None):
    location = native_dir or os.environ.get("GDMGS_NATIVE_DIR")
    if location and str(Path(location).resolve()) not in sys.path:
        sys.path.insert(0, str(Path(location).resolve()))
    try:
        return importlib.import_module("_gdmgs_anchor_native")
    except ImportError as exc:
        raise RuntimeError("Build gdmgs/unified_index/native/build.py and set GDMGS_NATIVE_DIR before using the anchor index") from exc


def _support_components(positions, offsets, activated_scales, settings=SupportSettings(), chunk_size=65536):
    """Build the explicitly selected support definition.

    The default encloses all decoder offsets and every possible normalized
    rotation. native_anchor_proxy_v1 instead encloses only the original FoV
    anchor proxy; it does not bound the complete decoded Gaussian bundle.

    Activated scales must be actual float32 decoder activations promoted to
    float64, not float64 exp of checkpoint logarithms. Multiplication/addition
    model the two separate float32 decoder operations. No opacity-dependent
    offset selection is allowed. Nonfinite/overflow scale or offset rows become
    [-inf,+inf] support and can never produce an occlusion certificate.
    """
    positions = _array(positions, np.float64, 2, "positions")
    offsets = _array(offsets, np.float64, 3, "offsets")
    scales = _array(activated_scales, np.float64, 2, "activated_scales")
    n = len(positions)
    if positions.shape != (n, 3) or offsets.shape[0] != n or offsets.shape[2] != 3 or offsets.shape[1] < 1 or scales.shape != (n, 6):
        raise ValueError("Expected positions Nx3, offsets NxKx3 and activated_scales Nx6")
    used_scales = scales[:, :3] if settings.is_anchor_proxy else scales
    if not np.isfinite(positions).all() or np.any(used_scales < 0):
        raise ValueError("Positions must be finite and activated scales nonnegative")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    result = np.empty((n, 6), dtype=np.float64)
    center_bounds = np.empty((n, 6), dtype=np.float64)
    radii = np.empty(n, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        for start in range(0, n, chunk_size):
            end = min(n, start + chunk_size)
            p = positions[start:end].astype(np.float32)
            o = offsets[start:end].astype(np.float32)
            s = scales[start:end].astype(np.float32)
            if settings.is_anchor_proxy:
                # Matches visibility.py's means=anchor, scales=activated[:3].
                # This is an image-space anchor proxy, not a decoder envelope.
                centers = p[:, None, :].astype(np.float64)
                upper_scale = np.nextafter(s[:, :3], np.float32(np.inf)).astype(np.float64)
                valid_inputs = np.isfinite(s[:, :3]).all(axis=1)
            else:
                centers = (p[:, None, :] + o * s[:, None, :3]).astype(np.float64)
                # The decoded product sigmoid*scale is at most scale; batch
                # FP16 conversion can round above it before gsplat upcasts.
                upper_scale = np.nextafter(s[:, 3:].astype(np.float16), np.float16(np.inf)).astype(np.float64)
                valid_inputs = np.isfinite(s).all(axis=1) & np.isfinite(o).all(axis=(1, 2))
            radius = settings.sigma * np.max(upper_scale, axis=1)
            # Quaternion renormalization, float32 multiply/add and camera
            # transform have rounding slack in both world and screen bounds.
            center_error = 64 * np.finfo(np.float32).eps * np.maximum(1, np.max(np.abs(centers), axis=(1, 2)))
            radius = np.nextafter(radius * (1 + 64 * np.finfo(np.float32).eps), np.inf)
            center_lo = np.min(centers, axis=1) - center_error[:, None]
            center_hi = np.max(centers, axis=1) + center_error[:, None]
            lo = center_lo - radius[:, None]
            hi = center_hi + radius[:, None]
            valid = np.isfinite(lo).all(axis=1) & np.isfinite(hi).all(axis=1) & valid_inputs
            result[start:end, :3] = np.where(valid[:, None], np.nextafter(lo, -np.inf), -np.inf)
            result[start:end, 3:] = np.where(valid[:, None], np.nextafter(hi, np.inf), np.inf)
            center_bounds[start:end, :3] = np.where(valid[:, None], np.nextafter(center_lo, -np.inf), -np.inf)
            center_bounds[start:end, 3:] = np.where(valid[:, None], np.nextafter(center_hi, np.inf), np.inf)
            radii[start:end] = np.where(valid, radius, np.inf)
    return result, center_bounds, radii


def support_bounds(positions, offsets, activated_scales, settings=SupportSettings(), chunk_size=65536):
    """Return conservative world bounds; see the support-component contract."""
    return _support_components(positions, offsets, activated_scales, settings, chunk_size)[0]


@dataclass(frozen=True)
class AnchorQueryResult:
    selected_anchor_ids: np.ndarray
    raw_ranges: np.ndarray
    formal_ranges: np.ndarray
    counters: dict
    timings: dict
    scene_token: str
    index_token: str
    camera_token: str
    mesh_token: str
    support_profile: str
    support_definition: str


def _source_record(scene):
    source = {"checkpoint_path": str(Path(scene.checkpoint_path).resolve()),
              "iteration": int(scene.iteration), "anchor_count": int(scene.anchor_count)}
    ply = getattr(scene, "_loaded_ply_path", None)
    if ply is not None:
        path = Path(ply).resolve()
        stat = path.stat()
        source.update(ply_path=str(path), ply_size=stat.st_size, ply_mtime_ns=stat.st_mtime_ns)
    return source


class AnchorIndex:
    """Native object-split octree, immutable PLY binding and exact DFS ranges."""
    FORMAT_VERSION = 4

    def __init__(self, native_tree, scene_token, settings, leaf_capacity, max_depth, build_seconds=0.0):
        self._tree = native_tree
        self.scene_token = str(scene_token)
        self.index_token = f"anchor-octree-v{self.FORMAT_VERSION}:{os.getpid()}:{time.time_ns()}"
        self.source_record = None
        self.support_settings = settings
        self._bound_settings = (settings.definition, settings.sigma, settings.scale_rounding)
        self.leaf_capacity = int(leaf_capacity)
        self.max_depth = int(max_depth)
        self.build_seconds = float(build_seconds)
        layout = self._tree.layout()
        self.dfs_to_row = layout["dfs_to_row"]
        self.rank_of_row = layout["rank_of_row"]
        self.intervals = layout["intervals"]
        for value in (self.dfs_to_row, self.rank_of_row, self.intervals):
            value.flags.writeable = False
        self.anchor_count = len(self.dfs_to_row)
        self.unbounded_support_count = int((~np.isfinite(layout["anchor_support_bounds"]).all(axis=1)).sum())

    @classmethod
    def from_arrays(cls, positions, offsets, activated_scales, scene_token="fixture", leaf_capacity=64,
                    max_depth=32, support_settings=SupportSettings(), native_dir=None):
        start = time.perf_counter()
        p = _array(positions, np.float64, 2, "positions")
        supports, centers, radii = _support_components(p, offsets, activated_scales, support_settings)
        tree = _native(native_dir).AnchorTree(p, supports, centers, radii, leaf_capacity, max_depth)
        return cls(tree, scene_token, support_settings, leaf_capacity, max_depth, time.perf_counter() - start)

    @classmethod
    def from_finalized(cls, scene, leaf_capacity=64, max_depth=32, support_settings=SupportSettings(), native_dir=None):
        """Build once from all original rows, independent of the current FoV."""
        import torch
        rows = scene._rows
        # torch.exp on the original device reproduces decoder activation; an
        # independent CPU exp can differ by an ulp and is not the same bound.
        with torch.no_grad():
            positions = rows["_anchor"].detach().to(device="cpu", dtype=torch.float64).numpy()
            offsets = (np.zeros((scene.anchor_count, 1, 3), dtype=np.float64)
                       if support_settings.is_anchor_proxy else
                       rows["_offset"].detach().to(device="cpu", dtype=torch.float64).numpy())
            scales = torch.exp(rows["_scaling"]).to(device="cpu", dtype=torch.float64).numpy()
        if len(positions) != scene.anchor_count:
            raise ValueError("Finalized row count changed")
        result = cls.from_arrays(positions, offsets, scales, scene.token, leaf_capacity, max_depth, support_settings, native_dir)
        result.source_record = _source_record(scene)
        return result

    def _assert_bound_definition(self):
        current = self.support_settings
        if (current.definition, current.sigma, current.scale_rounding) != self._bound_settings:
            raise ValueError("Support definition or world-bound settings changed; rebuild anchor geometry")

    def save(self, path):
        """Persist actual built topology/support; loading does not repartition."""
        self._assert_bound_definition()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"format_version": self.FORMAT_VERSION, "scene_token": self.scene_token,
                    "index_token": self.index_token, "anchor_count": self.anchor_count, "source_record": self.source_record,
                    "leaf_capacity": self.leaf_capacity, "max_depth": self.max_depth,
                    "support_settings": asdict(self.support_settings), "build_seconds": self.build_seconds}
        with path.open("wb") as stream:
            np.savez(stream, metadata=np.asarray(json.dumps(metadata)), **self.layout())
        return path

    @classmethod
    def load(cls, path, *, scene_token=None, finalized_scene=None, support_profile=None, native_dir=None):
        start = time.perf_counter()
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            if metadata["format_version"] not in {2, 3, cls.FORMAT_VERSION}:
                raise ValueError("Unsupported saved anchor index format")
            saved_settings = metadata["support_settings"]
            if metadata["format_version"] >= 4 and not saved_settings.get("definition"):
                raise ValueError("Saved support definition is missing")
            settings = SupportSettings(**saved_settings)
            if metadata["format_version"] < 4 and settings.is_anchor_proxy:
                raise ValueError("Anchor proxy requires explicit format-4 definition binding")
            if support_profile is not None and support_profile != settings.profile:
                raise ValueError("Saved anchor support profile differs from the requested profile")
            if scene_token is not None and scene_token != metadata["scene_token"]:
                raise ValueError("Saved index belongs to another finalized scene")
            layout = {key: np.ascontiguousarray(data[key]) for key in data.files if key != "metadata"}
        tree = _native(native_dir).AnchorTree.from_layout(layout, metadata["leaf_capacity"], metadata["max_depth"])
        result = cls(tree, metadata["scene_token"], settings,
                     metadata["leaf_capacity"], metadata["max_depth"], metadata["build_seconds"])
        if result.anchor_count != metadata["anchor_count"] or not isinstance(metadata["index_token"], str) or not metadata["index_token"]:
            raise ValueError("Saved index metadata/binding disagree")
        result.index_token = metadata["index_token"]
        result.source_record = metadata.get("source_record")
        if finalized_scene is not None:
            if result.source_record is None or result.source_record != _source_record(finalized_scene):
                raise ValueError("Saved index does not match the finalized checkpoint provenance")
            result.scene_token = finalized_scene.token
        result.load_seconds = time.perf_counter() - start
        return result

    def layout(self):
        """Return owned copies of all topology/binding/support buffers."""
        return self._tree.layout()

    def query(self, fov_ids, depth_bounds, w2c, angular_domain, image_size, *, mode="tree",
              camera_token="", mesh_token="", scene_token=None, depth_margin=0.01, range_state=None):
        start = time.perf_counter()
        self._assert_bound_definition()
        if scene_token is not None and scene_token != self.scene_token:
            raise ValueError("Anchor index belongs to another finalized scene")
        ids = _array(fov_ids, np.int64, 1, "fov_ids")
        depths = _array(depth_bounds, np.float64, 2, "depth_bounds")
        matrix = _array(w2c, np.float64, 2, "w2c")
        output = self._tree.query(ids, depths, matrix, tuple(angular_domain), tuple(image_size), mode,
                                  self.support_settings.pixel_pad, self.support_settings.near_z, depth_margin)
        if range_state is not None:
            # Every camera always discovers all current FoV candidates. State
            # reconciliation never restricts a query to previously selected IDs.
            output["formal_ranges"] = range_state.reconcile(output["formal_ranges"], self.index_token)
        output["timings"]["python_total_ms"] = 1000 * (time.perf_counter() - start)
        return AnchorQueryResult(**output, scene_token=self.scene_token, index_token=self.index_token,
                                 camera_token=str(camera_token), mesh_token=str(mesh_token),
                                 support_profile=self.support_settings.profile,
                                 support_definition=self.support_settings.definition)
