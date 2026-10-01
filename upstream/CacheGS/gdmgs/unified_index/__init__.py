"""Conservative anchor supports, independent octree partition and DFS ranges."""
from .anchor_index import AnchorIndex, AnchorQueryResult, SupportSettings, support_bounds

__all__ = ["AnchorIndex", "AnchorQueryResult", "SupportSettings", "support_bounds"]


def __getattr__(name):
    if name in {"GPUAnchorIndex", "GPUAnchorQueryResult"}:
        from .gpu_index import GPUAnchorIndex, GPUAnchorQueryResult
        return {"GPUAnchorIndex": GPUAnchorIndex, "GPUAnchorQueryResult": GPUAnchorQueryResult}[name]
    raise AttributeError(name)
