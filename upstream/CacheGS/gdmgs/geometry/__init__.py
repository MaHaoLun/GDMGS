"""Frozen-checkpoint geometry candidates; adoption requires image validation."""

from .depth_mesh import (DepthFrame, TriangleTable, backproject, grid_triangles,
                         load_triangle_table, save_triangle_table)

__all__ = ["DepthFrame", "TriangleTable", "backproject", "grid_triangles",
           "load_triangle_table", "save_triangle_table"]
