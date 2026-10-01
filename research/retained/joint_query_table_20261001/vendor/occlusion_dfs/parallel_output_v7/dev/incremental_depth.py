# Reuse the frozen, validated v5 native rasterizer without editing or rebuilding it.
from pathlib import Path
import importlib.util
_spec=importlib.util.spec_from_file_location("incremental_depth_frozen_v5",Path(__file__).resolve().parents[2]/"parallel_output_v5/dev/incremental_depth.py")
_module=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_module)
IncrementalDepth=_module.IncrementalDepth
native=_module.native
