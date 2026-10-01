from pathlib import Path
import importlib.util
_spec=importlib.util.spec_from_file_location("frozen_serial_v6",Path(__file__).resolve().parents[2]/"parallel_output_v6/serial/hybrid.py")
_module=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_module)
SerialHybrid=_module.Hybrid
