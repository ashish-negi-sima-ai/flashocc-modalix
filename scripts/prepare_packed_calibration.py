"""Compatibility entry point; implementation is in experiments/packing/."""
from pathlib import Path
import runpy
globals().update(runpy.run_path(str(Path(__file__).resolve().parents[1] / "experiments/packing/prepare_packed_calibration.py"), run_name=__name__))
