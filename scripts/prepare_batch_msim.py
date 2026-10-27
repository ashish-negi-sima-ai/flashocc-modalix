"""Compatibility entry point; implementation is in experiments/batch6/."""
from pathlib import Path
import runpy
globals().update(runpy.run_path(str(Path(__file__).resolve().parents[1] / "experiments/batch6/prepare_batch_msim.py"), run_name=__name__))
