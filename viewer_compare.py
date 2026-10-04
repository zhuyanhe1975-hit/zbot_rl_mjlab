"""Convenience entry point for the local sim-to-sim comparison viewer."""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).parent / "mjlab_mujoco311_bench" / "viewer_compare.py"), run_name="__main__")
