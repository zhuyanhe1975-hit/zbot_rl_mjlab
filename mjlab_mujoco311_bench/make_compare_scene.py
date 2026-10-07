"""Export a comparison scene using the project's active robot asset."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))
from scene_models import build_comparison_scene

from zbot_rl_mjlab.env_cfg import walking_cfg

if __name__ == "__main__":
    cfg = walking_cfg(play=True)
    scene, model = build_comparison_scene(cfg)
    scene.spec.option.timestep = model.opt.timestep
    scene.spec.option.iterations = model.opt.iterations
    scene.spec.option.ls_iterations = model.opt.ls_iterations
    scene.write(ROOT / "generated_scene")
    print("Exported active asset comparison scene to", ROOT / "generated_scene")
