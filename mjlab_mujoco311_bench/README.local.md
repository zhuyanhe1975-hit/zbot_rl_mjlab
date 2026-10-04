# Zbot sim-to-sim comparison

This directory contains the core files copied from the local `mjlab_mujoco311_bench` bundle. It compares a standalone MuJoCo 3.11 policy loop with the MJLab state loop in one Viewer.

From the project root:

```bash
source /home/yhzhu/AI/mjlab/.venv/bin/activate
python viewer_compare.py --task walking --frequency 0.5
```

Tasks:

```bash
python viewer_compare.py --task stepping
python viewer_compare.py --task walking
python viewer_compare.py --task disturbed-walking
```

Use `--checkpoint PATH` to override automatic loading. Without it, the newest checkpoint under the matching `logs/rsl_rl/<experiment>/` directory is selected. The copied scene currently displays two robots: standalone MuJoCo 3.11 and the MJLab state stream.

The comparison model/scene must have the same 31-dimensional actor contract as the current tasks. Regenerate `compare_scene.xml` with `make_compare_scene.py` if the asset topology changes.
