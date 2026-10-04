# MJLab ↔ MuJoCo 3.11 sim-to-sim bench

This temporary bench is outside `ZBotWorkBench`. It loads the packaged `model_599` actor, runs native MuJoCo Python 3.11 dynamics, and compares the state/observation/action stream against the external MJLab environment. The policy is also exported to TorchScript and invoked through a small C++ LibTorch process (`libtorch_smoke.cpp`).

Runtime used:

- MuJoCo Python: 3.11.0
- PyTorch: 2.14.0+cu130
- MJLab: external `~/AI/mjlab`
- Task source: external `~/myWorks_vips/zbot_rl_mjlab`

Run:

```bash
$HOME/AI/mjlab/.venv/bin/python compare.py --steps 600 --device cuda:0 --libtorch
```

The script writes `comparison-cuda-0-libtorch.json`. It disables auto-reset, uses the packaged 31-dimensional normalized actor, and reports observation/action parity plus both base trajectories. `libtorch_smoke` is compiled with the PyTorch headers and libraries from the same virtual environment:

```bash
g++ -std=c++20 -O0 libtorch_smoke.cpp \
  -I$HOME/AI/mjlab/.venv/lib/python3.13/site-packages/torch/include \
  -I$HOME/AI/mjlab/.venv/lib/python3.13/site-packages/torch/include/torch/csrc/api/include \
  -L$HOME/AI/mjlab/.venv/lib/python3.13/site-packages/torch/lib \
  -Wl,-rpath,$HOME/AI/mjlab/.venv/lib/python3.13/site-packages/torch/lib \
  -ltorch -ltorch_cpu -lc10 -o libtorch_smoke
```

Initial result at 600 control steps:

- LibTorch/PyTorch actor parity: max `9.54e-7` action difference.
- Same-state observation parity: max `3.07e-6`.
- Native MuJoCo base height: about `0.060 m` at the end.
- MJLab base height: about `0.273 m` at the end.
- Native MuJoCo base x: about `0.496 m`; MJLab: about `2.709 m`.

This isolates the first conclusion: network loading and observation/action construction are aligned to numerical precision, while the closed-loop dynamics diverge substantially. The next investigation should compare actuator gains, contact settings, timestep/decimation, reset state, and the exact MJLab compiled model before adding domain randomization.

Open-loop action replay (600 control steps) feeds exactly the actions produced by MJLab into both engines. It first exceeds 1 cm base-position error at control step 27 and reaches a maximum base error of about 2.42 m by step 600, isolating a dynamics/model mismatch before closed-loop observation drift.

Interactive viewer:

```bash
$HOME/AI/mjlab/.venv/bin/python viewer.py
```

This opens the native MuJoCo viewer, runs the PyTorch actor online, and advances MuJoCo 3.11 at 240 Hz physics / 50 Hz control. Close the viewer window to stop the loop.

Dual comparison viewer:

```bash
$HOME/AI/mjlab/.venv/bin/python viewer_compare.py
```

This scene contains two robots with labels above them: `pytorch+mujoco` is the standalone PyTorch actor driving native MuJoCo 3.11; `mjlab` is the MJLab policy/state stream copied into the second display robot. Both are rendered by the same MuJoCo viewer and share the same camera.

The comparison uses the current MJLab task contract: 31-dimensional observation, 0.005 s physics step, 4-step decimation, 0.25 direct joint-position action scale, 15/1.5 actuator gains, and a 0.4 Hz test frequency. The MJLab robot is display-only in the combined scene and is overwritten from MJLab qpos after every native MuJoCo step.
