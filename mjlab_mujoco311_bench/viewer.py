"""Interactive native MuJoCo 3.11 viewer for the MJLab walking actor."""

import pathlib
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent
CHECKPOINT = torch.load(ROOT / "model.pt", map_location="cpu", weights_only=False)[
    "actor_state_dict"
]
MEAN = CHECKPOINT["obs_normalizer._mean"].numpy().reshape(-1)
STD = CHECKPOINT["obs_normalizer._std"].numpy().reshape(-1)


class Actor(torch.nn.Module):
    def __init__(self, state):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(31, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 6),
        )
        self.net.load_state_dict(
            {k[4:]: v for k, v in state.items() if k.startswith("mlp.")}
        )

    def forward(self, x):
        return self.net((x - torch.from_numpy(MEAN)) / (torch.from_numpy(STD) + 1e-2))


actor = Actor(CHECKPOINT).eval()
sys.path.insert(0, str(ROOT.parent / "src"))
from compare import forward_axis
from compare import observe as native_observe
from mjlab.scene import Scene

from zbot_rl_mjlab.env_cfg import walking_cfg

cfg = walking_cfg(play=True)
model = Scene(cfg.scene, device="cpu").compile()
cfg.sim.mujoco.apply(model)
data = mujoco.MjData(model)
default = np.array([0.312, 0.837, -2.02, 2.02, -0.837, -0.312], dtype=np.float64)
joints = [model.joint(f"robot/joint{i}").id for i in range(1, 7)]
qadr = [model.jnt_qposadr[j] for j in joints]
dadr = [model.jnt_dofadr[j] for j in joints]
speed = 0.4
ACTION_SCALE = 0.05
mujoco.mj_resetData(model, data)
for adr, value in zip(qadr, default):
    data.qpos[adr] = value
mujoco.mj_forward(model, data)
initial_forward = forward_axis(data.xquat[model.body("robot/base").id])
previous_action = np.zeros(6, dtype=np.float32)
delta = np.zeros(6, dtype=np.float64)


def observe(step):
    return native_observe(
        model, data, default, previous_action, step, speed, initial_forward
    )


with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 1.2
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -20
    viewer.cam.lookat[:] = [0, 0, 0.2]
    step = 0
    with torch.inference_mode():
        while viewer.is_running():
            started = time.perf_counter()
            obs = torch.from_numpy(observe(step))[None]
            action = actor(obs).numpy()[0]
            previous_action[:] = action
            data.ctrl[:] = default + ACTION_SCALE * action
            for _ in range(4):
                mujoco.mj_step(model, data)
            step += 1
            viewer.sync()
            time.sleep(max(0.0, 0.02 - (time.perf_counter() - started)))
