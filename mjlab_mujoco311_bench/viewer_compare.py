"""Two-robot sim-to-sim viewer for Zbot MJLab tasks."""

import argparse
import pathlib
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    choices=(
        "Mjlab-Zbot-6dof-Walking",
        "Mjlab-Zbot-6dof-Periodic-Stepping",
        "Mjlab-Zbot-6dof-Disturbed-Walking",
    ),
    default="Mjlab-Zbot-6dof-Walking",
)
parser.add_argument("--checkpoint", type=pathlib.Path)
parser.add_argument("--frequency", type=float, default=0.5)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--force-scale", type=float, default=1.0)
parser.add_argument("--torque-scale", type=float, default=1.0)
args = parser.parse_args()
TASK_ID = args.task
if args.force_scale < 0 or args.torque_scale < 0:
    raise ValueError("force-scale and torque-scale must be non-negative")
if args.checkpoint is None:
    exp = {
        "Mjlab-Zbot-6dof-Walking": "zbot_walking",
        "Mjlab-Zbot-6dof-Periodic-Stepping": "zbot_periodic_stepping",
        "Mjlab-Zbot-6dof-Disturbed-Walking": "zbot_disturbed_walking",
    }[TASK_ID]
    candidates = list((ROOT.parent / "logs" / "rsl_rl" / exp).rglob("model_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found for {TASK_ID}")
    CHECKPOINT = max(candidates, key=lambda p: p.stat().st_mtime_ns)
else:
    CHECKPOINT = args.checkpoint
print(f"Task: {TASK_ID}\nCheckpoint: {CHECKPOINT}\nFrequency: {args.frequency} Hz")
sys.path.insert(0, str(ROOT.parent / "src"))
sys.path.insert(0, str(ROOT))
from dataclasses import asdict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import zbot_rl_mjlab  # noqa: F401

TASK_ENV_CFG = load_env_cfg(TASK_ID, play=True)
OBS_TERMS = tuple(TASK_ENV_CFG.observations["actor"].terms)
ACTION_SCALE = float(TASK_ENV_CFG.actions["joint_pos"].scale)
print(f"Action scale: {ACTION_SCALE}")

from scene_models import build_comparison_scene

_, m = build_comparison_scene(TASK_ENV_CFG)
d = mujoco.MjData(m)
ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["actor_state_dict"]


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", ck["obs_normalizer._mean"])
        self.register_buffer("std", ck["obs_normalizer._std"])
        weights = {
            k: v
            for k, v in ck.items()
            if k.startswith("mlp.") and k.endswith(".weight")
        }
        layers = []
        for index, name in enumerate(
            sorted(weights, key=lambda x: int(x.split(".")[1]))
        ):
            out_dim, in_dim = weights[name].shape
            layers.append(torch.nn.Linear(in_dim, out_dim))
            if index < len(weights) - 1:
                layers.append(torch.nn.ELU())
        self.net = torch.nn.Sequential(*layers)
        self.net.load_state_dict(
            {k[4:]: v for k, v in ck.items() if k.startswith("mlp.")}
        )

    def forward(self, x):
        return self.net((x - self.mean) / (self.std + 1e-2))


actor = Actor().eval()
default = np.array([0.312, 0.837, -2.02, -2.02, -0.837, -0.312], dtype=np.float64)
speed = args.frequency
# Correct source default for joint4 is +2.02.
default[3] = 2.02


def adr(prefix):
    return [
        m.jnt_qposadr[
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}/joint{i}")
        ]
        for i in range(1, 7)
    ]


def dof(prefix):
    return [
        m.jnt_dofadr[
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}/joint{i}")
        ]
        for i in range(1, 7)
    ]


def body(prefix):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}/base")


rq, nq = adr("ref"), adr("native")
rd, nd = dof("ref"), dof("native")
rb, nb = body("ref"), body("native")


def obs(prefix, b, qadr, dadr, last, step):
    r = d.xmat[b].reshape(3, 3)
    vel = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, vel, 0)
    lin = vel[3:] + np.cross(vel[:3], d.xpos[b] - d.xipos[b])
    body_forward = r[:, 0].copy()
    body_forward[2] = 0
    body_forward /= max(np.linalg.norm(body_forward), 1e-6)
    forward_w = -body_forward
    gravity = np.array([0.0, 0.0, -1.0])
    left_w = np.cross(forward_w, gravity)
    left_w /= max(np.linalg.norm(left_w), 1e-6)
    up_w = -gravity
    lin = np.array([(lin * axis).sum() for axis in (forward_w, left_w, up_w)])
    ang = np.array([(vel[:3] * axis).sum() for axis in (forward_w, left_w, up_w)])
    grav = r.T @ np.array([0.0, 0.0, -1.0])
    z = r[:, 2].copy()
    z[2] = 0
    z /= max(np.linalg.norm(z), 1e-6)
    f = np.cross([0.0, 0.0, -1.0], z)
    f /= max(np.linalg.norm(f), 1e-6)
    yaw = np.arctan2(-forward_w[1], forward_w[0])
    terms = [lin, ang]
    if "yaw_ang_vel" in OBS_TERMS:
        terms.append(np.array([ang[2]]))
    terms.extend(
        [
            grav,
            [yaw],
            d.qpos[qadr] - default,
            d.qvel[dadr],
            last,
            [
                np.sin(step * 0.02 * speed * 2 * np.pi),
                np.cos(step * 0.02 * speed * 2 * np.pi),
            ],
            [(speed - 0.2) / 0.8],
        ]
    )
    value = np.concatenate(terms).astype(np.float32)
    if value.size != int(ck["obs_normalizer._mean"].numel()):
        raise ValueError(
            "Native observation dimension "
            f"{value.size} does not match training config/checkpoint "
            f"{ck['obs_normalizer._mean'].numel()}"
        )
    return value


# external MJLab owner
a = TASK_ENV_CFG
a.test_frequency = args.frequency
a.scene.num_envs = 1
if TASK_ID == "Mjlab-Zbot-6dof-Disturbed-Walking" and "base_disturbance" in a.events:
    params = a.events["base_disturbance"].params
    params["disturbed_fraction"] = 1.0
    params["force_range"] = tuple(v * args.force_scale for v in params["force_range"])
    params["torque_range"] = tuple(
        v * args.torque_scale for v in params["torque_range"]
    )
    print(
        f"Disturbance force scale: {args.force_scale}; torque scale: {args.torque_scale}"
    )
env = ManagerBasedRlEnv(a, device=args.device)
agent = load_rl_cfg(TASK_ID)
wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
runner = MjlabOnPolicyRunner(wrapped, asdict(agent), device=args.device)
runner.load(
    str(CHECKPOINT), load_cfg={"actor": True}, strict=True, map_location=args.device
)
policy = runner.get_inference_policy(device=args.device)
obs_mj = wrapped.reset()[0]
# reset both roots; native at x=0, MJLab visualization at x=1.2
mujoco.mj_resetData(m, d)
for pfx, qadr in (("native", nq), ("ref", rq)):
    d.qpos[qadr] = default
    root = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{pfx}/root")]
    d.qpos[root : root + 7] = [0, -0.5 if pfx == "native" else 0.5, 0, 1, 0, 0, 0]
mujoco.mj_forward(m, d)
last = np.zeros(6)
with mujoco.viewer.launch_passive(m, d) as v:
    # User-scene label geoms stay in the world and identify each controller.
    v.user_scn.ngeom = 2
    for geom, label, pos, color in zip(
        v.user_scn.geoms[:2],
        ("pytorch+mujoco", "mjlab"),
        ((0.0, -0.5, 0.48), (0.0, 0.5, 0.48)),
        ((0.2, 0.7, 1.0, 1.0), (0.2, 1.0, 0.4, 1.0)),
    ):
        geom.type = mujoco.mjtGeom.mjGEOM_LABEL
        geom.pos[:] = pos
        geom.size[:] = (0.12, 0.12, 0.12)
        geom.rgba[:] = color
        geom.label = label
    v.cam.distance = 2.2
    v.cam.azimuth = 135
    v.cam.elevation = -20
    v.cam.lookat[:] = [0, -0.06, 0.2]
    with torch.inference_mode():
        for step in range(1000000):
            if not v.is_running():
                break
            ref_action_t = policy(obs_mj)
            ref_action = ref_action_t[0].cpu().numpy()
            native_action = actor(
                torch.from_numpy(obs("native", nb, nq, nd, last, step))[None]
            ).numpy()[0]
            last = native_action.copy()
            d.ctrl[:] = 0
            for i in range(6):
                d.ctrl[6 + i] = default[i] + ACTION_SCALE * native_action[i]
            for _ in range(4):
                mujoco.mj_step(m, d)
            obs_mj = wrapped.step(ref_action_t)[0]
            # MJLab is display-only in this combined scene: overwrite its qpos after
            # the native robot step so MuJoCo never advances the MJLab reference body.
            src = env.sim.data.qpos[0].detach().cpu().numpy()
            root = m.jnt_qposadr[
                mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ref/root")
            ]
            d.qpos[root : root + 7] = [
                src[0],
                src[1] + 0.5,
                src[2],
                src[3],
                src[4],
                src[5],
                src[6],
            ]
            for i, qa in enumerate(rq):
                d.qpos[qa] = src[7 + i]
            mujoco.mj_forward(m, d)
            # Draw the active MJLab disturbance on the reference robot in pink.
            v.user_scn.ngeom = 2
            if TASK_ID == "Mjlab-Zbot-6dof-Disturbed-Walking":
                env_body = mujoco.mj_name2id(
                    env.sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, "robot/base"
                )
                wrench = env.sim.data.xfrc_applied[0, env_body].detach().cpu().numpy()
                force = wrench[:3]
                if (
                    np.linalg.norm(force) > 1e-5
                    and v.user_scn.ngeom < v.user_scn.maxgeom
                ):
                    start = d.xpos[rb].copy()
                    end = start + force * 0.04
                    geom = v.user_scn.geoms[v.user_scn.ngeom]
                    mujoco.mjv_initGeom(
                        geom,
                        mujoco.mjtGeom.mjGEOM_ARROW.value,
                        np.zeros(3),
                        np.zeros(3),
                        np.zeros(9),
                        np.array((0.95, 0.05, 0.75, 1.0), dtype=np.float32),
                    )
                    mujoco.mjv_connector(
                        geom, mujoco.mjtGeom.mjGEOM_ARROW.value, 0.025, start, end
                    )
                    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
                    v.user_scn.ngeom += 1
            v.user_scn.geoms[0].pos[:] = [
                d.xpos[nb, 0],
                d.xpos[nb, 1],
                d.xpos[nb, 2] + 0.24,
            ]
            v.user_scn.geoms[1].pos[:] = [
                d.xpos[rb, 0],
                d.xpos[rb, 1],
                d.xpos[rb, 2] + 0.24,
            ]
            v.sync()
            time.sleep(0.02)
env.close()
