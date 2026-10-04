"""Interactive native MuJoCo 3.11 viewer for the MJLab walking actor."""
import pathlib, time
import mujoco
import mujoco.viewer
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent
CHECKPOINT = torch.load(ROOT / 'model.pt', map_location='cpu', weights_only=False)['actor_state_dict']
MEAN = CHECKPOINT['obs_normalizer._mean'].numpy().reshape(-1)
STD = CHECKPOINT['obs_normalizer._std'].numpy().reshape(-1)
class Actor(torch.nn.Module):
    def __init__(self, state):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(31, 256), torch.nn.ELU(), torch.nn.Linear(256, 128), torch.nn.ELU(), torch.nn.Linear(128, 6))
        self.net.load_state_dict({k[4:]: v for k, v in state.items() if k.startswith('mlp.')})
    def forward(self, x): return self.net((x - torch.from_numpy(MEAN)) / (torch.from_numpy(STD) + 1e-2))
actor = Actor(CHECKPOINT).eval()
model = mujoco.MjModel.from_xml_path(str(ROOT / 'model.xml'))
data = mujoco.MjData(model)
default = np.array([.312, .837, -2.02, 2.02, -.837, -.312], dtype=np.float64)
joints = [model.joint(f'robot/joint{i}').id for i in range(1, 7)]
qadr = [model.jnt_qposadr[j] for j in joints]
dadr = [model.jnt_dofadr[j] for j in joints]
qa = model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, 'rl_base_quat')]
aa = model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, 'rl_base_angvel')]
la = model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, 'rl_base_linvel')]
speed = .4
mujoco.mj_resetData(model, data)
for adr, value in zip(qadr, default): data.qpos[adr] = value
mujoco.mj_forward(model, data)
initial_forward = np.array([0., 1., 0.])
previous_action = np.zeros(6, dtype=np.float32)
delta = np.zeros(6, dtype=np.float64)

def observe(step):
    quat = data.sensordata[qa:qa+4]
    rot = np.empty(9); mujoco.mju_quat2Mat(rot, quat); rot = rot.reshape(3, 3)
    gravity = rot.T @ np.array([0., 0., -1.])
    body_z_world = rot[:, 2].copy(); body_z_world[2] = 0.; body_z_world /= max(np.linalg.norm(body_z_world), 1e-6)
    forward_world = np.cross([0., 0., -1.], body_z_world); forward_world /= max(np.linalg.norm(forward_world), 1e-6)
    yaw_error = np.arctan2(forward_world[0] * initial_forward[1] - forward_world[1] * initial_forward[0], np.dot(forward_world[:2], initial_forward[:2]))
    return np.concatenate([
        data.sensordata[la:la+3] @ rot, data.sensordata[aa:aa+3] @ rot, gravity, [yaw_error],
        data.qpos[qadr] - default, data.qvel[dadr], previous_action,
        [np.sin(step*.02*speed*2*np.pi), np.cos(step*.02*speed*2*np.pi)],
        [(speed-.2)/.8],
    ]).astype(np.float32)

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 1.2
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -20
    viewer.cam.lookat[:] = [0, 0, .2]
    step = 0
    with torch.inference_mode():
        while viewer.is_running():
            started = time.perf_counter()
            obs = torch.from_numpy(observe(step))[None]
            action = actor(obs).numpy()[0]
            previous_action[:] = action
            delta = np.clip(delta + np.pi * speed * np.tanh(action) * .02, -np.pi, np.pi)
            data.ctrl[:] = default + delta
            for _ in range(4): mujoco.mj_step(model, data)
            step += 1
            viewer.sync()
            time.sleep(max(0.0, .02 - (time.perf_counter() - started)))
