import math

import numpy as np
import torch
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse


def _base(env):
    robot = env.scene["robot"]
    return robot, robot.find_bodies("base")[0][0]

def sampled_frequency(env, min_frequency=None, max_frequency=None):
    fixed = getattr(env.cfg, "test_frequency", None)
    if fixed is not None:
        current = torch.full((env.num_envs,), fixed, device=env.device)
        env._zbot_step_frequency = current
        return current
    min_frequency = env.cfg.step_frequency_min if min_frequency is None else min_frequency
    max_frequency = env.cfg.step_frequency_max if max_frequency is None else max_frequency
    current = getattr(env, "_zbot_step_frequency", None)
    if current is None or current.shape[0] != env.num_envs:
        current = torch.full((env.num_envs,), min_frequency, device=env.device)
    fresh = env.episode_length_buf == 0
    if fresh.any():
        if min_frequency == max_frequency:
            values = torch.full((int(fresh.sum()),), min_frequency, device=env.device)
        else:
            values = torch.empty((int(fresh.sum()),), device=env.device).uniform_(min_frequency, max_frequency)
        current = current.clone()
        current[fresh] = values
    env._zbot_step_frequency = current
    return current

def phase(env, min_frequency=None, max_frequency=None):
    frequency = sampled_frequency(env, min_frequency, max_frequency)
    return (env.episode_length_buf.float() * env.step_dt * frequency * 2.0 * math.pi).remainder(2.0 * math.pi)

def phase_observation(env, min_frequency=None, max_frequency=None):
    p = phase(env, min_frequency, max_frequency)
    return torch.stack((torch.sin(p), torch.cos(p)), dim=-1)

def frequency_observation(env, min_frequency=None, max_frequency=None):
    """Expose the sampled episode frequency normalized to [0, 1]."""
    min_frequency = env.cfg.step_frequency_min if min_frequency is None else min_frequency
    max_frequency = env.cfg.step_frequency_max if max_frequency is None else max_frequency
    frequency = sampled_frequency(env, min_frequency, max_frequency)
    span = max(max_frequency - min_frequency, 1e-6)
    return ((frequency - min_frequency) / span).unsqueeze(-1)

def base_lin_vel(env):
    robot, i = _base(env); q = robot.data.body_link_quat_w[:, i]
    return quat_apply_inverse(q, robot.data.body_link_lin_vel_w[:, i])

def forward_velocity_reward(env, scale=0.2):
    """Bounded forward-speed reward with diminishing returns."""
    speed = base_lin_vel(env)[:, 0].clamp_min(0.0)
    return 1.0 - torch.exp(-speed / scale)

def lateral_velocity_penalty(env):
    """Penalize sideways motion in the Base-defined left direction."""
    return base_lin_vel(env)[:, 1].square()

class ForwardVelocityVisualizer(UniformVelocityCommand):
    """Viewer-only arrows for the Base forward axis and measured speed."""

    def _resample_command(self, env_ids):
        self.vel_command_b[env_ids] = 0.0

    def _debug_vis_impl(self, visualizer):
        indices = visualizer.get_env_indices(self.num_envs)
        if not indices:
            return
        robot, i = _base(self._env)
        q = robot.data.body_link_quat_w[:, i]
        forward_b, _, _ = locomotion_frame(q)
        forward_w = quat_apply(q, forward_b).cpu().numpy()
        pos = robot.data.body_link_pos_w[:, i].cpu().numpy()
        speed = base_lin_vel(self._env)[:, 0].detach().cpu().numpy()
        for j in indices:
            origin = pos[j] + np.array([0.0, 0.0, 0.12])
            scale = 0.25
            visualizer.add_arrow(origin, origin + forward_w[j] * scale, color=(0.0, 1.0, 1.0, 0.8), width=0.012)
            visualizer.add_arrow(origin, origin + forward_w[j] * (speed[j] * scale), color=(0.1, 1.0, 0.2, 0.9), width=0.018)

def base_ang_vel(env):
    robot, i = _base(env); q = robot.data.body_link_quat_w[:, i]
    return quat_apply_inverse(q, robot.data.body_link_ang_vel_w[:, i])

def projected_gravity(env):
    robot, i = _base(env); q = robot.data.body_link_quat_w[:, i]
    g = torch.zeros((env.num_envs, 3), device=env.device); g[:, 2] = -1
    return quat_apply_inverse(q, g)

def upright(env, std=0.45):
    robot, i = _base(env); q = robot.data.body_link_quat_w[:, i]
    z = torch.zeros((env.num_envs, 3), device=env.device); z[:, 2] = 1
    z_w = quat_apply(q, z)
    return torch.exp(-z_w[:, 2].square() / std**2)

def locomotion_frame(quat):
    world_up = torch.zeros_like(quat[..., :3]); world_up[..., 2] = 1.0
    gravity = -world_up
    body_z = torch.zeros_like(quat[..., :3]); body_z[..., 2] = 1.0
    body_z_w = quat_apply(quat, body_z); body_z_w[..., 2] = 0.0
    body_z_w = body_z_w / body_z_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    forward_w = torch.linalg.cross(gravity, body_z_w)
    forward_w = forward_w / forward_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    left_w = torch.linalg.cross(forward_w, gravity)
    left_w = left_w / left_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return quat_apply_inverse(quat, forward_w), quat_apply_inverse(quat, left_w), quat_apply_inverse(quat, gravity)

def _forward_world(env):
    robot, i = _base(env)
    q = robot.data.body_link_quat_w[:, i]
    forward_b, _, _ = locomotion_frame(q)
    return quat_apply(q, forward_b)

def _yaw_error(env):
    current = _forward_world(env)
    initial = getattr(env, "_zbot_initial_forward_w", None)
    if initial is None or initial.shape != current.shape:
        initial = current.clone()
    fresh = env.episode_length_buf == 0
    initial = torch.where(fresh.unsqueeze(-1), current, initial)
    env._zbot_initial_forward_w = initial
    cross_z = current[:, 0] * initial[:, 1] - current[:, 1] * initial[:, 0]
    dot = (current[:, :2] * initial[:, :2]).sum(-1)
    return torch.atan2(cross_z, dot)

def yaw_error_observation(env):
    return _yaw_error(env).unsqueeze(-1)

def yaw_drift_penalty(env):
    """Penalize yaw drift from each episode's initial forward direction."""
    return _yaw_error(env).square()

def feet_contact(env, sensor_name="feet_ground_contact"):
    found = env.scene[sensor_name].data.found
    assert found is not None
    return (found > 0).float()

def alternating_foot_phase(env, sensor_name="feet_ground_contact", min_frequency=None, max_frequency=None):
    p = phase(env, min_frequency, max_frequency)
    force = env.scene[sensor_name].data.force
    assert force is not None
    support = force.norm(dim=-1)
    left_force, right_force = support[:, 0], support[:, 1]
    total = left_force + right_force
    measured = (left_force - right_force) / total.clamp_min(1e-6)
    target = torch.sin(p)
    matched = 1.0 - (measured - target).square()
    return matched.clamp_min(0.0) * (total > 1e-3).float()

def swing_clearance(env, sensor_name="feet_ground_contact", height_sensor_name="foot_height_scan", target=0.05, min_frequency=None, max_frequency=None):
    del sensor_name
    p = phase(env, min_frequency, max_frequency)
    desired_left = (torch.sin(p) < 0).float()
    swing = torch.stack((1.0 - desired_left, desired_left), dim=-1)
    heights = env.scene[height_sensor_name].data.heights
    return (swing * torch.exp(-((heights - target) / 0.04).square())).mean(-1)

def support_foot_slip(env, sensor_name="feet_ground_contact"):
    contact = feet_contact(env, sensor_name)
    robot = env.scene["robot"]
    feet = robot.find_bodies(("foot_0", "foot_1"), preserve_order=True)[0]
    speed_sq = robot.data.body_link_lin_vel_w[:, feet, :2].square().sum(-1)
    return (contact * speed_sq).sum(-1)

def soft_landing(env, sensor_name="feet_ground_contact"):
    """Penalize impact force only on the first contact after swing."""
    sensor = env.scene[sensor_name]
    data = sensor.data
    assert data.force is not None
    first_contact = sensor.compute_first_contact(dt=env.step_dt)
    impact = data.force.norm(dim=-1) * first_contact.float()
    return impact.sum(-1)

def action_rate_l2(env):
    return (env.action_manager.action - env.action_manager.prev_action).square().sum(-1)

def joint_pos_limits(env):
    r = env.scene["robot"].data
    return (r.joint_pos_limits[..., 0] - r.joint_pos).clamp_min(0).square().sum(-1) + (r.joint_pos - r.joint_pos_limits[..., 1]).clamp_min(0).square().sum(-1)

def fallen(env, height=0.22, tilt=0.75):
    robot, i = _base(env); pos = robot.data.body_link_pos_w[:, i] - env.scene.env_origins
    q = robot.data.body_link_quat_w[:, i]
    z = torch.zeros((env.num_envs, 3), device=env.device); z[:, 2] = 1
    z_w = quat_apply(q, z)
    return (pos[:, 2] < height) | (z_w[:, 2].abs() > tilt)
