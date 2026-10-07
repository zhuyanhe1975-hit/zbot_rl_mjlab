import math

import numpy as np
import torch
from mjlab.envs import mdp as envs_mdp
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

_ASSET_HORIZONTAL_FRAME_FLIP = -1.0


class PartialBaseImpulse(envs_mdp.apply_body_impulse):
    """Run randomized impulses on a subset while leaving others disturbance-free."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        fraction = float(cfg.params.get("disturbed_fraction", 0.75))
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("disturbed_fraction must be in [0, 1]")
        self._disturbed_mask = (
            torch.rand(self._num_envs, device=self._device) < fraction
        )

    def __call__(self, env, env_ids, *args, **kwargs):
        kwargs.pop("disturbed_fraction", None)
        super().__call__(env, env_ids, *args, **kwargs)
        quiet = (~self._disturbed_mask).nonzero(as_tuple=False).flatten()
        if quiet.numel():
            zeros = torch.zeros((len(quiet), self._num_bodies, 3), device=self._device)
            self._asset.write_external_wrench_to_sim(
                zeros, zeros, env_ids=quiet, body_ids=self._body_ids
            )

    def debug_vis(self, visualizer):
        """Draw only actual disturbed bodies, with arrows ending at the base."""
        viz = self._viz_cfg
        body_ids = self._body_ids
        if isinstance(body_ids, slice):
            body_ids = range(self._asset.num_bodies)[body_ids]
        for env_id in visualizer.get_env_indices(self._num_envs):
            if not self._active[env_id] or not self._disturbed_mask[env_id]:
                continue
            for body_id in body_ids:
                force = self._asset.data.body_external_wrench[env_id, body_id, :3]
                if force.norm().item() <= viz.min_force:
                    continue
                point = self._asset.data.body_com_pos_w[env_id, body_id]
                if self._body_point_offset is not None:
                    quat = self._asset.data.body_com_quat_w[env_id, body_id]
                    point = point + quat_apply(quat, self._body_point_offset)
                end = point.detach().cpu().numpy()
                start = end - force.detach().cpu().numpy() * viz.scale
                visualizer.add_arrow(start, end, color=viz.rgba, width=viz.width)


def _base(env):
    robot = env.scene["robot"]
    return robot, robot.find_bodies("base")[0][0]


def reset_scene_to_default(env, env_ids):
    """Reset pose and synchronize PD targets with the configured biped pose."""
    envs_mdp.reset_scene_to_default(env, env_ids)
    envs_mdp.reset_root_state_uniform(
        env,
        env_ids,
        pose_range={"yaw": (-0.0 * math.pi, 0.0 * math.pi)},
        velocity_range={},
    )
    robot = env.scene["robot"]
    if env_ids is None:
        env_ids = slice(None)
    robot.set_joint_position_target(
        robot.data.default_joint_pos[env_ids], env_ids=env_ids
    )


def sampled_frequency(env, min_frequency=None, max_frequency=None):
    fixed = getattr(env.cfg, "test_frequency", None)
    if fixed is not None:
        current = torch.full((env.num_envs,), fixed, device=env.device)
        env._zbot_step_frequency = current
        return current
    min_frequency = (
        env.cfg.step_frequency_min if min_frequency is None else min_frequency
    )
    max_frequency = (
        env.cfg.step_frequency_max if max_frequency is None else max_frequency
    )
    current = getattr(env, "_zbot_step_frequency", None)
    if current is None or current.shape[0] != env.num_envs:
        current = torch.full((env.num_envs,), min_frequency, device=env.device)
    fresh = env.episode_length_buf == 0
    if fresh.any():
        if min_frequency == max_frequency:
            values = torch.full((int(fresh.sum()),), min_frequency, device=env.device)
        elif getattr(env.cfg, "frequency_sampling", "random") == "sequential":
            values = torch.linspace(
                min_frequency, max_frequency, env.num_envs, device=env.device
            )[fresh]
        else:
            values = torch.empty((int(fresh.sum()),), device=env.device).uniform_(
                min_frequency, max_frequency
            )
        current = current.clone()
        current[fresh] = values
    env._zbot_step_frequency = current
    return current


def phase(env, min_frequency=None, max_frequency=None):
    frequency = sampled_frequency(env, min_frequency, max_frequency)
    return (
        env.episode_length_buf.float() * env.step_dt * frequency * 2.0 * math.pi
    ).remainder(2.0 * math.pi)


def phase_observation(env, min_frequency=None, max_frequency=None):
    p = phase(env, min_frequency, max_frequency)
    return torch.stack((torch.sin(p), torch.cos(p)), dim=-1)


def frequency_observation(env, min_frequency=None, max_frequency=None):
    """Expose the sampled episode frequency normalized to [0, 1]."""
    min_frequency = (
        env.cfg.step_frequency_min if min_frequency is None else min_frequency
    )
    max_frequency = (
        env.cfg.step_frequency_max if max_frequency is None else max_frequency
    )
    frequency = sampled_frequency(env, min_frequency, max_frequency)
    span = max(max_frequency - min_frequency, 1e-6)
    return ((frequency - min_frequency) / span).unsqueeze(-1)


def base_lin_vel(env):
    robot, i = _base(env)
    q = robot.data.body_link_quat_w[:, i]
    forward_w, left_w, up_w = _locomotion_axes(q)
    velocity_w = robot.data.body_link_lin_vel_w[:, i]
    return torch.stack(
        (
            (velocity_w * forward_w).sum(-1),
            (velocity_w * left_w).sum(-1),
            (velocity_w * up_w).sum(-1),
        ),
        dim=-1,
    )


def base_ang_vel(env):
    robot, i = _base(env)
    q = robot.data.body_link_quat_w[:, i]
    forward_w, left_w, up_w = _locomotion_axes(q)
    velocity_w = robot.data.body_link_ang_vel_w[:, i]
    return torch.stack(
        (
            (velocity_w * forward_w).sum(-1),
            (velocity_w * left_w).sum(-1),
            (velocity_w * up_w).sum(-1),
        ),
        dim=-1,
    )


def forward_velocity_reward(env, scale=0.2):
    """Bounded forward-speed reward with diminishing returns."""
    speed = base_lin_vel(env)[:, 0].clamp_min(0.0)
    return 1.0 - torch.exp(-speed / scale)


def lateral_velocity_penalty(env):
    """Penalize velocity perpendicular to gravity and horizontal heading."""
    robot, i = _base(env)
    forward_w = _forward_world(env)
    gravity = torch.zeros_like(forward_w)
    gravity[:, 2] = -1.0
    lateral_w = torch.linalg.cross(forward_w, gravity)
    lateral_w = lateral_w / lateral_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    lateral_velocity = (robot.data.body_link_lin_vel_w[:, i] * lateral_w).sum(-1)
    return lateral_velocity.square()


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
            visualizer.add_arrow(
                origin,
                origin + forward_w[j] * scale,
                color=(0.0, 1.0, 1.0, 0.8),
                width=0.012,
            )
            visualizer.add_arrow(
                origin,
                origin + forward_w[j] * (speed[j] * scale),
                color=(0.1, 1.0, 0.2, 0.9),
                width=0.018,
            )


def yaw_ang_vel_observation(env):
    """Yaw rate around gravity in the Zbot locomotion frame."""
    return base_ang_vel(env)[:, 2:3]


def yaw_ang_vel_penalty(env):
    return base_ang_vel(env)[:, 2].square()


def projected_gravity(env):
    robot, i = _base(env)
    q = robot.data.body_link_quat_w[:, i]
    g = torch.zeros((env.num_envs, 3), device=env.device)
    g[:, 2] = -1
    return quat_apply_inverse(q, g)


def upright(env, std=0.45):
    robot, i = _base(env)
    q = robot.data.body_link_quat_w[:, i]
    z = torch.zeros((env.num_envs, 3), device=env.device)
    z[:, 2] = 1
    z_w = quat_apply(q, z)
    return torch.exp(-z_w[:, 2].square() / std**2)


def locomotion_frame(quat):
    return tuple(quat_apply_inverse(quat, axis) for axis in _locomotion_axes(quat))


def _locomotion_axes(quat):
    world_up = torch.zeros_like(quat[..., :3])
    world_up[..., 2] = 1.0
    gravity = -world_up
    body_forward = torch.zeros_like(quat[..., :3])
    body_forward[..., 0] = 1.0
    forward_w = quat_apply(quat, body_forward)
    forward_w[..., 2] = 0.0
    forward_w = forward_w / forward_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    forward_w = forward_w * _ASSET_HORIZONTAL_FRAME_FLIP
    forward_w = forward_w / forward_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    left_w = torch.linalg.cross(forward_w, gravity)
    left_w = left_w / left_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return forward_w, left_w, -gravity


def _forward_world(env):
    robot, i = _base(env)
    q = robot.data.body_link_quat_w[:, i]
    forward_b, _, _ = locomotion_frame(q)
    return quat_apply(q, forward_b)


def _yaw_error(env):
    current = _forward_world(env)
    # All environments aim at world +X (yaw=0), regardless of reset heading.
    return torch.atan2(-current[:, 1], current[:, 0])


def yaw_error_observation(env):
    return _yaw_error(env).unsqueeze(-1)


def yaw_drift_penalty(env):
    """Penalize heading error relative to the fixed world yaw=0 target."""
    return _yaw_error(env).square()


def feet_contact(env, sensor_name="feet_ground_contact"):
    found = env.scene[sensor_name].data.found
    assert found is not None
    return (found > 0).float()


def alternating_foot_phase(
    env, sensor_name="feet_ground_contact", min_frequency=None, max_frequency=None
):
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


def swing_clearance(
    env,
    sensor_name="feet_ground_contact",
    height_sensor_name="foot_height_scan",
    target=0.05,
    min_frequency=None,
    max_frequency=None,
):
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


def foot_self_collision_penalty(env, sensor_name="feet_self_contact", scale=10.0):
    """Bounded penalty for contact force between the two feet."""
    if scale <= 0:
        raise ValueError("Self-collision scale must be positive")
    force = env.scene[sensor_name].data.force
    assert force is not None
    return 1.0 - torch.exp(-force.norm(dim=-1).sum(-1) / scale)


def foot_nonparallel_ground_penalty(env, sensor_name="feet_ground_contact", std=0.25):
    """Penalize contact feet whose sole normal is not aligned with world up."""
    if std <= 0:
        raise ValueError("Foot parallelism std must be positive")
    contact = feet_contact(env, sensor_name)
    robot = env.scene["robot"]
    feet = robot.find_bodies(("foot_0", "foot_1"), preserve_order=True)[0]
    quat = robot.data.body_link_quat_w[:, feet]
    local_z = torch.zeros((*quat.shape[:-1], 3), device=quat.device)
    local_z[..., 2] = 1.0
    normal_w = quat_apply(quat.reshape(-1, 4), local_z.reshape(-1, 3)).reshape_as(
        local_z
    )
    alignment = normal_w[..., 2].clamp(-1.0, 1.0)
    reward = torch.exp(-((1.0 - alignment) / std).square())
    return ((1.0 - reward) * contact).sum(-1) / contact.sum(-1).clamp_min(1.0)


def swing_foot_acceleration_penalty(
    env, acceleration_sensors, acceleration_scale, sensor_name="feet_ground_contact"
):
    """Bounded absolute world-frame acceleration cost for airborne feet only."""
    if acceleration_scale <= 0:
        raise ValueError("Acceleration scale must be positive")
    accelerations = torch.stack(
        [env.scene[name].data for name in acceleration_sensors], dim=1
    )
    cost = (accelerations.abs().sum(-1) / acceleration_scale).clamp(max=1.0)
    return (cost * (1.0 - feet_contact(env, sensor_name))).sum(-1)


def swing_foot_excess_height_penalty(
    env,
    max_height,
    height_scale,
    sensor_name="feet_ground_contact",
    height_sensor_name="foot_height_scan",
):
    """Penalize airborne foot clearance above the allowed terrain-relative height."""
    if max_height < 0 or height_scale <= 0:
        raise ValueError("Height threshold must be non-negative and scale positive")
    heights = env.scene[height_sensor_name].data.heights
    cost = ((heights - max_height).clamp_min(0.0) / height_scale).clamp(max=1.0)
    return (cost * (1.0 - feet_contact(env, sensor_name))).sum(-1)


def joint_acceleration_penalty(env, entity_name="robot"):
    """Sum absolute joint accelerations (rad/s²), without velocity differencing."""
    return env.scene[entity_name].data.joint_acc.abs().sum(dim=-1)


def joint_energy_penalty(env, energy_scale, entity_name="robot"):
    """Bounded mechanical-power cost: sum(abs(torque * joint velocity))."""
    if energy_scale <= 0:
        raise ValueError("Energy scale must be positive")
    asset = env.scene[entity_name]
    power = (asset.data.actuator_force * asset.data.joint_vel).abs().sum(-1)
    return (power / energy_scale).clamp(max=1.0)


def soft_landing(
    env, sensor_name="feet_ground_contact", impact_threshold=5.0, scale=10.0
):
    """Penalize only excessive first-contact impact with a bounded cost."""
    if impact_threshold < 0 or scale <= 0:
        raise ValueError("Impact threshold must be non-negative and scale positive")
    sensor = env.scene[sensor_name]
    data = sensor.data
    assert data.force is not None
    first_contact = sensor.compute_first_contact(dt=env.step_dt)
    impact = data.force.norm(dim=-1)
    excess = (impact - impact_threshold).clamp_min(0.0)
    penalty = (1.0 - torch.exp(-excess / scale)) * first_contact.float()
    return penalty.sum(-1)


class ForwardStepLengthReward:
    """Reward each foot's forward displacement between its touchdowns."""

    def __init__(self, cfg, env):
        self._env = env
        self._sensor_name = cfg.params.get("sensor_name", "feet_ground_contact")
        self._max_step_length = float(cfg.params["max_step_length"])
        self._step_length_scale = float(cfg.params["step_length_scale"])
        self._balance_weight = float(cfg.params["balance_weight"])
        if self._max_step_length <= 0:
            raise ValueError("Maximum step length must be positive")
        if self._step_length_scale <= 0:
            raise ValueError("Step length scale must be positive")
        if self._balance_weight < 0:
            raise ValueError("Balance weight must be non-negative")
        robot = env.scene["robot"]
        self._feet = robot.find_bodies(("foot_0", "foot_1"), preserve_order=True)[0]
        self._has_touchdown = torch.zeros(
            (env.num_envs, len(self._feet)), dtype=torch.bool, device=env.device
        )
        self._previous_touchdown = torch.zeros(
            (env.num_envs, len(self._feet), 3), device=env.device
        )
        self._previous_touchdown_time = torch.zeros(
            (env.num_envs, len(self._feet)), device=env.device
        )
        self._last_step_lengths = torch.zeros(
            (env.num_envs, len(self._feet)), device=env.device
        )
        self._has_step_length = torch.zeros(
            (env.num_envs, len(self._feet)), dtype=torch.bool, device=env.device
        )

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._has_touchdown[env_ids] = False
        self._previous_touchdown[env_ids] = 0.0
        self._previous_touchdown_time[env_ids] = 0.0
        self._last_step_lengths[env_ids] = 0.0
        self._has_step_length[env_ids] = False

    def __call__(self, env, **kwargs):
        del kwargs
        robot = env.scene["robot"]
        positions = robot.data.body_link_pos_w[:, self._feet]
        sensor = env.scene[self._sensor_name]
        touchdown = sensor.compute_first_contact(dt=env.step_dt).bool()
        forward = _forward_world(env).unsqueeze(1)
        displacement = ((positions - self._previous_touchdown) * forward).sum(-1)
        valid = touchdown & self._has_touchdown
        now = (
            env.episode_length_buf.to(self._previous_touchdown_time.dtype) * env.step_dt
        )
        intervals = now.unsqueeze(-1) - self._previous_touchdown_time
        valid &= intervals > 0.0
        frequency_gate = alternating_foot_phase(env, self._sensor_name).clamp(0.0, 1.0)
        step_lengths = displacement.clamp(0.0, self._max_step_length)
        # First contact establishes a world-space anchor, not a measured stride.
        # Keep raw signed strides for symmetry; reward clipping must not hide it.
        updated_lengths = torch.where(valid, displacement, self._last_step_lengths)
        updated_valid = self._has_step_length | valid
        balance_penalty = (
            (updated_lengths[:, 0] - updated_lengths[:, 1]).abs()
            * updated_valid.all(dim=-1)
            * valid.any(dim=-1)
        )
        step_reward = (1.0 - torch.exp(-step_lengths / self._step_length_scale)) * valid
        event_period = torch.where(valid, intervals, torch.zeros_like(intervals))
        score = (step_reward * event_period).sum(-1)
        balance_period = event_period.max(dim=-1).values
        score = (
            score - self._balance_weight * balance_penalty * balance_period
        ) * frequency_gate
        self._previous_touchdown = torch.where(
            touchdown.unsqueeze(-1), positions, self._previous_touchdown
        )
        self._previous_touchdown_time = torch.where(
            touchdown, now.unsqueeze(-1), self._previous_touchdown_time
        )
        self._has_touchdown |= touchdown
        self._last_step_lengths = updated_lengths
        self._has_step_length = updated_valid
        return score / env.step_dt


def action_rate_l2(env):
    return (env.action_manager.action - env.action_manager.prev_action).square().sum(-1)


def joint_pos_limits(env):
    r = env.scene["robot"].data
    return (r.joint_pos_limits[..., 0] - r.joint_pos).clamp_min(0).square().sum(-1) + (
        r.joint_pos - r.joint_pos_limits[..., 1]
    ).clamp_min(0).square().sum(-1)


def fallen(env, height=0.22, tilt=0.75):
    robot, i = _base(env)
    pos = robot.data.body_link_pos_w[:, i] - env.scene.env_origins
    q = robot.data.body_link_quat_w[:, i]
    z = torch.zeros((env.num_envs, 3), device=env.device)
    z[:, 2] = 1
    z_w = quat_apply(q, z)
    return (pos[:, 2] < height) | (z_w[:, 2].abs() > tilt)
