"""MDP terms matching the original six-DOF quaternion task.

Source: zbot_rl_student curriculum_env.Zbot6DofQuatEnv and shared_rewards.
Isaac Lab body_lin_vel_w is COM velocity; body pose is the link frame.
"""

import torch
import numpy as np
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
from mjlab.utils.lab_api.math import matrix_from_quat
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand


def walking_frame(quat):
    """Preserve the source's gravity-cross-local-Z forward vector, unnormalized."""
    axis_z = torch.zeros_like(quat[..., :3])
    axis_z[..., 2] = 1
    gravity = quat_apply_inverse(quat, -axis_z)
    forward_b = torch.cross(gravity, axis_z, dim=-1)
    heading_error = -quat_apply(quat, forward_b)[..., 1]
    return gravity, forward_b, heading_error


def _base(env):
    robot = env.scene["robot"]
    return robot, robot.find_bodies("base")[0][0]


def locomotion_frame(quat):
    """Return forward, left and up axes in the base body frame."""
    down = quat_apply_inverse(quat, torch.tensor([0.0, 0.0, -1.0], device=quat.device).expand(quat.shape[0], -1))
    up = -down / down.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    lateral_reference = torch.tensor([0.0, 0.0, 1.0], device=quat.device).expand_as(up)
    forward = torch.linalg.cross(down, lateral_reference)
    forward = forward / forward.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    left = torch.linalg.cross(up, forward)
    left = left / left.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return forward, left, up


def base_lin_vel(env):
    robot, body_id = _base(env)
    q = robot.data.body_link_quat_w[:, body_id]
    vel = quat_apply_inverse(q, robot.data.body_link_lin_vel_w[:, body_id])
    forward, left, up = locomotion_frame(q)
    current = torch.stack((torch.sum(vel * forward, -1), torch.sum(vel * left, -1), torch.sum(vel * up, -1)), -1)
    step = int(getattr(env, "common_step_counter", 0))
    if getattr(env, "_zbot_lin_filter_step", -1) != step:
        state = getattr(env, "_zbot_lin_vel_ema", None)
        if state is None or state.shape != current.shape:
            state = current.clone()
        else:
            state.mul_(0.9).add_(current, alpha=0.1)
        env._zbot_lin_vel_ema = state
        env._zbot_lin_filter_step = step
    return env._zbot_lin_vel_ema


def base_ang_vel(env):
    robot, body_id = _base(env)
    q = robot.data.body_link_quat_w[:, body_id]
    vel = quat_apply_inverse(q, robot.data.body_link_ang_vel_w[:, body_id])
    forward, left, up = locomotion_frame(q)
    current = torch.stack((torch.sum(vel * forward, -1), torch.sum(vel * left, -1), torch.sum(vel * up, -1)), -1)
    step = int(getattr(env, "common_step_counter", 0))
    if getattr(env, "_zbot_ang_filter_step", -1) != step:
        state = getattr(env, "_zbot_ang_vel_ema", None)
        if state is None or state.shape != current.shape:
            state = current.clone()
        else:
            state.mul_(0.9).add_(current, alpha=0.1)
        env._zbot_ang_vel_ema = state
        env._zbot_ang_filter_step = step
    return env._zbot_ang_vel_ema


def projected_gravity(env):
    robot, body_id = _base(env)
    q = robot.data.body_link_quat_w[:, body_id]
    axis = torch.zeros((env.num_envs, 3), device=env.device)
    axis[:, 2] = -1.0
    return quat_apply_inverse(q, axis)


def track_linear_velocity(env, command_name="twist", std=0.5):
    command = env.command_manager.get_command(command_name)
    assert command is not None
    error = (command[:, :2] - base_lin_vel(env)[:, :2]).square().sum(-1)
    error = error + base_lin_vel(env)[:, 2].square()
    target_score = torch.exp(-error / std**2)
    stationary_score = torch.exp(-command[:, :2].square().sum(-1) / std**2)
    reward = (target_score - stationary_score) / (1.0 - stationary_score).clamp_min(1e-4)
    active = command[:, :2].norm(dim=-1) > 0.1
    return reward.clamp_min(0.0) * active.float()


def track_angular_velocity(env, command_name="twist", std=0.7):
    command = env.command_manager.get_command(command_name)
    assert command is not None
    actual = base_ang_vel(env)
    # The command controls yaw around gravity. Roll/pitch rates are handled by
    # the upright term and must not zero out yaw tracking during a gait.
    error = (command[:, 2] - actual[:, 2]).square()
    target_score = torch.exp(-error / std**2)
    stationary_score = torch.exp(-command[:, 2].square() / std**2)
    reward = (target_score - stationary_score) / (1.0 - stationary_score).clamp_min(1e-4)
    active = command[:, :2].norm(dim=-1) > 0.1
    active = active | (command[:, 2].abs() > 0.05)
    return reward.clamp_min(0.0) * active.float()


def upright(env, std=0.3):
    gravity = projected_gravity(env)
    return torch.exp(-gravity[:, :2].square().sum(-1) / std**2)


def action_rate_l2(env):
    return (env.action_manager.action - env.action_manager.prev_action).square().sum(-1)


def joint_pos_limits(env):
    robot = env.scene["robot"]
    return robot.data.joint_pos_limits[:, :, 0].sub(robot.data.joint_pos).clamp_min(0).square().sum(-1) + robot.data.joint_pos.sub(robot.data.joint_pos_limits[:, :, 1]).clamp_min(0).square().sum(-1)


def pose(env, std=0.35):
    robot = env.scene["robot"]
    # Center the posture regularizer at zero so the default pose does not grant
    # an immediate positive reward that can dominate locomotion terms.
    return torch.exp(-robot.data.joint_pos.sub(robot.data.default_joint_pos).square().sum(-1) / std**2) - 1.0


def feet_air_time(env, sensor_name="feet_ground_contact"):
    data = env.scene[sensor_name].data
    assert data.current_air_time is not None
    return data.current_air_time.clamp(0, 0.5).sum(-1)


def feet_contact(env, sensor_name="feet_ground_contact"):
    data = env.scene[sensor_name].data
    assert data.found is not None
    return (data.found > 0).float()


def feet_slip(env, sensor_name="feet_ground_contact"):
    contact = feet_contact(env, sensor_name)
    robot = env.scene["robot"]
    feet = robot.find_bodies(("foot_0", "foot_1"), preserve_order=True)[0]
    speed = robot.data.body_link_lin_vel_w[:, feet, :2].norm(dim=-1)
    return (speed * contact).sum(-1)


def feet_swing_velocity(env, sensor_name="feet_ground_contact", target=0.25, std=0.2):
    contact = feet_contact(env, sensor_name)
    robot = env.scene["robot"]
    feet = robot.find_bodies(("foot_0", "foot_1"), preserve_order=True)[0]
    speed = robot.data.body_link_lin_vel_w[:, feet, :2].norm(dim=-1)
    swing = (1.0 - contact) * torch.exp(-((speed - target) / std).square())
    return swing.sum(-1)


def feet_clearance(env, sensor_name="feet_ground_contact", target=0.06, std=0.04):
    contact = feet_contact(env, sensor_name)
    robot = env.scene["robot"]
    feet = robot.find_bodies(("foot_0", "foot_1"), preserve_order=True)[0]
    height = robot.data.body_link_pos_w[:, feet, 2]
    swing = (1.0 - contact) * torch.exp(-((height - target) / std).square())
    return swing.sum(-1)


def small_negative_baseline(env):
    return torch.full((env.num_envs,), -0.35, device=env.device)


def self_collision(env, sensor_name="self_collision", force_threshold=1.0):
    data = env.scene[sensor_name].data
    if data.force_history is not None:
        return (data.force_history.norm(dim=-1).amax(dim=(-1, -2)) > force_threshold)
    assert data.force is not None
    return (data.force.norm(dim=-1).amax(dim=-1) > force_threshold)


def self_collision_penalty(env, sensor_name="self_collision", force_threshold=1.0, penalty=5.0):
    return -self_collision(env, sensor_name, force_threshold).float() * penalty / env.step_dt


def alive_reward(env):
    return torch.ones(env.num_envs, device=env.device)


class ZbotVelocityCommand(UniformVelocityCommand):
    """G1 command sampler with debug arrows anchored to zbot's base body."""

    def _update_metrics(self):
        actual_lin = base_lin_vel(self._env)
        actual_ang = base_ang_vel(self._env)
        max_command_steps = self.cfg.resampling_time_range[1] / self._env.step_dt
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - actual_lin[:, :2], dim=-1)
            / max_command_steps
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - actual_ang[:, 2])
            / max_command_steps
        )

    def _debug_vis_impl(self, visualizer):
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        base_id = self.robot.find_bodies("base")[0][0]
        pos = self.robot.data.body_link_pos_w[:, base_id].cpu().numpy()
        quat = self.robot.data.body_link_quat_w[:, base_id]
        mats = matrix_from_quat(quat).cpu().numpy()
        vel = quat_apply_inverse(quat, self.robot.data.body_link_lin_vel_w[:, base_id])
        ang = quat_apply_inverse(quat, self.robot.data.body_link_ang_vel_w[:, base_id])
        forward, left, up = locomotion_frame(quat)
        q = torch.stack((torch.sum(vel * forward, -1), torch.sum(vel * left, -1), torch.sum(vel * up, -1)), -1).cpu().numpy()
        w = torch.stack((torch.sum(ang * forward, -1), torch.sum(ang * left, -1), torch.sum(ang * up, -1)), -1).cpu().numpy()
        axes = torch.stack((forward, left, up), dim=-1)
        world_axes = torch.matmul(matrix_from_quat(quat), axes).cpu().numpy()
        cmds = self.command.cpu().numpy()
        for i in env_indices:
            origin = pos[i] + mats[i] @ np.array([0.0, 0.0, self.cfg.viz.z_offset])
            def point(v):
                return origin + world_axes[i] @ (np.asarray(v) * self.cfg.viz.scale)
            visualizer.add_arrow(point((0, 0, 0)), point((cmds[i, 0], cmds[i, 1], 0)), color=(0.2, 0.2, 0.6, 0.8), width=0.015)
            visualizer.add_arrow(point((0, 0, 0)), point((0, 0, cmds[i, 2])), color=(0.2, 0.6, 0.2, 0.8), width=0.015)
            visualizer.add_arrow(point((0, 0, 0)), point((q[i, 0], q[i, 1], 0)), color=(0.0, 0.6, 1.0, 0.8), width=0.015)
            visualizer.add_arrow(point((0, 0, 0)), point((0, 0, w[i, 2])), color=(0.0, 1.0, 0.4, 0.8), width=0.015)


def policy_observation(env):
    robot, base_id = _base(env)
    data = robot.data
    action = env.action_manager.get_term("joint_position")
    ids = action.joint_ids
    return torch.cat(
        (
            data.body_link_quat_w[:, base_id],
            data.body_link_ang_vel_w[:, base_id],
            data.joint_pos[:, ids] - data.default_joint_pos[:, ids],
            data.joint_vel[:, ids],
            action.bounded_action,
            action.speed,
        ),
        dim=-1,
    )


def fallen(env, card):
    robot, base_id = _base(env)
    pos = robot.data.body_link_pos_w[:, base_id] - env.scene.env_origins
    forces = env.scene["self_collision"].data.force_history
    contact = forces.norm(dim=-1).amax(dim=(-1, -2)) > card.non_foot_contact_force
    return (
        contact
        | (pos[:, 2] < card.termination_height)
        | (pos[:, 1].abs() > card.maximum_lateral_deviation)
    )


def source_time_out(env):
    # Source DirectRLEnv checks length >= max_episode_length - 1.
    return env.episode_length_buf >= env.max_episode_length - 1


def forward_velocity(env, card):
    robot, base_id = _base(env)
    quat = robot.data.body_link_quat_w[:, base_id]
    _, forward_b, _ = walking_frame(quat)
    vel_b = quat_apply_inverse(quat, robot.data.body_com_lin_vel_w[:, base_id])
    return card.moving_direction * (vel_b * forward_b).sum(dim=-1)


def base_vel_forward(env, card):
    velocity = forward_velocity(env, card)
    env.extras.setdefault("log", {})["Metrics/forward_velocity"] = float(velocity.mean().item())
    return torch.exp(-((velocity - card.target_forward_speed) / card.velocity_sigma).square())


def feet_downward(env, card):
    del card
    robot = env.scene["robot"]
    feet = robot.find_bodies(("foot_0", "foot_1"), preserve_order=True)[0]
    quat = robot.data.body_link_quat_w[:, feet]
    axis_z = torch.zeros_like(quat[..., :3])
    axis_z[..., 2] = 1
    return quat_apply(quat, axis_z)[..., :2].norm(dim=-1).sum(dim=-1)


def feet_forward(env, card):
    del card
    robot, base_id = _base(env)
    feet = robot.find_bodies(("foot_0", "foot_1"), preserve_order=True)[0]
    quat = robot.data.body_link_quat_w[:, base_id]
    feet_quat = robot.data.body_link_quat_w[:, feet]
    _, forward_b, _ = walking_frame(quat)
    axis_x = torch.zeros_like(feet_quat[..., :3])
    axis_x[..., 0] = 1
    feet_x_b = quat_apply_inverse(quat[:, None].expand(-1, 2, -1), quat_apply(feet_quat, axis_x))
    return (feet_x_b - forward_b[:, None]).norm(dim=-1).sum(dim=-1)


def terminal_penalty(env, card):
    # RewardManager integrates all terms with dt. This keeps the source's discrete cost.
    return -env.reset_terminated.float() * card.terminated_reward_penalty / env.step_dt


class WalkingReward:
    """Two-stage reward with global, consecutive-step promotion.

    Stage is global and intentionally survives episode resets. Per-episode
    force/heading integrals and touchdown history reset only for selected envs.
    """

    def __init__(self, cfg, env):
        self.env = env
        self.card = cfg.params["card"]
        self.stage = self.card.initial_stage
        self.promotion_counter = 0
        self.last_metric = 0.0
        self.robot, self.base_id = _base(env)
        self.feet_ids = self.robot.find_bodies(("foot_0", "foot_1"), preserve_order=True)[0]
        self.force_integral = torch.zeros(env.num_envs, device=env.device)
        self.heading_integral = torch.zeros_like(self.force_integral)
        self.last_force = torch.zeros((env.num_envs, 2), device=env.device)
        self.step_length = torch.zeros_like(self.last_force)
        self.touchdown_pos = torch.zeros((env.num_envs, 2, 3), device=env.device)
        self.initialize_touchdown = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for tensor in (
            self.force_integral,
            self.heading_integral,
            self.last_force,
            self.step_length,
            self.touchdown_pos,
        ):
            tensor[ids] = 0
        self.initialize_touchdown[ids] = True

    def _promote(self, metric):
        if self.stage != 1:
            return
        self.last_metric = float(metric.mean().item())
        self.promotion_counter = (
            self.promotion_counter + 1 if self.last_metric > self.card.promotion_threshold else 0
        )
        if self.promotion_counter >= self.card.promotion_window_steps:
            self.stage = 2

    def __call__(self, env, card):
        data = self.robot.data
        action = env.action_manager.get_term("joint_position")
        q = data.body_link_quat_w[:, self.base_id]
        _, forward, heading = walking_frame(q)
        velocity = quat_apply_inverse(q, data.body_com_lin_vel_w[:, self.base_id])
        forward_velocity = card.moving_direction * (velocity * forward).sum(dim=-1)
        feet_q = data.body_link_quat_w[:, self.feet_ids]
        feet_pos = data.body_link_pos_w[:, self.feet_ids]
        axis_z = torch.zeros_like(feet_pos)
        axis_z[..., 2] = 1
        axis_x = torch.zeros_like(feet_pos)
        axis_x[..., 0] = 1
        feet_z = quat_apply(feet_q, axis_z)
        feet_x = quat_apply_inverse(q[:, None].expand(-1, 2, -1), quat_apply(feet_q, axis_x))
        sensor = env.scene["feet_contact"].data
        # netforce is world-frame force; use upward magnitude for load balance.
        force = sensor.force_history[..., 2].mean(dim=-1).abs()
        force_diff = self.force_integral.sign() * (force[:, 1] - force[:, 0])
        raw = {
            "feet_downward": feet_z[..., :2].norm(dim=-1).sum(dim=-1),
            "feet_forward": (feet_x - forward[:, None]).norm(dim=-1).sum(dim=-1),
            "base_heading_x": heading.abs(),
            "feet_force_diff": force_diff,
        }
        # Keep original ordering: force-difference uses the PREVIOUS integral.
        if self.stage == 1:
            self.force_integral += 0.001 * (force[:, 0] - force[:, 1])
            raw["feet_force_sum"] = self.force_integral.abs()
            scales = card.stage_1_rewards
        else:
            self.heading_integral.add_(0.01 * heading).clamp_(-1, 1)
            init = self.initialize_touchdown
            self.touchdown_pos[init] = feet_pos[init]
            self.initialize_touchdown[:] = False
            touchdown = (force > 10) & (self.last_force < 10)
            step_vec = quat_apply_inverse(
                q[:, None].expand(-1, 2, -1), feet_pos - self.touchdown_pos
            )
            lengths = (step_vec * forward[:, None]).sum(dim=-1)
            self.step_length[touchdown] = lengths[touchdown]
            self.touchdown_pos[touchdown] = feet_pos[touchdown]
            self.last_force.copy_(force)
            air = sensor.last_air_time
            pos = data.body_link_pos_w[:, self.base_id]
            origins = env.scene.env_origins
            raw.update(
                {
                    "base_vel_forward": torch.tanh(10 * forward_velocity / action.speed[:, 0]),
                    "similar_to_default": (data.joint_pos - data.default_joint_pos)
                    .abs()
                    .sum(dim=-1),
                    "base_heading_x_sum": self.heading_integral.abs(),
                    "step_length": torch.tanh(
                        15 * card.moving_direction * self.step_length.amin(dim=-1)
                    ),
                    "airtime_balance": (air[:, 0] - air[:, 1]).abs(),
                    "airtime_sum": torch.tanh(air.sum(dim=-1)),
                    "action_rate": (action.bounded_action - action.previous_action)
                    .square()
                    .sum(dim=-1),
                    "torques": 0.002 * data.qfrc_actuator.square().sum(dim=-1),
                    "feet_slide": (
                        data.body_com_lin_vel_w[:, self.feet_ids, :2].norm(dim=-1) * (force > 1)
                    ).sum(dim=-1),
                    "base_pos_y_err": 10
                    * (
                        (feet_pos[:, 0, 1] + feet_pos[:, 1, 1] - 2 * origins[:, 1]).abs()
                        + (pos[:, 1] - origins[:, 1]).abs()
                    ),
                }
            )
            scales = card.stage_2_rewards
        result = sum(raw[name] * weight for name, weight in scales.items())
        self._promote(force_diff)
        logs = env.extras.setdefault("log", {})
        logs["Metrics/forward_velocity"] = float(forward_velocity.mean().item())
        for name in dict.fromkeys((*card.stage_1_rewards, *card.stage_2_rewards)):
            logs[f"Reward/{name}"] = (
                float((raw[name] * scales[name]).mean().item()) if name in scales else 0.0
            )
        logs["Curriculum/stage"] = float(self.stage)
        logs["Curriculum/feet_force_diff_mean"] = self.last_metric
        # RewardManager multiplies this rate by dt; the -20 terminal cost is discrete.
        return result - env.reset_terminated.float() * card.terminated_reward_penalty / env.step_dt
