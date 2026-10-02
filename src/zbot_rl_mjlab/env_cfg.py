"""Zbot velocity task using the MJLab G1-style reward and PPO structure."""
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from . import mdp
from .robot import robot_cfg
from .task_card import WalkingTaskCard


def walking_env_cfg(play=False, card=None):
    card = card or WalkingTaskCard()
    feet = ContactSensorCfg(name="feet_ground_contact", primary=ContactMatch(mode="body", pattern=("foot_0", "foot_1"), entity="robot"), secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force"), reduce="netforce", track_air_time=True, history_length=card.contact_history_length)
    self_collision = ContactSensorCfg(name="self_collision", primary=ContactMatch(mode="subtree", pattern="foot_0", entity="robot"), secondary=ContactMatch(mode="subtree", pattern="foot_0", entity="robot"), fields=("found", "force"), reduce="none", history_length=4)
    actor = {
        "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
        "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel),
        "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
        "joint_pos": ObservationTermCfg(func=envs_mdp.joint_pos_rel),
        "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel),
        "actions": ObservationTermCfg(func=envs_mdp.last_action),
        "command": ObservationTermCfg(func=envs_mdp.generated_commands, params={"command_name": "twist"}),
    }
    observations = {"actor": ObservationGroupCfg(actor, concatenate_terms=True, enable_corruption=True), "critic": ObservationGroupCfg(actor, concatenate_terms=True, enable_corruption=False)}
    rewards = {
        "track_linear_velocity": RewardTermCfg(func=mdp.track_linear_velocity, weight=2.0, params={"command_name": "twist", "std": 0.3}),
        "track_angular_velocity": RewardTermCfg(func=mdp.track_angular_velocity, weight=0.5, params={"command_name": "twist", "std": 0.5}),
        "upright": RewardTermCfg(func=mdp.upright, weight=1.0, params={"std": 0.3}),
        "pose": RewardTermCfg(func=mdp.pose, weight=0.1, params={"std": 0.35}),
        "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-0.1),
        "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.02),
        "air_time": RewardTermCfg(func=mdp.feet_air_time, weight=0.1, params={"sensor_name": feet.name}),
        "foot_slip": RewardTermCfg(func=mdp.feet_slip, weight=-0.03, params={"sensor_name": feet.name}),
        "foot_swing_velocity": RewardTermCfg(func=mdp.feet_swing_velocity, weight=0.1, params={"sensor_name": feet.name, "target": 0.25, "std": 0.2}),
        "foot_clearance": RewardTermCfg(func=mdp.feet_clearance, weight=0.1, params={"sensor_name": feet.name, "target": 0.06, "std": 0.04}),
        "self_collision_penalty": RewardTermCfg(func=mdp.self_collision_penalty, weight=1.0, params={"sensor_name": self_collision.name, "force_threshold": 1.0, "penalty": 10.0}),
        "alive": RewardTermCfg(func=mdp.alive_reward, weight=0.5),
    }
    class ZbotCommandCfg(UniformVelocityCommandCfg):
        def build(self, env):
            return mdp.ZbotVelocityCommand(self, env)
    commands = {"twist": ZbotCommandCfg(entity_name="robot", resampling_time_range=(5.0, 10.0), rel_standing_envs=0.0, rel_heading_envs=0.0, rel_forward_envs=0.2, heading_command=False, debug_vis=True, ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(0.05, 0.4), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.2, 0.2), heading=None))}
    commands["twist"].viz.scale = 0.25
    commands["twist"].viz.z_offset = 0.25
    return ManagerBasedRlEnvCfg(seed=42, decimation=card.decimation, episode_length_s=card.episode_length_s, scene=SceneCfg(num_envs=1 if play else card.num_envs, env_spacing=4.0, terrain=TerrainEntityCfg(terrain_type="plane"), entities={"robot": robot_cfg(card)}, sensors=(feet, self_collision)), observations=observations, actions={"joint_pos": JointPositionActionCfg(entity_name="robot", actuator_names=(".*",), scale=0.25, use_default_offset=True)}, commands=commands, rewards=rewards, terminations={"time_out": TerminationTermCfg(func=mdp.source_time_out, time_out=True), "fallen": TerminationTermCfg(func=mdp.fallen, params={"card": card}), "self_collision": TerminationTermCfg(func=mdp.self_collision, params={"sensor_name": self_collision.name, "force_threshold": 1.0})}, events={"reset_scene_to_default": EventTermCfg(func=envs_mdp.reset_scene_to_default, mode="reset")}, sim=SimulationCfg(mujoco=MujocoCfg(timestep=card.physics_dt, gravity=card.gravity, cone=card.friction_cone, impratio=card.friction_impedance_ratio), njmax=card.constraint_capacity), viewer=ViewerConfig(entity_name="robot", body_name="base", distance=1.2, elevation=-20.0, azimuth=135.0, width=640, height=480, origin_type=ViewerConfig.OriginType.ASSET_BODY))


def walking_ppo_cfg():
    return RslRlOnPolicyRunnerCfg(actor=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True, distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.5, "std_type": "scalar"}), critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True), algorithm=RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.02, num_learning_epochs=5, num_mini_batches=4, learning_rate=1e-3, schedule="adaptive", gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0), experiment_name="zbot_g1_velocity", num_steps_per_env=24, max_iterations=30_000, save_interval=50, logger="tensorboard")
