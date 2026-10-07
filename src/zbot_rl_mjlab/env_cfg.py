import math
import os
from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
    BuiltinSensorCfg,
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from . import mdp
from .robot import robot_cfg

TASK_ID = "Mjlab-Zbot-6dof-Periodic-Stepping"
WALKING_TASK_ID = "Mjlab-Zbot-6dof-Walking"
STEP_LENGTH_WALKING_TASK_ID = "Mjlab-Zbot-6dof-Step-Length-Walking"
DISTURBED_WALKING_TASK_ID = "Mjlab-Zbot-6dof-Disturbed-Walking"
MIN_STEP_FREQUENCY = 0.2
MAX_STEP_FREQUENCY = 1.0


@dataclass
class ZbotEnvCfg(ManagerBasedRlEnvCfg):
    step_frequency_min: float = MIN_STEP_FREQUENCY
    step_frequency_max: float = MAX_STEP_FREQUENCY
    test_frequency: float | None = None
    frequency_sampling: str = "random"


def env_cfg(play=False):
    fixed = os.environ.get("ZBOT_TEST_FREQUENCY") if play else None
    fixed = float(fixed) if fixed is not None else None
    if fixed is not None and (not math.isfinite(fixed) or fixed <= 0):
        raise ValueError("Test frequency must be finite and positive")
    feet = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="body", pattern=("foot_0", "foot_1"), entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        track_air_time=True,
    )
    feet_self = ContactSensorCfg(
        name="feet_self_contact",
        primary=ContactMatch(mode="body", pattern=("foot_0", "foot_1"), entity="robot"),
        secondary=ContactMatch(mode="body", pattern="foot_0", entity="robot"),
        fields=("force",),
        reduce="netforce",
    )
    height = TerrainHeightSensorCfg(
        name="foot_height_scan",
        frame=(
            ObjRef(type="body", name="foot_0", entity="robot"),
            ObjRef(type="body", name="foot_1", entity="robot"),
        ),
        ray_alignment="yaw",
        pattern=RingPatternCfg.single_ring(radius=0.03, num_samples=6),
        max_distance=1.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        debug_vis=True,
    )
    actor = {
        "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
        "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel),
        "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
        "yaw_error": ObservationTermCfg(func=mdp.yaw_error_observation),
        "joint_pos": ObservationTermCfg(func=envs_mdp.joint_pos_rel),
        "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel),
        "actions": ObservationTermCfg(func=envs_mdp.last_action),
        "phase": ObservationTermCfg(func=mdp.phase_observation),
        "frequency": ObservationTermCfg(func=mdp.frequency_observation),
    }
    rewards = {
        "frequency_tracking": RewardTermCfg(
            func=mdp.alternating_foot_phase,
            weight=2.0,
            params={"sensor_name": feet.name},
        ),
        "swing_clearance": RewardTermCfg(
            func=mdp.swing_clearance,
            weight=0.5,
            params={"sensor_name": feet.name, "height_sensor_name": height.name},
        ),
        "forward_velocity": RewardTermCfg(
            func=mdp.forward_velocity_reward, weight=1.0, params={"scale": 0.2}
        ),
        "lateral_velocity": RewardTermCfg(
            func=mdp.lateral_velocity_penalty, weight=-0.5
        ),
        "upright": RewardTermCfg(func=mdp.upright, weight=0.2),
        "yaw_drift": RewardTermCfg(func=mdp.yaw_drift_penalty, weight=-0.5),
        "yaw_ang_vel": RewardTermCfg(func=mdp.yaw_ang_vel_penalty, weight=-0.05),
        "support_foot_slip": RewardTermCfg(
            func=mdp.support_foot_slip, weight=-20.0, params={"sensor_name": feet.name}
        ),
        "soft_landing": RewardTermCfg(
            func=mdp.soft_landing,
            weight=-5.0,
            params={"sensor_name": feet.name, "impact_threshold": 5.0, "scale": 5.0},
        ),
        "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
        "joint_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-0.1),
    }

    class VizCfg(UniformVelocityCommandCfg):
        def build(self, env):
            return mdp.ForwardVelocityVisualizer(self, env)

    commands = (
        {
            "forward_velocity_viz": VizCfg(
                entity_name="robot",
                resampling_time_range=(1.0, 1.0),
                ranges=UniformVelocityCommandCfg.Ranges(
                    lin_vel_x=(0.0, 0.0),
                    lin_vel_y=(0.0, 0.0),
                    ang_vel_z=(0.0, 0.0),
                    heading=None,
                ),
                debug_vis=True,
            )
        }
        if play
        else {}
    )
    return ZbotEnvCfg(
        seed=42,
        test_frequency=fixed,
        step_frequency_min=MIN_STEP_FREQUENCY,
        step_frequency_max=MAX_STEP_FREQUENCY,
        decimation=4,
        episode_length_s=20.0,
        scene=SceneCfg(
            num_envs=1 if play else 4096,
            env_spacing=4.0,
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": robot_cfg()},
            sensors=(feet, feet_self, height),
        ),
        observations={
            "actor": ObservationGroupCfg(
                actor, concatenate_terms=True, enable_corruption=not play
            ),
            "critic": ObservationGroupCfg(
                actor, concatenate_terms=True, enable_corruption=False
            ),
        },
        actions={
            "joint_pos": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                scale=0.25,
                use_default_offset=True,
            )
        },
        commands=commands,
        rewards=rewards,
        terminations={
            "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
            "fallen": TerminationTermCfg(func=mdp.fallen),
        },
        events={
            "reset_scene_to_default": EventTermCfg(
                func=mdp.reset_scene_to_default, mode="reset"
            )
        },
        sim=SimulationCfg(
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
                ccd_iterations=50,
                gravity=(0, 0, -9.81),
            ),
            njmax=300,
            nconmax=None,
            contact_sensor_maxmatch=64,
        ),
        viewer=ViewerConfig(
            entity_name="robot",
            body_name="base",
            distance=1.2,
            elevation=-20.0,
            azimuth=135.0,
            width=640,
            height=480,
            origin_type=ViewerConfig.OriginType.WORLD,
            lookat=(0.0, 0.0, 0.2),
        ),
    )


def ppo_cfg():
    actor = RslRlModelCfg(
        hidden_dims=(256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    )
    critic = RslRlModelCfg(
        hidden_dims=(256, 128), activation="elu", obs_normalization=True
    )
    return RslRlOnPolicyRunnerCfg(
        actor=actor,
        critic=critic,
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="zbot_periodic_stepping",
        num_steps_per_env=24,
        max_iterations=30000,
        save_interval=50,
        logger="tensorboard",
    )


def _add_swing_foot_penalties(cfg):
    foot_accelerometers = tuple(
        BuiltinSensorCfg(
            name=f"foot_{index}_lin_acc",
            sensor_type="framelinacc",
            obj=ObjRef(type="body", name=f"foot_{index}", entity="robot"),
        )
        for index in range(2)
    )
    cfg.scene.sensors = (*cfg.scene.sensors, *foot_accelerometers)
    cfg.rewards["swing_foot_acceleration"] = RewardTermCfg(
        func=mdp.swing_foot_acceleration_penalty,
        weight=-0.1,
        params={
            "acceleration_sensors": tuple(
                sensor.prefixed_name for sensor in foot_accelerometers
            ),
            "acceleration_scale": 50.0,
        },
    )
    cfg.rewards["swing_foot_excess_height"] = RewardTermCfg(
        func=mdp.swing_foot_excess_height_penalty,
        weight=-20.0,
        params={"max_height": 0.05, "height_scale": 0.05},
    )
    cfg.rewards["joint_energy"] = RewardTermCfg(
        func=mdp.joint_energy_penalty,
        weight=-0.2,
        params={"energy_scale": 10.0},
    )
    cfg.rewards["feet_self_collision"] = RewardTermCfg(
        func=mdp.foot_self_collision_penalty,
        weight=-0.5,
        params={"sensor_name": "feet_self_contact", "scale": 10.0},
    )
    cfg.rewards["foot_nonparallel_ground"] = RewardTermCfg(
        func=mdp.foot_nonparallel_ground_penalty,
        weight=-0.2,
        params={"sensor_name": "feet_ground_contact", "std": 0.25},
    )


def walking_base_cfg(play=False, swing_penalties=True):
    """Shared velocity Walking configuration for derived experiments."""
    cfg = env_cfg(play=play)
    cfg.rewards["forward_velocity"] = RewardTermCfg(
        func=mdp.forward_velocity_reward, weight=2.0, params={"scale": 0.1}
    )
    if swing_penalties:
        _add_swing_foot_penalties(cfg)
        cfg.rewards["joint_acceleration"] = RewardTermCfg(
            func=mdp.joint_acceleration_penalty,
            weight=-0.001,
        )
    return cfg


def walking_cfg(play=False):
    return walking_base_cfg(play=play)


def stepping_cfg(play=False):
    """Periodic stepping task sharing Walking's observation/action contract."""
    cfg = walking_base_cfg(play=play, swing_penalties=False)
    cfg.rewards.pop("forward_velocity", None)
    cfg.rewards.pop("lateral_velocity", None)
    return cfg


def stepping_ppo_cfg():
    cfg = walking_ppo_cfg()
    cfg.experiment_name = "zbot_periodic_stepping"
    return cfg


def walking_ppo_cfg():
    cfg = ppo_cfg()
    cfg.experiment_name = "zbot_walking"
    return cfg


def step_length_walking_cfg(play=False):
    cfg = walking_base_cfg(play=play)
    cfg.rewards["frequency_tracking"].weight = 1.0
    cfg.rewards.pop("forward_velocity")
    cfg.rewards["forward_step_length"] = RewardTermCfg(
        func=mdp.ForwardStepLengthReward,
        weight=1.0,
        params={
            "sensor_name": "feet_ground_contact",
            "max_step_length": 0.5,
            "step_length_scale": 0.2,
            "balance_weight": 0.1,
        },
    )
    return cfg


def step_length_walking_ppo_cfg():
    cfg = walking_ppo_cfg()
    cfg.experiment_name = "zbot_step_length_walking"
    return cfg


def disturbed_walking_cfg(play=False):
    cfg = walking_base_cfg(play=play)
    cfg.rewards["forward_velocity"].weight = 2.0
    cfg.events["base_disturbance"] = EventTermCfg(
        func=mdp.PartialBaseImpulse,
        mode="step",
        params={
            "force_range": (-5.0, 5.0),
            "torque_range": (-0.5, 0.5),
            "duration_s": (0.08, 0.20),
            "cooldown_s": (0.8, 2.0),
            "asset_cfg": SceneEntityCfg("robot", body_names=("base",)),
            "body_point_offset": (0.0, 0.0, 0.0),
            "disturbed_fraction": 0.25,
        },
    )
    cfg.events["randomize_foot_friction"] = EventTermCfg(
        mode="startup",
        func=dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", geom_names=(r"foot_0_.*", r"foot_1_.*")
            ),
            "operation": "abs",
            "ranges": (0.3, 1.2),
            "shared_random": True,
        },
    )
    cfg.events["randomize_encoder_bias"] = EventTermCfg(
        mode="startup",
        func=dr.encoder_bias,
        params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.015, 0.015)},
    )
    cfg.events["randomize_base_com"] = EventTermCfg(
        mode="startup",
        func=dr.body_com_offset,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("base",)),
            "operation": "add",
            "ranges": {0: (-0.025, 0.025), 1: (-0.025, 0.025), 2: (-0.03, 0.03)},
        },
    )
    return cfg


def disturbed_walking_ppo_cfg():
    cfg = walking_ppo_cfg()
    cfg.experiment_name = "zbot_disturbed_walking"
    return cfg
