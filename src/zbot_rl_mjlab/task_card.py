"""Parameters aligned to Isaac Lab Zbot-Direct-6dof-bipedal-quat-v0."""

from dataclasses import dataclass, field
from typing import Literal

TASK_ID = "Mjlab-Zbot-6dof-Bipedal-Walking"
JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))
DEFAULT_JOINT_POS = (0.312, 0.837, -2.02, 2.02, -0.837, -0.312)
OBSERVATION_TERMS = (
    ("base_linear_velocity", 3),
    ("base_angular_velocity", 3),
    ("projected_gravity", 3),
    ("joint_position_error", 6),
    ("joint_velocity", 6),
    ("action", 6),
    ("velocity_command", 3),
)
OBSERVATION_DIM = sum(width for _, width in OBSERVATION_TERMS)


@dataclass(frozen=True)
class WalkingTaskCard:
    num_envs: int = 4096
    physics_dt: float = 1 / 60
    decimation: int = 2
    contact_history_length: int = 5
    constraint_capacity: int = 256
    friction_cone: Literal["pyramidal", "elliptic"] = "pyramidal"
    friction_impedance_ratio: float = 1.0
    contact_dimension: int = 3
    foot_sliding_friction: float = 1.0
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    episode_length_s: float = 20.0
    action_scale: float = 1.0
    joint_speed_range: tuple[float, float] = (0.2, 2.0)
    stiffness: float = 15.0
    damping: float = 1.5
    effort_limit: float = 200.0
    termination_height: float = 0.22
    maximum_lateral_deviation: float = 0.5
    non_foot_contact_force: float = 1.0
    terminated_reward_penalty: float = 20.0
    moving_direction: float = 1.0
    # Quat observations are fixed; the default reward now follows ordinary 6DOF walking.
    reward_mode: str = "bipedal_curriculum"
    initial_stage: int = 1
    promotion_threshold: float = 0.5
    promotion_window_steps: int = 50
    stage_1_rewards: dict[str, float] = field(
        default_factory=lambda: {
            "feet_downward": -0.5,
            "feet_forward": -0.25,
            "base_heading_x": -0.25,
            "feet_force_diff": 1.0,
            "feet_force_sum": -0.05,
        }
    )
    stage_2_rewards: dict[str, float] = field(
        default_factory=lambda: {
            "base_vel_forward": 2.0,
            "feet_downward": -0.5,
            "feet_forward": -0.25,
            "similar_to_default": -0.25,
            "base_heading_x": -0.25,
            "base_heading_x_sum": -0.25,
            "step_length": 0.5,
            "airtime_balance": -2.0,
            "airtime_sum": 0.5,
            "action_rate": -0.05,
            "torques": -0.1,
            "feet_slide": -1.0,
            "base_pos_y_err": -0.05,
        }
    )
    target_forward_speed: float = 0.5
    velocity_sigma: float = 0.3
    reward_scales: dict[str, float] = field(
        default_factory=lambda: {
            "base_vel_forward": 1.0,
            "feet_downward": -1.0,
            "feet_forward": -0.5,
        }
    )

    def __post_init__(self):
        if self.foot_sliding_friction <= 0:
            raise ValueError("foot_sliding_friction must be positive")
        if self.contact_dimension not in (3, 4, 6):
            raise ValueError("contact_dimension must be 3, 4 or 6")
        if (
            self.friction_cone not in ("pyramidal", "elliptic")
            or self.friction_impedance_ratio <= 0
        ):
            raise ValueError("Invalid friction solver settings")
        if self.reward_mode not in ("quat", "bipedal_curriculum"):
            raise ValueError("Unknown reward_mode")
        if self.initial_stage not in (1, 2) or self.promotion_window_steps < 1:
            raise ValueError("Invalid curriculum stage/window")
        if self.num_envs < 1 or self.decimation < 1 or self.physics_dt <= 0:
            raise ValueError("num_envs, decimation and physics_dt must be positive")
        if not (0 < self.joint_speed_range[0] <= self.joint_speed_range[1]):
            raise ValueError("joint_speed_range must be positive and ordered")
        if self.velocity_sigma <= 0 or self.contact_history_length < 1:
            raise ValueError("velocity_sigma and contact_history_length must be positive")
        if set(self.reward_scales) != {"base_vel_forward", "feet_downward", "feet_forward"}:
            raise ValueError("Quat task requires exactly the three source rewards")
