"""MuJoCo entity definition; the MJCF retains the original USD link frames."""

from pathlib import Path
from functools import partial
import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from .task_card import DEFAULT_JOINT_POS, JOINT_NAMES, WalkingTaskCard

ASSET_PATH = Path(__file__).resolve().parents[2] / "assets/zbot_6dof/robot_geometry_inertia.xml"


# MuJoCo 3.8.1 exports MjSpec at runtime, but its distributed stubs omit it.
def get_spec(contact_dimension=3, foot_sliding_friction=1.0) -> mujoco.MjSpec:  # pyright: ignore[reportAttributeAccessIssue]
    spec = mujoco.MjSpec.from_file(str(ASSET_PATH))  # pyright: ignore[reportAttributeAccessIssue]
    for geom in spec.geoms:
        if geom.contype or geom.conaffinity:
            geom.condim = contact_dimension
            if geom.name.startswith("foot_"):
                geom.friction = (foot_sliding_friction, geom.friction[1], geom.friction[2])
    return spec


def robot_cfg(card: WalkingTaskCard) -> EntityCfg:
    return EntityCfg(
        spec_fn=partial(
            get_spec,
            contact_dimension=card.contact_dimension,
            foot_sliding_friction=card.foot_sliding_friction,
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                BuiltinPositionActuatorCfg(
                    target_names_expr=JOINT_NAMES,
                    stiffness=card.stiffness,
                    damping=card.damping,
                    effort_limit=card.effort_limit,
                ),
            )
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, -0.06, 0.0),
            joint_pos=dict(zip(JOINT_NAMES, DEFAULT_JOINT_POS)),
            joint_vel={".*": 0.0},
        ),
    )
