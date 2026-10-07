from functools import partial
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

ASSET_PATH = Path(__file__).resolve().parents[2] / "assets/zbot_6/ZBOT6.xml"
JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))
DEFAULT_JOINT_POS = (0.312, 0.837, -2.02, 2.02, -0.837, -0.312)
PD_STIFFNESS = 15.0
PD_DAMPING = 1.5
PD_EFFORT_LIMIT = 20.0
PD_NATURAL_FREQUENCY_HZ = 10.0
JOINT_DAMPING = 0.05
PD_ARMATURE = 0.0036


def get_spec(contact_dimension=3):
    spec = mujoco.MjSpec.from_file(str(ASSET_PATH))
    # Preserve the task's semantic names without modifying the supplied asset.
    aliases = {"a1": "foot_0", "a4": "base", "b6": "foot_1"}
    for body in spec.bodies:
        if body.name in aliases:
            body.name = aliases[body.name]
            for geom in body.geoms:
                geom.name = f"{body.name}_{geom.name}"
    # MJLab creates the configured six PD actuators; do not retain a second set.
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    for joint_name in JOINT_NAMES:
        spec.joint(joint_name).damping[0] = JOINT_DAMPING
        spec.joint(joint_name).armature = PD_ARMATURE
    for geom in spec.geoms:
        if geom.contype or geom.conaffinity:
            # Match G1's collision profile: only feet use 3D frictional
            # contacts; internal/self-collision geoms use condim=1.
            is_foot = geom.name.startswith(("foot_0_", "foot_1_"))
            geom.condim = contact_dimension if is_foot else 1
            if is_foot:
                geom.friction = (0.6, geom.friction[1], geom.friction[2])
    return spec


def robot_cfg():
    return EntityCfg(
        spec_fn=partial(get_spec),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                BuiltinPositionActuatorCfg(
                    target_names_expr=JOINT_NAMES,
                    stiffness=PD_STIFFNESS,
                    damping=PD_DAMPING,
                    effort_limit=PD_EFFORT_LIMIT,
                    armature=PD_ARMATURE,
                ),
            )
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, -0.06, 0.0),
            joint_pos=dict(zip(JOINT_NAMES, DEFAULT_JOINT_POS)),
            joint_vel={".*": 0.0},
        ),
    )
