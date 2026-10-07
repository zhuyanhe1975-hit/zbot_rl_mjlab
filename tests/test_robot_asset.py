import numpy as np
from mjlab.scene import Scene

from mjlab_mujoco311_bench.scene_models import build_comparison_scene
from zbot_rl_mjlab.env_cfg import walking_cfg
from zbot_rl_mjlab.robot import ASSET_PATH, PD_ARMATURE, get_spec


def test_asset_adapter_and_single_actuator_set():
    assert ASSET_PATH.name == "ZBOT6.xml"
    spec = get_spec()
    assert not list(spec.actuators)
    assert {"foot_0", "base", "foot_1"} <= {body.name for body in spec.bodies}
    cfg = walking_cfg(play=True)
    model = Scene(cfg.scene, device="cpu").compile()
    cfg.sim.mujoco.apply(model)
    assert model.nq == 13 and model.nu == 6
    for index in range(1, 7):
        actuator = model.actuator(f"robot/joint{index}").id
        assert model.actuator_gainprm[actuator, 0] == 15
        assert np.allclose(model.actuator_biasprm[actuator, 1:3], [-15, -1.5])
        assert np.allclose(model.actuator_forcerange[actuator], [-20, 20])
        assert np.allclose(model.dof_armature[6 + index - 1], PD_ARMATURE)


def test_comparison_uses_same_new_asset_and_drive():
    cfg = walking_cfg(play=True)
    model = Scene(cfg.scene, device="cpu").compile()
    _, comparison = build_comparison_scene(cfg)
    assert comparison.nu == 12
    for prefix in ("ref", "native"):
        for body_name in ("foot_0", "base", "foot_1", "b3"):
            source = model.body(f"robot/{body_name}").id
            target = comparison.body(f"{prefix}/{body_name}").id
            assert np.allclose(model.body_mass[source], comparison.body_mass[target])
            assert np.allclose(
                model.body_inertia[source], comparison.body_inertia[target]
            )
