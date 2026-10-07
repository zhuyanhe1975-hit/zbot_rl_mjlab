"""Build native comparison scenes from the active MJLab robot asset."""

from copy import deepcopy

from mjlab.scene import Scene

from zbot_rl_mjlab.robot import robot_cfg


def build_comparison_scene(env_cfg):
    cfg = deepcopy(env_cfg.scene)
    cfg.entities = {"ref": robot_cfg(), "native": robot_cfg()}
    cfg.sensors = ()
    scene = Scene(cfg, device="cpu")
    model = scene.compile()
    env_cfg.sim.mujoco.apply(model)
    return scene, model
