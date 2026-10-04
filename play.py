"""Direct MJLab-style CLI for Zbot checkpoint playback."""

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import mjlab
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

import zbot_rl_mjlab  # noqa: F401
from zbot_rl_mjlab.env_cfg import ZbotEnvCfg


@dataclass
class PlayConfig:
    env: ZbotEnvCfg
    checkpoint_file: str | None = None
    log_root: str = "logs/rsl_rl"
    device: str = "cuda:0"
    viewer: str = "native"


def main():
    task, remaining = tyro.cli(
        tyro.extras.literal_type_from_choices(
            [
                "Mjlab-Zbot-6dof-Periodic-Stepping",
                "Mjlab-Zbot-6dof-Walking",
                "Mjlab-Zbot-6dof-Disturbed-Walking",
                "Mjlab-Zbot-6dof-Step-Length-Walking",
            ]
        ),
        add_help=False, return_unknown_args=True, config=mjlab.TYRO_FLAGS,
    )
    cfg = tyro.cli(PlayConfig, args=remaining,
                   default=PlayConfig(env=load_env_cfg(task, play=True)),
                   config=mjlab.TYRO_FLAGS)
    cfg.env.scene.env_spacing = 1.0
    cfg.env.viewer.enable_shadows = True
    cfg.env.viewer.enable_reflections = True
    disturbance = cfg.env.events.get("base_disturbance")
    if disturbance is not None:
        params = disturbance.params
        force_range = tuple(params.get("force_range", ()))
        torque_range = tuple(params.get("torque_range", ()))
        if force_range == (0.0, 0.0) and torque_range == (0.0, 0.0):
            cfg.env.events.pop("base_disturbance")
            print("Disturbance disabled: base_disturbance event removed")
    if not 0 < cfg.env.step_frequency_min <= cfg.env.step_frequency_max:
        raise ValueError("Require 0 < step-frequency-min <= step-frequency-max")
    agent = load_rl_cfg(task)
    checkpoint = Path(cfg.checkpoint_file) if cfg.checkpoint_file else None
    if checkpoint is None:
        root = Path(cfg.log_root) / agent.experiment_name
        candidates = [p for p in root.rglob("model_*.pt") if p.stem[6:].isdigit()]
        if not candidates:
            raise FileNotFoundError(f"No checkpoints under {root}")
        checkpoint = max(candidates, key=lambda p: p.stat().st_mtime_ns)
    print(f"Checkpoint: {checkpoint}")
    print(f"Frequency: {cfg.env.step_frequency_min}–{cfg.env.step_frequency_max} Hz")
    # Keep frequency encoding consistent with training even for fixed tests.
    if cfg.env.step_frequency_min == cfg.env.step_frequency_max:
        cfg.env.test_frequency = cfg.env.step_frequency_min
        cfg.env.step_frequency_min = 0.2
        cfg.env.step_frequency_max = 1.0
    if cfg.viewer not in ("native", "viser"):
        raise ValueError("viewer must be native or viser")
    env = ManagerBasedRlEnv(cfg.env, device=cfg.device)
    try:
        # MuJoCo's default shadow clip is only 1 m.  Extend it so shadows do
        # not disappear when the robot walks away from the initial origin.
        model = getattr(env.sim, "_mj_model", None)
        if model is not None:
            model.vis.map.shadowclip = 10.0
            model.vis.map.shadowscale = 0.9
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = MjlabOnPolicyRunner(wrapped, asdict(agent), device=cfg.device)
        runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True,
                    map_location=cfg.device)
        policy = runner.get_inference_policy(device=cfg.device)
        viewer = NativeMujocoViewer if cfg.viewer == "native" else ViserPlayViewer
        viewer(wrapped, policy).run()
    finally:
        env.close()


if __name__ == "__main__":
    main()
