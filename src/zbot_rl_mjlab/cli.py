"""Small local CLI: one task, finite smoke/evaluation, native mjlab training."""

import argparse
from dataclasses import asdict, replace
import importlib.metadata
import json
import re
from pathlib import Path

import torch

from . import TASK_ID, walking_env_cfg, walking_ppo_cfg
from .task_card import OBSERVATION_DIM, WalkingTaskCard
from .runner import checkpoint_config


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def latest_checkpoint(root):
    """Newest run by modification time; largest numeric iteration within that run."""
    candidates = [p for p in Path(root).rglob("model_*.pt") if p.stem[6:].isdigit()]
    if not candidates:
        raise FileNotFoundError(f"No model_<iteration>.pt under {root}; run train first.")
    newest = max(candidates, key=lambda p: p.stat().st_mtime).parent
    return max((p for p in candidates if p.parent == newest), key=lambda p: int(p.stem[6:]))


def parser():
    result = argparse.ArgumentParser(description="ZBot 6DOF walking on mjlab")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("describe", help="Print task and environment parameters")
    train = commands.add_parser("train", help="PPO training on one GPU")
    train.add_argument("--num-envs", type=positive_int, default=1024)
    train.add_argument("--iterations", type=positive_int, default=15000)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--log-root", type=Path, default=Path("logs/rsl_rl"))
    train.add_argument("--run-name", default="")
    train.add_argument("--resume", type=Path, help="New-project checkpoint, including optimizer")
    train.add_argument("--save-interval", type=positive_int, default=50)
    for name in ("smoke", "evaluate", "play"):
        sub = commands.add_parser(name)
        sub.add_argument("--num-envs", type=positive_int, default=4 if name == "smoke" else 1)
        if name != "play":
            sub.add_argument("--seed", type=int, default=42)
        sub.add_argument("--device", default="cuda:0")
        if name != "smoke":
            sub.add_argument("--checkpoint", type=Path)
            sub.add_argument("--log-root", type=Path, default=Path("logs/rsl_rl"))
        if name == "play":
            sub.add_argument("--viewer", choices=("native", "viser"), default="native")
        else:
            sub.add_argument("--steps", type=positive_int, default=128 if name == "smoke" else 300)
            sub.add_argument("--output", type=Path, default=Path(f"outputs/{name}.json"))
            if name == "smoke":
                sub.add_argument("--actions", choices=("zero", "random"), default="zero")
            else:
                sub.add_argument("--video", type=Path, help="Optional finite MP4 recording")
    return result


def require_device(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. GPU simulation needs access to the NVIDIA driver.")


def restored_configs(checkpoint, num_envs):
    """Reconstruct physical/control and PPO settings, allowing only environment count."""
    from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

    config = checkpoint_config(checkpoint)
    card_data = dict(config["task_card"])
    for key in ("joint_speed_range", "gravity"):
        if key in card_data:
            card_data[key] = tuple(card_data[key])
    card = replace(WalkingTaskCard(**card_data), num_envs=num_envs)
    ppo = dict(config["ppo"])
    for key in ("actor", "critic"):
        model = dict(ppo[key])
        model["hidden_dims"] = tuple(model["hidden_dims"])
        ppo[key] = RslRlModelCfg(**model)
    ppo["algorithm"] = RslRlPpoAlgorithmCfg(**ppo["algorithm"])
    agent = RslRlOnPolicyRunnerCfg(**ppo)
    env = walking_env_cfg(card=card)
    env.seed = agent.seed
    return env, agent


def run_train(args):
    from mjlab.scripts.train import TrainConfig, launch_training

    require_device("cuda:0")
    if args.resume:
        cfg, agent = restored_configs(args.resume.resolve(strict=True), args.num_envs)
    else:
        cfg = walking_env_cfg(card=WalkingTaskCard(num_envs=args.num_envs))
        agent = walking_ppo_cfg()
    cfg.seed = args.seed
    agent.seed = args.seed
    agent.max_iterations = args.iterations
    agent.save_interval = args.save_interval
    agent.run_name = args.run_name
    if args.resume:
        path = args.resume.resolve(strict=True)
        agent.resume = True
        agent.load_run = re.escape(path.parent.name) + "$"
        if path.parent.parent != (args.log_root / agent.experiment_name).resolve():
            raise ValueError("--resume must be inside --log-root/6dof_bipedal_walking/<run>/")
        agent.load_checkpoint = re.escape(path.name) + "$"
    launch_training(TASK_ID, TrainConfig(env=cfg, agent=agent, log_root=str(args.log_root)))


def run_play(args):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
    from .runner import WalkingRunner

    require_device(args.device)
    checkpoint = (args.checkpoint or latest_checkpoint(args.log_root)).resolve(strict=True)
    cfg, agent = restored_configs(checkpoint, args.num_envs)
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = WalkingRunner(wrapped, asdict(agent), device=args.device)
        runner.load(str(checkpoint), load_cfg={"actor": True}, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        viewer = NativeMujocoViewer if args.viewer == "native" else ViserPlayViewer
        viewer(wrapped, policy).run()
    finally:
        env.close()


def run_rollout(args):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_runner_cls
    from mjlab.utils.wrappers import VideoRecorder

    require_device(args.device)
    checkpoint = None
    if args.command == "evaluate":
        checkpoint = args.checkpoint or latest_checkpoint(args.log_root)
        checkpoint = checkpoint.resolve(strict=True)
    torch.manual_seed(args.seed)
    if checkpoint:
        cfg, agent = restored_configs(checkpoint, args.num_envs)
    else:
        cfg = walking_env_cfg(card=WalkingTaskCard(num_envs=args.num_envs))
        agent = walking_ppo_cfg()
    cfg.seed = args.seed
    video = getattr(args, "video", None)
    base_env = ManagerBasedRlEnv(
        cfg, device=args.device, render_mode="rgb_array" if video else None
    )
    env = base_env
    try:
        if video:
            video.parent.mkdir(parents=True, exist_ok=True)
            env = VideoRecorder(
                base_env,
                video_folder=video.parent,
                step_trigger=lambda step: step == 0,
                video_length=args.steps,
                name_prefix=video.stem,
                disable_logger=True,
            )
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        policy = None
        if checkpoint:
            runner_cls = load_runner_cls(TASK_ID)
            if runner_cls is None:
                raise RuntimeError("Walking runner is not registered")
            runner = runner_cls(wrapped, asdict(agent), device=args.device)
            runner.load(
                str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=args.device
            )
            policy = runner.get_inference_policy(device=args.device)
        obs = wrapped.get_observations()
        reward_sum = 0.0
        resets = 0
        heights = []
        for _ in range(args.steps):
            with torch.inference_mode():
                if policy is not None:
                    action = policy(obs)
                elif args.actions == "zero":
                    action = torch.zeros((args.num_envs, 6), device=args.device)
                else:
                    action = torch.randn((args.num_envs, 6), device=args.device) * 0.2
                obs, reward, done, _ = wrapped.step(action)
            if not torch.isfinite(reward).all() or not all(
                torch.isfinite(obs[key]).all() for key in ("actor", "critic")
            ):
                raise RuntimeError("Non-finite observation or reward during rollout")
            if obs["actor"].shape != (args.num_envs, OBSERVATION_DIM):
                raise RuntimeError(f"Observation contract broken: {obs['actor'].shape}")
            reward_sum += float(reward.mean())
            resets += int(done.sum())
            base_id = base_env.scene["robot"].find_bodies("base")[0][0]
            heights.append(
                float(base_env.scene["robot"].data.body_link_pos_w[:, base_id, 2].mean())
            )
        data = {
            "task": TASK_ID,
            "steps": args.steps,
            "num_envs": args.num_envs,
            "seed": args.seed,
            "device": args.device,
            "finite": True,
            "observation_dim": OBSERVATION_DIM,
            "action_dim": 6,
            "mean_reward_per_step": reward_sum / args.steps,
            "episode_resets": resets,
            "mean_base_height": sum(heights) / len(heights),
            "checkpoint": str(checkpoint) if checkpoint else None,
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("mjlab", "mujoco", "rsl-rl-lib", "torch")
            },
        }
        if video:
            generated = video.parent / f"{video.stem}-step-0.mp4"
            if not generated.exists():
                raise RuntimeError(f"Video recorder did not produce {generated}")
            generated.replace(video)
            data["video"] = str(video.resolve())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, indent=2) + "\n")
        print(json.dumps(data, indent=2))
    finally:
        env.close()


def main():
    args = parser().parse_args()
    if args.command == "describe":
        print(
            json.dumps(
                {
                    "task": TASK_ID,
                    "observation_dim": OBSERVATION_DIM,
                    "action_dim": 6,
                    "card": asdict(WalkingTaskCard()),
                },
                indent=2,
            )
        )
    elif args.command == "train":
        run_train(args)
    elif args.command == "play":
        run_play(args)
    else:
        run_rollout(args)


if __name__ == "__main__":
    main()
