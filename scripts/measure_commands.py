"""Measure zbot's measured base velocity under fixed command targets."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from zbot_rl_mjlab.env_cfg import walking_env_cfg, walking_ppo_cfg
from zbot_rl_mjlab.mdp import base_ang_vel, base_lin_vel
from zbot_rl_mjlab.task_card import WalkingTaskCard
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper


def latest_checkpoint(root: Path) -> Path:
    files = list(root.rglob("model_*.pt"))
    files = [p for p in files if p.stem[6:].isdigit()]
    if not files:
        raise FileNotFoundError(f"No checkpoint under {root}")
    run = max({p.parent for p in files}, key=lambda p: p.stat().st_mtime)
    return max((p for p in files if p.parent == run), key=lambda p: int(p.stem[6:]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("outputs/command_sweep.json"))
    args = parser.parse_args()
    root = Path("logs/rsl_rl/zbot_g1_velocity")
    checkpoint = (args.checkpoint or latest_checkpoint(root)).resolve(strict=True)
    card = WalkingTaskCard(num_envs=args.num_envs)
    env = ManagerBasedRlEnv(walking_env_cfg(card=card), device=args.device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=walking_ppo_cfg().clip_actions)
    agent = walking_ppo_cfg()
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent), device=args.device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=False, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)
    command_term = env.command_manager.get_term("twist")
    commands = [(0.2, 0.0, 0.0), (0.4, 0.0, 0.0), (0.6, 0.0, 0.0), (0.0, 0.2, 0.0), (0.0, 0.0, 0.2)]
    results = []
    try:
        for target in commands:
            obs, _ = env.reset()
            command_term.vel_command_b[:] = torch.tensor(target, device=env.device)
            command_term.is_standing_env[:] = False
            command_term.is_heading_env[:] = False
            command_term.is_world_env[:] = False
            linear, angular = [], []
            for step in range(args.steps):
                command_term.vel_command_b[:] = torch.tensor(target, device=env.device)
                with torch.inference_mode():
                    action = policy(obs)
                obs, _, _, _ = wrapped.step(action)
                if step >= args.steps // 2:
                    linear.append(base_lin_vel(env).detach().cpu())
                    angular.append(base_ang_vel(env).detach().cpu())
            lin = torch.cat(linear)
            ang = torch.cat(angular)
            target_t = torch.tensor(target)
            results.append({
                "command": target,
                "measured_linear_mean": lin.mean(0).tolist(),
                "measured_linear_std": lin.std(0).tolist(),
                "measured_angular_mean": ang.mean(0).tolist(),
                "measured_angular_std": ang.std(0).tolist(),
                "linear_error_norm": float((lin[:, :2] - target_t[:2]).norm(dim=1).mean()),
                "angular_error_abs": float((ang[:, 2] - target_t[2]).abs().mean()),
            })
    finally:
        env.close()
    payload = {"checkpoint": str(checkpoint), "device": args.device, "steps": args.steps, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
