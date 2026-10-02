"""PPO checkpoints carry the observation contract and training configuration."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import torch
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper

from .task_card import WalkingTaskCard

OBSERVATION_CONTRACT = "isaaclab-quat-v1"
CURRICULUM_STATE_KEY = "walking_curriculum_state"


def _canonical_task_card(card, *, checkpoint=False):
    if not isinstance(card, dict):
        raise ValueError("Checkpoint is missing its task/PPO configuration")
    data = deepcopy(card)
    if checkpoint and "reward_mode" not in data:
        # Checkpoints created before reward modes existed used the three quaternion rewards.
        data["reward_mode"] = "quat"
    try:
        return json.loads(json.dumps(asdict(WalkingTaskCard(**data))))
    except (TypeError, ValueError) as exc:
        raise ValueError("Checkpoint task card is invalid") from exc


def _environment_task_card(env):
    rewards = env.unwrapped.cfg.rewards
    ordered_names = (*("walking", "base_vel_forward"), *rewards)
    for name in dict.fromkeys(ordered_names):
        reward = rewards.get(name)
        params = getattr(reward, "params", None)
        if isinstance(params, dict) and "card" in params:
            return params["card"]
    # The velocity task has no legacy WalkingReward term; retain a serializable
    # card for checkpoint metadata without coupling the runner to old rewards.
    return WalkingTaskCard(num_envs=env.unwrapped.num_envs)


def checkpoint_config(path):
    """Read JSON-compatible metadata before creating an environment or network."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = (checkpoint.get("infos") or {}).get("walking_config")
    if not isinstance(config, dict) or config.get("contract") != OBSERVATION_CONTRACT:
        raise ValueError(
            "Checkpoint observation contract is missing or incompatible. "
            "The aligned quaternion task requires a new training run (isaaclab-quat-v1)."
        )
    if not isinstance(config.get("task_card"), dict) or not isinstance(config.get("ppo"), dict):
        raise ValueError("Checkpoint is missing its task/PPO configuration")
    result = json.loads(json.dumps(config))
    result["task_card"] = _canonical_task_card(result["task_card"], checkpoint=True)
    return result


class WalkingRunner(MjlabOnPolicyRunner):
    def __init__(self, env: RslRlVecEnvWrapper, train_cfg, log_dir=None, device="cpu"):
        # The parent strips optional model fields in place; preserve the original config.
        self.walking_config = {
            "contract": OBSERVATION_CONTRACT,
            "task_card": asdict(_environment_task_card(env)),
            "ppo": deepcopy(train_cfg),
        }
        self.walking_config = json.loads(json.dumps(self.walking_config))
        super().__init__(env, train_cfg, log_dir, device)
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            (Path(log_dir) / "walking_config.json").write_text(
                json.dumps(self.walking_config, indent=2) + "\n"
            )

    def _curriculum_reward(self):
        env = getattr(self, "env", None)
        manager = getattr(getattr(env, "unwrapped", None), "reward_manager", None)
        if manager is None:
            return None
        try:
            reward = manager.get_term_cfg("walking").func
        except (AttributeError, ValueError):
            return None
        required = ("stage", "promotion_counter", "last_metric")
        return reward if all(hasattr(reward, name) for name in required) else None

    def save(self, path: str, infos=None) -> None:
        checkpoint_infos = {**(infos or {}), "walking_config": self.walking_config}
        reward = self._curriculum_reward()
        if reward is not None:
            checkpoint_infos[CURRICULUM_STATE_KEY] = {
                "stage": reward.stage,
                "promotion_counter": reward.promotion_counter,
                "last_metric": reward.last_metric,
            }
        super().save(path, checkpoint_infos)

    def load(self, path: str, load_cfg=None, strict=True, map_location=None):
        config = checkpoint_config(path)
        saved_card = {k: v for k, v in config["task_card"].items() if k != "num_envs"}
        canonical_current = _canonical_task_card(self.walking_config["task_card"])
        current_card = {k: v for k, v in canonical_current.items() if k != "num_envs"}
        if saved_card != current_card:
            raise ValueError("Environment differs from checkpoint; restore its task card first")
        for key in ("actor", "critic", "algorithm", "num_steps_per_env", "clip_actions"):
            if config["ppo"].get(key) != self.walking_config["ppo"].get(key):
                raise ValueError(f"PPO configuration '{key}' differs from checkpoint")
        infos = super().load(path, load_cfg, strict, map_location)
        state = (infos or {}).get(CURRICULUM_STATE_KEY)
        if state is not None:
            reward = self._curriculum_reward()
            if reward is None:
                raise ValueError(
                    "Checkpoint has curriculum state but environment has no curriculum reward"
                )
            if (
                not isinstance(state, dict)
                or state.get("stage") not in (1, 2)
                or not isinstance(state.get("promotion_counter"), int)
                or state["promotion_counter"] < 0
                or not isinstance(state.get("last_metric"), (int, float))
            ):
                raise ValueError("Checkpoint curriculum state is invalid")
            reward.stage = state["stage"]
            reward.promotion_counter = state["promotion_counter"]
            reward.last_metric = float(state["last_metric"])
        return infos
