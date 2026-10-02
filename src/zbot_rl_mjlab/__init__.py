"""ZBot tasks for mjlab. Importing this package registers the walking task."""

from mjlab.tasks.registry import register_mjlab_task
from .env_cfg import walking_env_cfg, walking_ppo_cfg
from .task_card import TASK_ID
from .runner import WalkingRunner

register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=walking_env_cfg(),
    play_env_cfg=walking_env_cfg(play=True),
    rl_cfg=walking_ppo_cfg(),
    runner_cls=WalkingRunner,
)
__all__ = ["TASK_ID", "walking_env_cfg", "walking_ppo_cfg"]
