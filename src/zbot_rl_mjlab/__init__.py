from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import (
    TASK_ID,
    WALKING_TASK_ID,
    env_cfg,
    ppo_cfg,
    walking_cfg,
    walking_ppo_cfg,
)

register_mjlab_task(task_id=TASK_ID, env_cfg=env_cfg(), play_env_cfg=env_cfg(play=True), rl_cfg=ppo_cfg())
register_mjlab_task(task_id=WALKING_TASK_ID, env_cfg=walking_cfg(), play_env_cfg=walking_cfg(play=True), rl_cfg=walking_ppo_cfg())
