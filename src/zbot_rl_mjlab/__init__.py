from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import (
    DISTURBED_WALKING_TASK_ID,
    STEP_LENGTH_WALKING_TASK_ID,
    TASK_ID,
    WALKING_TASK_ID,
    disturbed_walking_cfg,
    disturbed_walking_ppo_cfg,
    step_length_walking_cfg,
    step_length_walking_ppo_cfg,
    stepping_cfg,
    stepping_ppo_cfg,
    walking_cfg,
    walking_ppo_cfg,
)

register_mjlab_task(task_id=TASK_ID, env_cfg=stepping_cfg(), play_env_cfg=stepping_cfg(play=True), rl_cfg=stepping_ppo_cfg())
register_mjlab_task(task_id=WALKING_TASK_ID, env_cfg=walking_cfg(), play_env_cfg=walking_cfg(play=True), rl_cfg=walking_ppo_cfg())
register_mjlab_task(task_id=DISTURBED_WALKING_TASK_ID, env_cfg=disturbed_walking_cfg(), play_env_cfg=disturbed_walking_cfg(play=True), rl_cfg=disturbed_walking_ppo_cfg())
register_mjlab_task(task_id=STEP_LENGTH_WALKING_TASK_ID, env_cfg=step_length_walking_cfg(), play_env_cfg=step_length_walking_cfg(play=True), rl_cfg=step_length_walking_ppo_cfg())
