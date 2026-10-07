from types import SimpleNamespace

import pytest
import torch

from zbot_rl_mjlab import mdp
from zbot_rl_mjlab.env_cfg import (
    disturbed_walking_cfg,
    step_length_walking_cfg,
    stepping_cfg,
    walking_cfg,
)


def make_env():
    return SimpleNamespace(
        scene={
            "feet_ground_contact": SimpleNamespace(
                data=SimpleNamespace(found=torch.tensor([[1, 0], [0, 1], [1, 1]]))
            ),
            "left_acc": SimpleNamespace(
                data=torch.tensor(
                    [[100.0, 0.0, 0.0], [-2.0, 3.0, 0.0], [100.0, 0.0, 0.0]]
                )
            ),
            "right_acc": SimpleNamespace(
                data=torch.tensor(
                    [[0.0, 0.0, 2.0], [100.0, 0.0, 0.0], [100.0, 0.0, 0.0]]
                )
            ),
            "foot_height_scan": SimpleNamespace(
                data=SimpleNamespace(
                    heights=torch.tensor([[0.3, 0.05], [0.105, 0.3], [0.3, 0.3]])
                )
            ),
        }
    )


def test_acceleration_excludes_support_feet_and_uses_absolute_values():
    env = make_env()
    result = mdp.swing_foot_acceleration_penalty(env, ("left_acc", "right_acc"), 10.0)
    assert torch.allclose(result, torch.tensor([0.2, 0.5, 0.0]))
    env.scene["left_acc"].data[1, 0] = -1000.0
    assert (
        mdp.swing_foot_acceleration_penalty(env, ("left_acc", "right_acc"), 10.0)[1]
        == 1.0
    )
    with pytest.raises(ValueError):
        mdp.swing_foot_acceleration_penalty(env, ("left_acc", "right_acc"), 0.0)


def test_height_only_penalizes_excess_clearance_of_airborne_feet():
    env = make_env()
    result = mdp.swing_foot_excess_height_penalty(
        env, max_height=0.08, height_scale=0.05
    )
    assert torch.allclose(result, torch.tensor([0.0, 0.5, 0.0]), atol=1e-6)
    env.scene["foot_height_scan"].data.heights[1, 0] = 1.0
    assert mdp.swing_foot_excess_height_penalty(env, 0.08, 0.05)[1] == 1.0


def test_walking_experiments_get_new_sensors_and_rewards():
    original = walking_cfg()
    for factory, play in (
        (walking_cfg, False),
        (walking_cfg, True),
        (step_length_walking_cfg, False),
        (step_length_walking_cfg, True),
    ):
        cfg = factory(play=play)
        names = {sensor.prefixed_name for sensor in cfg.scene.sensors}
        for name in cfg.rewards["swing_foot_acceleration"].params[
            "acceleration_sensors"
        ]:
            assert name in names
        assert cfg.rewards["swing_foot_acceleration"].weight < 0
        assert cfg.rewards["swing_foot_excess_height"].weight < 0
        assert list(cfg.observations["actor"].terms) == list(
            original.observations["actor"].terms
        )
    cfg = stepping_cfg()
    assert "swing_foot_acceleration" not in cfg.rewards
    assert "swing_foot_excess_height" not in cfg.rewards
    assert len(cfg.scene.sensors) == 3
    cfg = disturbed_walking_cfg()
    for name in ("swing_foot_acceleration", "swing_foot_excess_height", "joint_energy"):
        assert name in cfg.rewards
    assert len(cfg.scene.sensors) == 5
