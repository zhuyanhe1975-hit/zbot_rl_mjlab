from types import SimpleNamespace

import pytest
import torch

from zbot_rl_mjlab import mdp
from zbot_rl_mjlab.env_cfg import step_length_walking_cfg, walking_cfg


def test_step_length_uses_same_foot_touchdown_displacement(monkeypatch):
    positions = torch.zeros(1, 2, 3)
    landings = iter((torch.tensor([[1, 0]]), torch.tensor([[0, 0]]), torch.tensor([[1, 0]])))
    sensor = SimpleNamespace(compute_first_contact=lambda dt: next(landings))
    robot = SimpleNamespace(data=SimpleNamespace(body_link_pos_w=positions),
                            find_bodies=lambda names, preserve_order: ([0, 1], names))
    env = SimpleNamespace(scene={'robot': robot, 'feet_ground_contact': sensor}, step_dt=0.02, num_envs=1, device='cpu')
    monkeypatch.setattr(mdp, '_forward_world', lambda env: torch.tensor([[1., 0., 0.]]))
    term = mdp.ForwardStepLengthReward(SimpleNamespace(params={'scale': 0.05}), env)
    assert term(env).item() == 0.0
    positions[0, 0, 0] = 0.05
    assert term(env).item() == 0.0
    positions[0, 0, 0] = 0.10
    score = term(env).item() * env.step_dt
    assert score == pytest.approx(0.10)


def test_experiment_preserves_walking_observations_and_original_reward():
    original = walking_cfg()
    for play in (False, True):
        cfg = step_length_walking_cfg(play=play)
        assert 'forward_velocity' not in cfg.rewards
        assert 'forward_step_length' in cfg.rewards
        for group in ('actor', 'critic'):
            assert list(cfg.observations[group].terms) == list(original.observations[group].terms)
    assert 'forward_velocity' in original.rewards
    assert 'forward_step_length' not in original.rewards
