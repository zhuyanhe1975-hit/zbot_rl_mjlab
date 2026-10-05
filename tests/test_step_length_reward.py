from types import SimpleNamespace

import pytest
import torch

from zbot_rl_mjlab import mdp
from zbot_rl_mjlab.env_cfg import step_length_walking_cfg, walking_cfg


def make_term(monkeypatch, num_envs=2):
    # Nonzero, different world origins expose accidental first-contact strides.
    positions = torch.zeros(num_envs, 2, 3)
    positions[:, :, 0] = torch.arange(num_envs).unsqueeze(-1) * 10 + 5
    sensor = SimpleNamespace(landing=torch.zeros(num_envs, 2, dtype=torch.bool))
    sensor.compute_first_contact = lambda dt: sensor.landing
    robot = SimpleNamespace(data=SimpleNamespace(body_link_pos_w=positions),
                            find_bodies=lambda names, preserve_order: ([0, 1], names))
    env = SimpleNamespace(scene={'robot': robot, 'feet_ground_contact': sensor},
                          step_dt=0.02, num_envs=num_envs, device='cpu', gate=torch.ones(num_envs),
                          episode_length_buf=torch.zeros(num_envs, dtype=torch.long))
    monkeypatch.setattr(mdp, '_forward_world', lambda env: torch.tensor([[1., 0., 0.]]).expand(num_envs, 3))
    monkeypatch.setattr(mdp, 'alternating_foot_phase', lambda env, sensor_name: env.gate)
    term = mdp.ForwardStepLengthReward(SimpleNamespace(params={
        'max_step_length': 0.4, 'step_length_scale': 0.2, 'balance_weight': 0.2,
    }), env)
    return term, env, positions, sensor


def event(term, env, sensor, feet):
    sensor.landing[:] = torch.tensor(feet, dtype=torch.bool)
    result = term(env) * env.step_dt
    env.episode_length_buf += 1
    return result


def test_initial_contact_is_only_anchor(monkeypatch):
    term, env, positions, sensor = make_term(monkeypatch)
    assert torch.equal(event(term, env, sensor, [1, 0]), torch.zeros(2))
    positions[:, 0, 0] += 0.1
    # First contact of the other foot must not introduce symmetry penalties.
    assert torch.equal(event(term, env, sensor, [0, 1]), torch.zeros(2))
    assert not term._has_step_length.any()
    expected = (1 - torch.exp(torch.tensor(-0.5))) * 0.04
    assert torch.allclose(event(term, env, sensor, [1, 0]), expected.expand(2), atol=1e-5)
    assert not term._has_step_length[:, 1].any()
    assert torch.equal(event(term, env, sensor, [0, 0]), torch.zeros(2))


def test_reward_curve_and_backward_stride(monkeypatch):
    term, env, positions, sensor = make_term(monkeypatch, num_envs=5)
    event(term, env, sensor, [1, 0])
    positions[:, 0, 0] += torch.tensor([0., -0.1, 0.1, 0.2, 0.8])
    scores = event(term, env, sensor, [1, 0])
    assert torch.allclose(scores, torch.tensor([0., 0., 1 - torch.exp(torch.tensor(-0.5)),
                                             1 - torch.exp(torch.tensor(-1.)),
                                             1 - torch.exp(torch.tensor(-2.))]) * 0.02, atol=1e-5)
    # Reward grows; equal extra distance brings less additional reward.
    assert scores[3] - scores[2] < scores[2]


def test_linear_balance_uses_raw_stride_and_frequency_gate(monkeypatch):
    term, env, positions, sensor = make_term(monkeypatch)
    event(term, env, sensor, [1, 1])
    positions[:, 0, 0] += 0.6
    event(term, env, sensor, [1, 0])
    positions[:, 1, 0] += 0.9
    env.gate[:] = torch.tensor([1., 0.5])
    scores = event(term, env, sensor, [0, 1])
    expected = ((1 - torch.exp(torch.tensor(-2.))).item() - 0.2 * 0.3) * 0.04
    assert scores[0].item() == pytest.approx(expected, abs=1e-5)
    assert scores[1].item() == pytest.approx(expected / 2, abs=1e-5)
    env.gate.zero_()
    positions[:, 0, 0] += 0.1
    assert torch.equal(event(term, env, sensor, [1, 0]), torch.zeros(2))


def test_partial_reset_clears_only_selected_histories(monkeypatch):
    term, env, positions, sensor = make_term(monkeypatch)
    event(term, env, sensor, [1, 1])
    positions[:, :, 0] += 0.2
    event(term, env, sensor, [1, 1])
    term.reset(torch.tensor([0]))
    assert not term._has_touchdown[0].any()
    assert not term._has_step_length[0].any()
    assert term._has_step_length[1].all()
    assert torch.allclose(term._last_step_lengths[1], torch.tensor([0.2, 0.2]), atol=1e-5)
    positions[:, :, 0] += 0.1
    scores = event(term, env, sensor, [1, 1])
    assert scores[0].item() == 0
    assert scores[1].item() > 0
    term.reset()
    assert not term._has_touchdown.any()
    assert not term._has_step_length.any()


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
