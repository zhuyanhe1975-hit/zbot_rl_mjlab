"""Integrate bounded policy velocities into persistent joint position targets."""

from dataclasses import dataclass
import math
import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from .task_card import JOINT_NAMES


def integrate_target(delta, actions, speed, dt, scale=1.0):
    return (delta + math.pi * speed * scale * torch.tanh(actions) * dt).clamp(-math.pi, math.pi)


@dataclass(kw_only=True)
class IntegratedJointPositionActionCfg(ActionTermCfg):
    action_scale: float = 1.0
    joint_speed_range: tuple[float, float] = (0.2, 2.0)

    def build(self, env):
        return IntegratedJointPositionAction(self, env)


class IntegratedJointPositionAction(ActionTerm):
    cfg: IntegratedJointPositionActionCfg

    def __init__(self, cfg: IntegratedJointPositionActionCfg, env):
        super().__init__(cfg, env)
        self.joint_ids, names = self._entity.find_joints(JOINT_NAMES, preserve_order=True)
        if tuple(names) != JOINT_NAMES:
            raise ValueError(f"Unexpected ZBot joint order: {names}")
        self._raw = torch.zeros((env.num_envs, 6), device=env.device)
        self.bounded_action = torch.zeros_like(self._raw)
        self.previous_action = torch.zeros_like(self._raw)
        self.delta = torch.zeros_like(self._raw)
        lo, hi = cfg.joint_speed_range
        # Original samples once per environment, not again at every episode reset.
        self.speed = torch.rand((env.num_envs, 1), device=env.device) * (hi - lo) + lo

    @property
    def action_dim(self):
        return 6

    @property
    def raw_action(self):
        return self._raw

    def process_actions(self, actions):
        self.previous_action.copy_(self.bounded_action)
        self._raw.copy_(actions)
        self.bounded_action.copy_(torch.tanh(actions))
        self.delta.copy_(
            integrate_target(
                self.delta, actions, self.speed, self._env.step_dt, self.cfg.action_scale
            )
        )

    def apply_actions(self):
        target = self.delta + self._entity.data.default_joint_pos[:, self.joint_ids]
        self._entity.set_joint_position_target(target, joint_ids=self.joint_ids)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for tensor in (self._raw, self.bounded_action, self.previous_action, self.delta):
            tensor[ids] = 0
