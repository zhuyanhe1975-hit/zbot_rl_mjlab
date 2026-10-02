"""Quaternion policy with exact initialization from the legacy walking MLP.

The environment still supplies only its fixed 26 observations. Gravity and
heading are deterministic internal features of the supplied world quaternion;
they add no privileged information or runtime teacher dependency.
"""

from copy import deepcopy
from collections.abc import Mapping

import torch
from torch import nn
from rsl_rl.models import MLPModel

from .mdp import walking_frame


QUAT_FEATURE_MODEL = "zbot_rl_mjlab.models:QuatFeatureMLP"


def quaternion_features(observation: torch.Tensor) -> torch.Tensor:
    """Append projected gravity and heading, preserving all 26 raw inputs."""
    gravity, _, heading = walking_frame(observation[..., :4])
    return torch.cat((observation, gravity, heading.unsqueeze(-1)), dim=-1)


class QuatFeatureMLP(MLPModel):
    """RSL-RL model with a 26D external interface and a trainable 30D MLP."""

    def _get_latent_dim(self) -> int:
        if self.obs_dim != 26:
            raise ValueError("QuatFeatureMLP requires the fixed 26D quaternion observation")
        if self.obs_normalization:
            raise ValueError("QuatFeatureMLP requires unnormalized physical observations")
        return 30

    def get_latent(self, obs, masks=None, hidden_state=None) -> torch.Tensor:
        return quaternion_features(super().get_latent(obs, masks, hidden_state))

    def as_jit(self) -> nn.Module:
        return _QuatExport(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        return _QuatOnnxExport(self, verbose)


class _QuatExport(nn.Module):
    def __init__(self, model: QuatFeatureMLP):
        super().__init__()
        self.mlp = deepcopy(model.mlp)
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module()
            if model.distribution is not None
            else nn.Identity()
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.deterministic_output(self.mlp(quaternion_features(observation)))

    @torch.jit.export
    def reset(self) -> None:
        pass


class _QuatOnnxExport(_QuatExport):
    is_recurrent = False

    def __init__(self, model: QuatFeatureMLP, verbose: bool):
        super().__init__(model)
        self.verbose = verbose
        self.input_size = 26

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]


def map_legacy_state(
    model: QuatFeatureMLP, source: Mapping[str, torch.Tensor], *, init_std: float | None = None
) -> dict[str, torch.Tensor]:
    """Map legacy actor OR critic parameters without fitting an approximation.

    Source columns are yaw, gravity[3], heading, and the remaining joint/action
    inputs[19]. Unused quaternion/omega-XY columns start at zero and remain
    trainable. Floating-point reduction order may differ from the 24D MLP.
    The legacy architecture must match and must not use observation normalization.
    """
    if not isinstance(model, QuatFeatureMLP):
        raise ValueError("Target must be a QuatFeatureMLP")
    if init_std is not None and (not torch.isfinite(torch.tensor(init_std)) or init_std <= 0):
        raise ValueError("init_std must be finite and positive")
    target = model.state_dict()
    if set(source) != set(target):
        raise ValueError("Legacy state keys must match; normalized/different models are unsupported")
    mapped = {}
    for name, destination in target.items():
        original = source[name]
        if not torch.isfinite(original).all():
            raise ValueError(f"Nonfinite legacy parameter: {name}")
        if name == "mlp.0.weight":
            if original.shape != (destination.shape[0], 24):
                raise ValueError("Legacy first layer must have 24 inputs and matching hidden width")
            value = torch.zeros_like(destination)
            value[:, 6] = original[:, 0]
            value[:, 26:29] = original[:, 1:4]
            value[:, 29] = original[:, 4]
            value[:, 7:26] = original[:, 5:24]
        else:
            if original.shape != destination.shape:
                raise ValueError(f"Legacy tensor shape mismatch: {name}")
            value = original.to(destination).clone()
            if name == "distribution.std_param" and init_std is not None:
                value.fill_(init_std)
        mapped[name] = value
    return mapped
