from typing import Union, Sequence, Tuple
import warnings

import torch
import wandb

from src.models.nn import *
from src.models.base import BaseModel
from src.training.utils import do_gradient_descend_step, TrainingWarning


class NF(BaseModel):
    has_rsample = True
    has_log_prob = True
    has_critic = False

    def __init__(
        self,
        data_shape: Union[int, Sequence[int]],
        flow_cfg,
        n_stacked=1,
    ):
        super().__init__()
        self.flow = eval(flow_cfg["build"])(
            shape=data_shape,
            **flow_cfg["parameters"],
        )

    def sample_noise(self, shape: Union[int, Sequence[int]]):
        return self.flow().base.rsample(shape)

    def transform_noise(self, noise: torch.Tensor):
        return self.flow().transform.inv(noise)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flow().log_prob(x)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self(x)

    def sample(self, shape: Sequence[int]) -> torch.Tensor:
        return self.flow().rsample(shape)

    def rsample_and_log_prob(
        self, shape: Sequence[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.flow().rsample_and_log_prob(shape)


class NFLoss(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -self.model.log_prob(x).mean()


def get_model(data_shape, cfg, **kwargs):
    return NF(data_shape, **cfg)


def get_loss_fn(cfg, model):
    return NFLoss(model)


def step(x, model, loss_fn, optimizer, grad_clip, optimize=True):
    loss = loss_fn(x)
    if optimize:
        if loss.isfinite():
            loss.backward()
            do_gradient_descend_step(grad_clip, model, optimizer)
        else:
            warnings.warn("Infinite loss!", TrainingWarning)
    wandb.log(
        {
            f"{'train' if optimize else 'val'}/gen-loss": (
                loss := loss.detach().item()
            )
        }
    )
    return {"loss": loss}


Generator = NF
