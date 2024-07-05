from typing import Union, Sequence, Dict
import warnings

import torch
import wandb

from src.models.nn import *
from src.models.base import BaseModel
from src.training.utils import (
    do_gradient_descend_step,
    TrainingWarning,
    log_wandb_hist,
)


class GAN(BaseModel):
    has_rsample = True
    has_log_prob = False
    has_critic = True

    def __init__(
        self,
        data_shape: Union[int, Sequence[int]],
        critic_cfg,
        generator_cfg,
        n_stacked=1,
    ):
        super().__init__()
        self.critic = eval(critic_cfg["build"])(
            data_shape,
            **critic_cfg["parameters"],
            n_stacked=n_stacked,
        )
        self.generator = eval(generator_cfg["build"])(
            data_shape, **generator_cfg["parameters"], n_stacked=n_stacked
        )

    def sample_noise(self, shape: Union[int, Sequence[int]]):
        return self.generator.noise(shape)

    def transform_noise(self, noise: torch.Tensor):
        return self.generator(noise=noise)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def sample(self, shape: Sequence[int]) -> torch.Tensor:
        return self.generator.rsample(shape)


class GANLoss(torch.nn.Module):
    def __init__(self, critic: torch.nn.Module, k=1, flipped=False):
        super().__init__()
        self.critic = critic
        self.k = k
        self._critic_opt_step_counter = 0
        self.critic_grad = torch.vmap(
            torch.func.grad(
                self.critic.forward,
            ),
            in_dims=0,
            out_dims=0,
            randomness="different",
        )
        self.vectorized_critic_grad = torch.vmap(
            self.critic_grad,
            in_dims=0,
            out_dims=0,
            randomness="different",
        )
        self.vectorized_critic = torch.vmap(
            self.critic,
            in_dims=0,
            out_dims=0,
            randomness="different",
        )
        self.flipped = -1 if flipped else 1

    def increment_step_counter(self):
        self._critic_opt_step_counter += 1

    def zero_step_counter(self):
        self._critic_opt_step_counter = 0

    def run_generator_opt_step(self):
        return self._critic_opt_step_counter >= self.k

    def forward(
        self,
        x_fake: torch.Tensor,
        x_real: torch.Tensor = None,
        wandb_prefix="train",
    ) -> Dict[str, torch.Tensor]:
        if x_real is not None:
            log_r_real, log_r_fake = self.vectorized_critic(
                x := torch.stack((x_real, x_fake), dim=0),
            )
            l1 = torch.nn.functional.logsigmoid(
                self.flipped * log_r_real
            ).mean()
            l0 = torch.nn.functional.logsigmoid(
                -self.flipped * log_r_fake
            ).mean()
            with torch.no_grad():
                log_wandb_hist(
                    (log_r_sigmoid := log_r_real.sigmoid()),
                    "d-real",
                    wandb_prefix,
                )
                log_wandb_hist(
                    (log_r_prime_sigmoid := log_r_fake.sigmoid()),
                    "d-fake",
                    wandb_prefix,
                )
                f_grad_real, f_grad_fake = self.vectorized_critic_grad(x).abs()
                log_wandb_hist(
                    f_grad_real.mean(-1),
                    "d-real-grad-avg",
                    wandb_prefix,
                )
                log_wandb_hist(
                    f_grad_fake.mean(-1),
                    "d-fake-grad-avg",
                    wandb_prefix,
                )
            return {
                "loss": l1 + l0,
                "d-real-mean": log_r_sigmoid.mean(),
                "d-fake-mean": log_r_prime_sigmoid.mean(),
                "d-real-std": log_r_sigmoid.std(),
                "d-fake-std": log_r_prime_sigmoid.std(),
                "d-real-min": log_r_sigmoid.min(),
                "d-fake-min": log_r_prime_sigmoid.min(),
                "d-real-max": log_r_sigmoid.max(),
                "d-fake-max": log_r_prime_sigmoid.max(),
            }
        else:
            log_r_fake = self.critic(x_fake)
            l0 = -torch.nn.functional.logsigmoid(
                self.flipped * log_r_fake
            ).mean()
            return {"loss": l0}


def get_model(data_shape, cfg, **kwargs):
    return GAN(data_shape, **cfg)


def get_loss_fn(cfg, model):
    return GANLoss(model.critic, **cfg)


def step(x, model, loss_fn, optimizer, grad_clip, optimize=True):
    x_fake = model.generator(
        (x.shape[0],),
    )
    critic_loss = loss_fn(
        x_real=x,
        x_fake=x_fake.detach(),
        wandb_prefix="train" if optimize else "val",
    )
    if optimize:
        if critic_loss["loss"].isfinite():
            critic_loss["loss"].backward(
                retain_graph=False,
                inputs=tuple(
                    filter(
                        lambda param: param.requires_grad,
                        model.critic.parameters(),
                    )
                ),
            )
            do_gradient_descend_step(grad_clip, model, optimizer, key="critic")
            loss_fn.increment_step_counter()
        else:
            warnings.warn("Infinite critic loss!", TrainingWarning)
    if optimize:
        if loss_fn.run_generator_opt_step():
            generator_loss = loss_fn(x_fake=x_fake, wandb_prefix="train")
            if generator_loss["loss"].isfinite():
                generator_loss["loss"].backward(
                    retain_graph=False,
                    inputs=tuple(
                        filter(
                            lambda param: param.requires_grad,
                            model.generator.parameters(),
                        )
                    ),
                )
                do_gradient_descend_step(
                    grad_clip, model, optimizer, key="generator"
                )
                loss_fn.zero_step_counter()
            else:
                warnings.warn("Infinite generator loss!", TrainingWarning)
        else:
            generator_loss = dict()
    else:
        generator_loss = loss_fn(x_fake=x_fake, wandb_prefix="val")

    loss = {
        **{f"critic-{k}": v.detach().item() for k, v in critic_loss.items()},
        **{
            f"generator-{k}": v.detach().item()
            for k, v in generator_loss.items()
        },
    }
    wandb.log(
        {f"{'train' if optimize else 'val'}/{k}": (v) for k, v in loss.items()}
    )
    return loss


Generator = GAN
