from typing import Dict

import torch

from src.models.gan import GAN, step, GANLoss
from src.training.utils import (
    log_wandb_hist,
)


WGAN = GAN


class WGANLoss(GANLoss):
    def __init__(
        self,
        critic: torch.nn.Module,
        k=1,
        lmbda_gp=-10,
    ):
        super().__init__(critic=critic, k=k)
        self.lmbda_gp = lmbda_gp
        self._critic_opt_step_counter = 0

    def gradient_penalty(self, x_real, x_fake):
        parameters_interpolated = torch.lerp(
            input=x_real,
            end=x_fake,
            weight=torch.rand(
                x_real.size()[:1],
                dtype=x_real.dtype,
                layout=x_real.layout,
                device=x_real.device,
            )
            .view(-1, *[1] * (x_real.ndim - 1))
            .expand_as(x_real),
        )
        grad = self.critic_grad(
            parameters_interpolated,
        )
        grad_norm = torch.linalg.vector_norm(
            grad.flatten(start_dim=1), ord=2, dim=-1
        )
        return (grad_norm - 1).pow(2)

    def forward(
        self,
        x_fake: torch.Tensor,
        x_real: torch.Tensor = None,
        wandb_prefix="train",
    ) -> Dict[str, torch.Tensor]:
        if x_real is not None:
            c_real, c_fake = self.vectorized_critic(
                x := torch.stack((x_real, x_fake), dim=0),
            )
            l1 = c_real.mean()
            l0 = c_fake.mean()
            gp = self.gradient_penalty(x_real=x_real, x_fake=x_fake).mean()
            with torch.no_grad():
                log_wandb_hist(
                    c_real,
                    "c-real",
                    wandb_prefix,
                )
                log_wandb_hist(
                    c_fake,
                    "c-fake",
                    wandb_prefix,
                )
                c_grad_real, c_grad_fake = torch.vmap(
                    self.critic_grad, randomness="different"
                )(x).abs()
                log_wandb_hist(
                    c_grad_real.mean(-1),
                    "c-real-grad-avg",
                    wandb_prefix,
                )
                log_wandb_hist(
                    c_grad_fake.mean(-1),
                    "c-fake-grad-avg",
                    wandb_prefix,
                )
            return {
                "loss": l1 - l0 + self.lmbda_gp * gp,
                "c-real-mean": c_real.mean(),
                "c-fake-mean": c_fake.mean(),
                "c-real-std": c_real.std(),
                "c-fake-std": c_fake.std(),
                "c-real-min": c_real.min(),
                "c-fake-min": c_fake.min(),
                "c-real-max": c_real.max(),
                "c-fake-max": c_fake.max(),
                "gp": gp,
            }
        else:
            c_fake = self.critic(x_fake)
            l0 = c_fake.mean()
            return {"loss": -l0}


def get_model(data_shape, cfg, **kwargs):
    return WGAN(data_shape, **cfg)


def get_loss_fn(cfg, model):
    return WGANLoss(model.critic, **cfg)


Generator = WGAN
