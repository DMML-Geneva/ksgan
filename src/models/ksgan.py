from typing import Dict
import warnings

import torch
import wandb

from src.models.gan import GAN, GANLoss
from src.training.utils import (
    do_gradient_descend_step,
    TrainingWarning,
    log_wandb_hist,
)
from src.utils import parse_torch_nn_class


KSGAN = GAN


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return torch.nn.functional.hardtanh(grad_output)


def heaviside_relaxation(x: torch.Tensor, epsilon=1):
    return 1 / (1 + (-2 * epsilon * x).exp())


class KSGANLoss(GANLoss):
    def __init__(
        self,
        critic: torch.nn.Module,
        k=1,
        lmbda_sp=-1,
        critic_output_activation=None,
        mean=False,
        sorting=False,
    ):
        super().__init__(
            critic=critic,
            k=k,
        )
        if (
            act := parse_torch_nn_class(critic_output_activation, None)
        ) is not None:
            self.critic_output_activation = act()
        else:
            self.critic_output_activation = torch.nn.Identity()
        self.lmbda_sp = lmbda_sp
        self.mean = mean
        self.indicator = STE.apply

    def score_penalty(self, x):
        grad = self.critic_grad(
            x,
        )
        return torch.linalg.vector_norm(
            grad.flatten(start_dim=1), ord=2, dim=-1
        ).pow(2)

    def forward(
        self,
        x_fake: torch.Tensor,
        x_real: torch.Tensor,
        critic_step=True,
        wandb_prefix="train",
    ) -> Dict[str, torch.Tensor]:
        if critic_step:
            c = self.critic_output_activation(
                self.vectorized_critic(
                    x := torch.stack((x_real, x_fake), dim=0),
                )
            )
            score_penalty = torch.vmap(
                self.score_penalty, randomness="different"
            )(x)

            with torch.no_grad():
                c_real, c_fake = c
                score_penalty_real, score_penalty_fake = score_penalty
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
                log_wandb_hist(
                    score_penalty_real,
                    "sp-real",
                    wandb_prefix,
                )
                log_wandb_hist(
                    score_penalty_fake,
                    "sp-fake",
                    wandb_prefix,
                )
            max_likelihood_loss = c[1].mean() - c[0].mean()
            return {
                "loss": max_likelihood_loss
                + self.lmbda_sp * (score_penalty := score_penalty.mean()),
                "c-real-mean": c_real.mean(),
                "c-fake-mean": c_fake.mean(),
                "c-real-std": c_real.std(),
                "c-fake-std": c_fake.std(),
                "c-real-min": c_real.min(),
                "c-fake-min": c_fake.min(),
                "c-real-max": c_real.max(),
                "c-fake-max": c_fake.max(),
                "score_penalty": score_penalty,
                "ml-loss": max_likelihood_loss,
            }
        else:
            c = self.critic_output_activation(
                self.vectorized_critic(
                    torch.stack((x_real, x_fake), dim=0),
                )
            )
            c_below_levels_tensor = self.indicator(
                (c_diff := c.view(1, -1, 1) - c.unsqueeze(1))
            )
            neg_c_below_levels_tensor = self.indicator(-c_diff)
            concentration_error_f_levels = torch.abs(
                c_below_levels_tensor[0].mean(1)
                - c_below_levels_tensor[1].mean(1)
            )
            concentration_error_g_levels = torch.abs(
                neg_c_below_levels_tensor[0].mean(1)
                - neg_c_below_levels_tensor[1].mean(1)
            )
            concentration_error = torch.stack(
                (
                    concentration_error_f_levels,
                    concentration_error_g_levels,
                ),
                dim=0,
            )

            if self.mean:
                concentration_error = concentration_error.mean(dim=1)
            else:
                concentration_error = concentration_error.max(dim=1).values
            with torch.no_grad():
                return_dict = {
                    "conc-error-f-lvl-avg": concentration_error_f_levels.mean(),
                    "conc-error-g-lvl-avg": concentration_error_g_levels.mean(),
                    "conc-error-f-lvl-max": concentration_error_f_levels.max(),
                    "conc-error-g-lvl-max": concentration_error_g_levels.max(),
                }
            return {
                "loss": concentration_error.sum(),
                **return_dict,
            }


def get_model(data_shape, cfg, **kwargs):
    return KSGAN(data_shape, **cfg)


def get_loss_fn(cfg, model):
    return KSGANLoss(model.critic, **cfg)


def step(x, model, loss_fn, optimizer, grad_clip, optimize=True):
    x_fake = model.generator(
        (x.shape[0],),
    )
    critic_loss = loss_fn(
        x_real=x,
        x_fake=x_fake.detach(),
        wandb_prefix="train" if optimize else "val",
        critic_step=True,
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
            generator_loss = loss_fn(
                x_real=x,
                x_fake=x_fake,
                wandb_prefix="train",
                critic_step=False,
            )
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
        generator_loss = loss_fn(
            x_real=x,
            x_fake=x_fake,
            wandb_prefix="val",
            critic_step=False,
        )

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


Generator = KSGAN
