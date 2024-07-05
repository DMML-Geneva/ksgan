import inspect
import importlib
import os
from itertools import chain, repeat
import pickle
import warnings

from matplotlib import pyplot as plt
import numpy as np
import torch
import wandb
from torch import autograd
from torch import optim
from tqdm import tqdm

from src.data.utils import get_dataloaders
from src.utils import (
    DefaultOrderedDict,
    set_torch_defaults,
    set_seed,
    wandb_log_params,
    TrainingWarning,
    load_model,
    get_grid,
    get_func_on_data,
)
from src.evaluation.utils import get_samples


class CompositeLRScheduler:
    # Based on https://github.com/pytorch/fairseq/blob/265df7144c79446f5ea8d835bda6e727f54dad9d/fairseq/optim/composite.py#L161
    def __init__(self, lr_schedulers: dict = None):
        if lr_schedulers is None:
            self.lr_schedulers = dict()
        else:
            self.lr_schedulers = lr_schedulers

    def __iter__(self):
        return iter(self.lr_schedulers.items())

    def __getitem__(self, item):
        return self.lr_schedulers[item]

    def add_schedulers(self, lrs_dict):
        self.lr_schedulers.update(lrs_dict)

    def state_dict(self):
        """Return the LR scheduler state dict."""
        return {k: s.state_dict() for k, s in self.lr_schedulers.items()}

    def load_state_dict(self, state_dict):
        """Load an LR scheduler state dict."""
        for k, state in state_dict.items():
            self.lr_schedulers[k].load_state_dict(state)

    def step(self, metrics=None, epoch=None):
        """Update the learning rate at the end of the given epoch."""
        if metrics is None:
            for s in self.lr_schedulers.values():
                s.step()
        elif isinstance(metrics, (int, float)):
            for s in self.lr_schedulers.values():
                s.step(metrics)
        else:
            for k, s in self.lr_schedulers.items():
                s.step(metrics[k])

    def get_last_lr(self):
        return {k: lrs.get_last_lr() for k, lrs in self.lr_schedulers.items()}


class CompositeOptimizer:
    # Based on https://github.com/pytorch/fairseq/blob/265df7144c79446f5ea8d835bda6e727f54dad9d/fairseq/optim/composite.py#L127
    def __init__(self, optimizers: dict = None):
        if optimizers is None:
            self.optimizers = dict()
        else:
            self.optimizers = optimizers

    def __iter__(self):
        return iter(self.optimizers.items())

    def __getitem__(self, item):
        return self.optimizers[item]

    def add_optimizers(self, optimizers_dict):
        self.optimizers.update(optimizers_dict)

    def step(self, closure=None, groups=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for k, opt in self.optimizers.items():
            if groups is None or k in groups:
                opt.step()

        return loss

    def zero_grad(self, set_to_none=False):
        for opt in self.optimizers.values():
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        """Return the LR scheduler state dict."""
        return {k: s.state_dict() for k, s in self.optimizers.items()}

    def load_state_dict(self, state_dict):
        """Load an LR scheduler state dict."""
        for k, state in state_dict.items():
            self.optimizers[k].load_state_dict(state)

    def add_param_group(self, param_group):
        for k, params in param_group.items():
            self.optimizers[k].add_param_group(params)


def get_optimizer(cfg, model: torch.nn.Module, device):
    if "type" in cfg.keys():
        grad_clip = dict()
        grad_clip["norm"] = cfg.pop("gradient_norm_clip", None)
        grad_clip["value"] = cfg.pop("gradient_value_clip", None)
        lr_scheduler_cfg = cfg.pop("lr_scheduler")
        optimizer_class = getattr(optim, cfg.pop("type"))
        fused_available = (
            "fused" in inspect.signature(optimizer_class).parameters
        )
        if fused_available and device.type == "cuda" and "fused" not in cfg:
            cfg["fused"] = True
        # Copied form the one and only: https://github.com/karpathy/nanoGPT/blob/325be85d9be8c81b436728a420e85796c57dba7e/model.py#L263
        param_dict = {pn: p for pn, p in model.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameter that is at least 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings + kernels decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {
                "params": decay_params,
                "weight_decay": cfg.pop("weight_decay", 0.0),
            },
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        optimizer = optimizer_class(
            params=optim_groups,
            **cfg,
        )
        if (
            lr_scheduler_cfg is not None
            and lr_scheduler_cfg["name"] is not None
        ):
            lr_scheduler = getattr(
                optim.lr_scheduler, lr_scheduler_cfg["name"]
            )(optimizer, **lr_scheduler_cfg["params"])
        else:
            lr_scheduler = None
    else:
        grad_clip = dict()
        lr_scheduler = dict()
        optimizer = dict()
        for key, key_cfg in cfg.items():
            _optimizer, _grad_clip, _lr_scheduler = get_optimizer(
                key_cfg, model.get_submodule(key), device
            )
            grad_clip[key] = _grad_clip
            if _lr_scheduler is not None:
                lr_scheduler[key] = _lr_scheduler
            optimizer[key] = _optimizer
        optimizer = CompositeOptimizer(optimizer)
        lr_scheduler = CompositeLRScheduler(lr_scheduler)
    return optimizer, grad_clip, lr_scheduler


def do_gradient_descend_step(grad_clip, model, optimizer, key=None):
    if key is not None:
        grad_clip = grad_clip[key]
        optimizer = optimizer[key]
        model = model.get_submodule(key)
    if isinstance(optimizer, CompositeOptimizer):
        for optimizer_key, _ in optimizer:
            do_gradient_descend_step(
                grad_clip, model, optimizer, optimizer_key
            )
    else:
        step = True
        grads = [
            p.grad.detach() for p in model.parameters() if p.grad is not None
        ]
        for g in grads:
            if g.isnan().any() or g.isinf().any():
                step = False
                warnings.warn(
                    (
                        "Inf/NaN gradient found!"
                        if key is None
                        else f"Inf/NaN gradient found in {key}!"
                    ),
                    TrainingWarning,
                )
                break
        if step:
            if grad_clip["value"]["value"] is not None:
                torch.nn.utils.clip_grad_value_(
                    model.parameters(),
                    grad_clip["value"]["value"],
                )
            if grad_clip["norm"]["value"] is not None:
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip["norm"]["value"],
                    norm_type=grad_clip["norm"]["type"],
                    error_if_nonfinite=False,
                )
                if not norm.isfinite():
                    step = False
                    warnings.warn(
                        (
                            "Infinite gradient norm found!"
                            if key is None
                            else f"Infinite gradient norm found in {key}!"
                        ),
                        TrainingWarning,
                    )
        if step:
            optimizer.step()


def save_model(
    path, model, optimizer, lr_scheduler, best_val_loss=None, best_weights=None
):
    checkpoint = {
        "model": {k: v.cpu() for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": (
            lr_scheduler.state_dict() if lr_scheduler is not None else None
        ),
        "best_val_loss": best_val_loss,
        "best_model": best_weights,
    }
    torch.save(checkpoint, os.path.join(path, "model.pt"))


def log_wandb_hist(tensor, wandb_key, prefix=None, commit=False):
    tmin = tensor.min().item()
    tmax = tensor.max().item()
    nbins = tensor.shape[0] // 8
    tensor = tensor.histc(bins=nbins, min=tmin, max=tmax)
    tensor = tensor.cpu().detach().clone()
    bins = torch.linspace(tmin, tmax, steps=nbins + 1)
    wandb.log(
        {
            (
                wandb_key if prefix is None else f"{prefix}/{wandb_key}"
            ): wandb.Histogram(np_histogram=(tensor.tolist(), bins.tolist())),
        },
        commit=commit,
    )


def optimizer_to(optim, device):
    # Copied from https://github.com/pytorch/pytorch/issues/8741#issuecomment-402129385 and modified
    if isinstance(optim, CompositeOptimizer):
        for _, _optim in optim:
            optimizer_to(_optim, device)
    else:
        for param in optim.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(
                                device
                            )


def scheduler_to(sched, device):
    # Copied from https://github.com/pytorch/pytorch/issues/8741#issuecomment-402129385 and modified
    if isinstance(sched, CompositeLRScheduler):
        for _, _sched in sched:
            scheduler_to(_sched, device)
    else:
        for param in sched.__dict__.values():
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)


def train_fn(
    epochs,
    model,
    loss_fn,
    step_fn,
    optimizer,
    grad_clip,
    lr_scheduler,
    dataloaders,
    device,
    detect_anomaly=False,
    checkpoint=None,
    vis_path=None,
    vis_freq=-1,
    vis_bs=1024,
    vis_n=65536,
    vis_range=((-4, 4), (-4, 4)),
    vis_grid=None,
    vis_grid_resolution=None,
    vis_noise=None,
    vis_fn=None,
):
    if checkpoint is None:
        best_val_loss = float("inf")
        best_weights = {k: v.cpu() for k, v in model.state_dict().items()}
        loss_hist = {
            "train": DefaultOrderedDict(list),
            "validation": DefaultOrderedDict(list),
        }
    else:
        best_val_loss = checkpoint["best_val_loss"]
        best_weights = checkpoint["best_model"]
        loss_hist = checkpoint["loss_history"]
    if vis_grid is not None:
        grid_x, grid_y = vis_grid[:, 0].reshape(
            vis_grid_resolution, vis_grid_resolution
        ), vis_grid[:, 1].reshape(vis_grid_resolution, vis_grid_resolution)
    if model.has_rsample and vis_freq != -1 and model.epochs_trained == 0:
        if vis_grid is not None:
            fig = get_model_visualization_figure(
                device,
                grid_x,
                grid_y,
                model,
                vis_bs,
                vis_grid,
                vis_grid_resolution,
                vis_n,
                vis_range,
            )
            fig.savefig(
                os.path.join(vis_path, "0.png"),
                dpi=150,
                bbox_inches="tight",
                transparent=True,
                pad_inches=0,
            )
            plt.close(fig)
        else:
            with torch.no_grad():
                vis_fn(
                    model.transform_noise(vis_noise).cpu().numpy(),
                    os.path.join(vis_path, "0.png"),
                )
    with autograd.set_detect_anomaly(detect_anomaly):
        with tqdm(
            range(
                model.epochs_trained.item() + 1,
                model.epochs_trained.item() + epochs + 1,
            ),
            unit="epoch",
        ) as tq:
            for epoch in tq:
                model.train()
                train_data_iter = iter(dataloaders["train"])
                while True:
                    try:
                        optimizer.zero_grad(set_to_none=True)
                        loss_hist["train"][epoch].append(
                            step_fn(
                                next(train_data_iter),
                                model,
                                loss_fn,
                                optimizer,
                                grad_clip,
                            )
                        )
                    except StopIteration:
                        break
                if "loss" in next(iter(loss_hist["train"][epoch])):
                    train_loss = np.mean(
                        tuple(l["loss"] for l in loss_hist["train"][epoch])
                    )
                else:
                    train_loss = {
                        "critic": np.nanmean(
                            tuple(
                                l.get("critic-loss", np.nan)
                                for l in loss_hist["train"][epoch]
                            )
                        ),
                        "generator": np.nanmean(
                            tuple(
                                l.get("generator-loss", np.nan)
                                for l in loss_hist["train"][epoch]
                            )
                        ),
                    }
                model.epoch_completed()
                model.eval()
                with torch.no_grad():
                    validation_data_iter = iter(dataloaders["validation"])
                    while True:
                        try:
                            loss_hist["validation"][epoch].append(
                                step_fn(
                                    next(validation_data_iter),
                                    model,
                                    loss_fn,
                                    optimizer,
                                    grad_clip,
                                    optimize=False,
                                )
                            )
                        except StopIteration:
                            break
                    if "loss" in next(iter(loss_hist["validation"][epoch])):
                        valid_loss = np.mean(
                            tuple(
                                l["loss"]
                                for l in loss_hist["validation"][epoch]
                            )
                        )
                    else:
                        valid_loss = {
                            "critic": np.mean(
                                tuple(
                                    l["critic-loss"]
                                    for l in loss_hist["validation"][epoch]
                                )
                            ),
                            "generator": np.mean(
                                tuple(
                                    l["generator-loss"]
                                    for l in loss_hist["validation"][epoch]
                                )
                            ),
                        }
                    if lr_scheduler is not None:
                        lr_scheduler.step(train_loss)
                        lr_d = {"epoch": epoch}
                        if isinstance(optimizer, CompositeOptimizer):
                            for key, o in optimizer:
                                lr_d[f"lr/{key}"] = next(iter(o.param_groups))[
                                    "lr"
                                ]
                        else:
                            lr_d["lr/model"] = next(
                                iter(optimizer.param_groups)
                            )["lr"]
                        wandb.log(lr_d)
                    else:
                        wandb.log({"epoch": epoch})
                if isinstance(valid_loss, float):
                    if valid_loss < best_val_loss:
                        best_val_loss = valid_loss
                        best_weights = {
                            k: v.cpu() for k, v in model.state_dict().items()
                        }

                tq.set_postfix(
                    train_loss=(
                        train_loss
                        if isinstance(train_loss, float)
                        else train_loss["generator"]
                    ),
                    valid_loss=(
                        valid_loss
                        if isinstance(valid_loss, float)
                        else valid_loss["generator"]
                    ),
                )
                if (
                    model.has_rsample
                    and vis_freq != -1
                    and model.epochs_trained % vis_freq == 0
                ):
                    if vis_grid is not None:
                        fig = get_model_visualization_figure(
                            device,
                            grid_x,
                            grid_y,
                            model,
                            vis_bs,
                            vis_grid,
                            vis_grid_resolution,
                            vis_n,
                            vis_range,
                        )
                        fig.savefig(
                            os.path.join(
                                vis_path, f"{model.epochs_trained.item()}.png"
                            ),
                            dpi=150,
                            bbox_inches="tight",
                            transparent=True,
                            pad_inches=0,
                        )
                        plt.close(fig)
                    else:
                        with torch.no_grad():
                            vis_fn(
                                model.transform_noise(vis_noise).cpu().numpy(),
                                os.path.join(
                                    vis_path,
                                    f"{model.epochs_trained.item()}.png",
                                ),
                            )

    return (
        loss_hist,
        best_val_loss,
        (
            best_weights
            if best_val_loss != float("inf")
            else {k: v.cpu() for k, v in model.state_dict().items()}
        ),
    )


def get_model_visualization_figure(
    device,
    grid_x,
    grid_y,
    model,
    vis_bs,
    vis_grid,
    vis_grid_resolution,
    vis_n,
    vis_range,
):
    model.eval()
    samples = get_samples(model, n=vis_n, batch_size=vis_bs)
    if model.has_critic:
        critic_eval = get_func_on_data(
            model.critic,
            x=vis_grid,
            batch_size=vis_bs,
            comp_device=device,
        )
        fig, axs = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(8, 4),
            squeeze=True,
        )
        axs[1].hist2d(
            samples[:, 0],
            samples[:, 1],
            bins=512,
            range=vis_range,
        )
        axs[1].axis("off")
        axs[1].axis("equal")
        axs[0].pcolormesh(
            grid_x,
            grid_y,
            critic_eval.view(vis_grid_resolution, vis_grid_resolution),
        )
        axs[0].axis("off")
        axs[0].axis("equal")
    elif model.has_log_prob:
        log_prob_eval = get_func_on_data(
            model.log_prob,
            x=vis_grid,
            batch_size=vis_bs,
            comp_device=device,
        )
        fig, axs = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(8, 4),
            squeeze=True,
        )
        axs[1].hist2d(
            samples[:, 0],
            samples[:, 1],
            bins=512,
            range=vis_range,
        )
        axs[1].axis("off")
        axs[1].axis("equal")
        axs[0].pcolormesh(
            grid_x,
            grid_y,
            np.exp(log_prob_eval).view(
                vis_grid_resolution, vis_grid_resolution
            ),
        )
        axs[0].axis("off")
        axs[0].axis("equal")
    else:
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(4, 4),
            squeeze=True,
        )
        ax.hist2d(
            samples[:, 0],
            samples[:, 1],
            bins=512,
            range=vis_range,
        )
        ax.axis("off")
        ax.axis("equal")
    return fig


def train(
    model_name,
    simulator,
    iterations,
    config,
    config_global,
    simulation_budget,
    data_path,
    out_path,
    vis_path,
    seed,
    model_seed,
    cuda,
    checkpoint_path=None,
    is_dataset=False,
):
    warnings.filterwarnings("always", category=TrainingWarning)
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(vis_path, exist_ok=True)
    if checkpoint_path is not None:
        with open(os.path.join(checkpoint_path, "wandb_group"), "r") as f:
            wandb_group = f.read()
    else:
        if simulation_budget != -1:
            prefix = f"sb{simulation_budget}--"
        else:
            prefix = ""
        if (exp_name := os.getenv("DVC_EXP_NAME")) is not None:
            wandb_group = (
                f"{prefix}{exp_name}--{wandb.util.generate_id()}--{model_seed}"
            )
        else:
            wandb_group = f"{prefix}{wandb.util.generate_id()}--{model_seed}"
    with open(os.path.join(out_path, "wandb_group"), "w") as f:
        f.write(wandb_group)
    wandb.init(
        project=f"ksgan-{model_name}-{simulator}",
        group=wandb_group,
        name=f"{wandb_group}-train",
        job_type="train",
        dir="/dev/shm",
        id=f"{wandb_group}-train",
        resume="allow",
        settings=wandb.Settings(
            start_method="fork",
        ),
    )
    wandb.config.update(
        {
            "precision": config_global["precision"],
            "cuda": cuda,
            "model": config["model"],
            "loss": config["loss"],
            "optimizer": config["optimizer"],
            "train": config["train"],
        }
    )
    use_cuda = torch.cuda.is_available() if cuda else False
    device = torch.device("cuda" if use_cuda else "cpu")
    set_torch_defaults(config_global["precision"])
    simulator_module = importlib.import_module(
        f"src.{'datasets' if is_dataset else 'simulators'}.{simulator}"
    )
    with open(os.path.join(data_path, "std.pkl"), "rb") as f:
        std = pickle.load(f)
    model_module = importlib.import_module(f"src.models.{model_name}")
    env_context, _ = set_seed(model_seed)
    with env_context:
        model = model_module.get_model(
            simulator_module.shape,
            cfg=config["model"],
            std=torch.from_numpy(std).to(torch.get_default_dtype()),
        )
    loss_fn = model_module.get_loss_fn(
        config["loss"] if config["loss"] is not None else dict(), model
    )
    env_context, _ = set_seed(seed)
    dataloaders = get_dataloaders(
        path=data_path,
        simulation_budget=simulation_budget,
        cached=config_global["cached_dataset"][simulator],
        cached_device=torch.device(
            "cuda"
            if use_cuda and config_global["use_cuda_cached_dataset"][simulator]
            else "cpu"
        ),
        validation_fraction=(
            None if is_dataset else config_global["validation_fraction"]
        ),
        batch_size=config["train"]["batch_size"],
        device=device,
        dataloaders=("train", "validation"),
        seed=seed,
        n_stacked=config["model"].get("n_stacked", 1),
        k=getattr(loss_fn, "k", 1),
    )
    if simulator_module.shape == (2,):
        with open(os.path.join(data_path, "extent.pkl"), "rb") as f:
            extent = pickle.load(f)
    model.to(device)
    optimizer, grad_clip, lr_scheduler = get_optimizer(
        config["optimizer"], model, device
    )
    if checkpoint_path is not None:
        checkpoint = torch.load(
            os.path.join(checkpoint_path, "model.pt"), map_location=device
        )
        with open(
            os.path.join(checkpoint_path, "loss_history.pkl"), "rb"
        ) as f:
            checkpoint["loss_history"] = pickle.load(f)
        load_model(
            checkpoint=checkpoint,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            best=False,
            map_location=device,
        )
        optimizer_to(optimizer, device)
        if lr_scheduler is not None:
            scheduler_to(lr_scheduler, device)
    else:
        checkpoint = None

    if config.get("wandb_log_freq", 0) > 0:
        model_children = tuple(model.children())
        wandb.watch(
            model_children,
            log_freq=config["wandb_log_freq"],
            log="all",
            log_graph=False,
        )
        wandb_log_params(models=model_children)
    epoch_length = (
        dataloaders["train"].n_instances() // config["train"]["batch_size"]
    )
    epochs = iterations // epoch_length
    vis_freq = config["train"].get("vis_freq", -1)
    if vis_freq != -1:
        vis_freq = max(1, 128 // epoch_length) * vis_freq
    with env_context:
        if simulator_module.shape == (2,):
            grid, _ = get_grid(
                extent,
                resolution=128,
                device=torch.device("cpu"),
            )
            loss_history, best_val_loss, best_weights = train_fn(
                epochs,
                model,
                loss_fn,
                model_module.step,
                optimizer,
                grad_clip,
                lr_scheduler,
                dataloaders,
                device,
                config.get("detect_anomaly", False),
                checkpoint=checkpoint,
                vis_path=vis_path,
                vis_freq=vis_freq,
                vis_grid=grid,
                vis_grid_resolution=128,
            )
        elif is_dataset and model.has_rsample:
            vis_noise = model.sample_noise(shape=(128,))
            loss_history, best_val_loss, best_weights = train_fn(
                epochs,
                model,
                loss_fn,
                model_module.step,
                optimizer,
                grad_clip,
                lr_scheduler,
                dataloaders,
                device,
                config.get("detect_anomaly", False),
                checkpoint=checkpoint,
                vis_path=vis_path,
                vis_freq=vis_freq,
                vis_noise=vis_noise,
                vis_fn=simulator_module.save_samples,
            )
        else:
            loss_history, best_val_loss, best_weights = train_fn(
                epochs,
                model,
                loss_fn,
                model_module.step,
                optimizer,
                grad_clip,
                lr_scheduler,
                dataloaders,
                device,
                config.get("detect_anomaly", False),
                checkpoint=checkpoint,
            )

    with open(os.path.join(out_path, "loss_history.pkl"), "wb") as f:
        pickle.dump(loss_history, f)
    save_model(
        out_path,
        model,
        optimizer,
        lr_scheduler,
        best_val_loss,
        best_weights,
    )
