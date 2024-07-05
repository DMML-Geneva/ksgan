import importlib
import os
import pickle
from _operator import itemgetter

import numpy as np
from matplotlib import pyplot as plt
from seaborn import kdeplot
import torch

from src.utils import get_grid


def get_epochs(history):
    return len(history["train"].keys())


def get_keys(history):
    return tuple(next(iter(next(iter(history["train"].values())))).keys())


def extract_keys(list_of_dicts, key):
    return tuple(map(itemgetter(key), list_of_dicts))


def get_losses(history, epochs, loss_key):
    losses = []
    locations = []
    for epoch in range(epochs):
        _losses = extract_keys(history[epoch], loss_key)
        losses.extend(_losses)
        locations.extend(
            np.linspace(epoch, epoch + 1, len(_losses), endpoint=False)
        )
    return locations, losses


def visualize_training(history_file, out_path):
    os.makedirs(out_path, exist_ok=True)
    with open(history_file, "rb") as f:
        history = pickle.load(f)
    epochs = get_epochs(history)
    loss_keys = get_keys(history)
    fig, axs = plt.subplots(
        nrows=1,
        ncols=len(loss_keys),
        figsize=(len(loss_keys) * 8, 8),
        squeeze=False,
    )
    for ax, key in zip(axs[0], loss_keys):
        train_epochs, train_losses = get_losses(history["train"], epochs, key)
        val_epochs, val_losses = get_losses(history["validation"], epochs, key)
        ax.plot(train_epochs, train_losses, label="train")
        ax.plot(val_epochs, val_losses, label="validation")
        ax.title.set_text(key)
        ax.set_xlabel("Epoch")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=2)
    fig.savefig(
        os.path.join(out_path, "training.png"),
        bbox_inches="tight",
        dpi=300,
    )


def read_landscapes(path):
    if os.path.isfile(
        log_prob_landscape_file := os.path.join(path, "log_prob_landscape.npy")
    ):
        log_prob_landscape = np.load(log_prob_landscape_file)
    else:
        log_prob_landscape = None

    if os.path.isfile(samples_file := os.path.join(path, "samples.npy")):
        samples = np.load(samples_file)
    else:
        samples = None

    if os.path.isfile(
        critic_landscape_file := os.path.join(path, "critic_landscape.npy")
    ):
        critic_landscape = np.load(critic_landscape_file)
    else:
        critic_landscape = None

    return log_prob_landscape, samples, critic_landscape


def save_landscape_figures(
    log_probs,
    samples,
    critic_landscape,
    out_path,
    grid,
    extent,
    resolution,
    file_name="hist.png",
):
    grid_x, grid_y = grid[:, 0].reshape(resolution, resolution), grid[
        :, 1
    ].reshape(resolution, resolution)
    fig, axs = plt.subplots(
        nrows=1,
        ncols=(
            ncols := 0
            + (1 if log_probs is not None else 0)
            + (1 if samples is not None else 0)
            + (1 if critic_landscape is not None else 0)
        ),
        figsize=(ncols * 8, 8),
        squeeze=False,
    )
    axs_done = 0
    if log_probs is not None:
        probs = np.exp(log_probs.reshape(resolution, resolution))
        axs[0, axs_done].pcolormesh(
            grid_x,
            grid_y,
            probs,
        )
        axs[0, axs_done].axis("off")
        axs[0, axs_done].axis("equal")
        axs[0, axs_done].set(
            xlim=(extent[0][0], extent[1][0]),
            ylim=(extent[0][1], extent[1][1]),
        )
        axs_done += 1
    if critic_landscape is not None:
        critic_landscape = critic_landscape.reshape(resolution, resolution)
        axs[0, axs_done].pcolormesh(
            grid_x,
            grid_y,
            critic_landscape,
        )
        axs[0, axs_done].axis("off")
        axs[0, axs_done].axis("equal")
        axs[0, axs_done].set(
            xlim=(extent[0][0], extent[1][0]),
            ylim=(extent[0][1], extent[1][1]),
        )
        axs_done += 1
    if samples is not None:
        smin = samples.min(0).tolist()
        smax = samples.max(0).tolist()
        axis_range = (
            (min(smin[0], extent[0][0]), max(smax[0], extent[1][0])),
            (min(smin[1], extent[0][1]), max(smax[1], extent[1][1])),
        )
        axs[0, axs_done].hist2d(
            samples[:, 0], samples[:, 1], bins=512, range=axis_range
        )
        axs[0, axs_done].axis("off")
        axs[0, axs_done].axis("equal")
        axs[0, axs_done].set(xlim=axis_range[0], ylim=axis_range[1])
        if axs_done != 0:
            for i in range(axs_done):
                axs[0, i].set(xlim=axis_range[0], ylim=axis_range[1])
        axs_done += 1
    fig.savefig(
        os.path.join(out_path, file_name),
        dpi=300,
        bbox_inches="tight",
        transparent=True,
        pad_inches=0,
    )
    plt.close(fig)


def visualize_evaluation(
    evaluation_dir, out_path, config, simulator, data_path
):
    os.makedirs(out_path, exist_ok=True)
    if os.path.isfile(
        file_path := os.path.join(
            evaluation_dir,
            "log_probs.npy",
        )
    ):
        log_probs = np.load(file_path)
    else:
        log_probs = None

    if os.path.isfile(
        file_path := os.path.join(
            evaluation_dir,
            "normalized_log_probs.npy",
        )
    ):
        normalized_log_probs = np.load(file_path)
    else:
        normalized_log_probs = None

    if os.path.isfile(
        file_path := os.path.join(
            evaluation_dir,
            "critic_scores.npy",
        )
    ):
        critic_scores = np.load(file_path)
    else:
        critic_scores = None

    fig, axs = plt.subplots(
        nrows=1,
        ncols=(
            ncols := 0
            + (1 if log_probs is not None else 0)
            + (1 if normalized_log_probs is not None else 0)
            + (1 if critic_scores is not None else 0)
        ),
        figsize=(ncols * 8, 8),
        squeeze=False,
    )
    keys = []
    if log_probs is not None:
        keys.append("log_probs")
    if normalized_log_probs is not None:
        keys.append("normalized_log_probs")
    if critic_scores is not None:
        keys.append("critic_scores")
    for ax, key in zip(
        axs[0],
        keys,
    ):
        if key == "log_probs":
            kdeplot(data=log_probs, ax=ax)
            ax.set_title("log-posterior density")
        elif key == "normalized_log_probs":
            kdeplot(data=normalized_log_probs, ax=ax)
            ax.set_title("norm. log-posterior density")
        elif key == "critic_scores":
            kdeplot(data=critic_scores, ax=ax)
            ax.set_title("critic_scores")
        else:
            pass
    fig.savefig(
        os.path.join(out_path, "evaluation.png"),
        bbox_inches="tight",
        dpi=300,
    )
    simulator_module = importlib.import_module(f"src.simulators.{simulator}")
    if simulator_module.shape == (2,):
        log_prob_landscape, samples, critic_landscape = read_landscapes(
            evaluation_dir
        )
        with open(os.path.join(data_path, "extent.pkl"), "rb") as f:
            extent = pickle.load(f)
        grid, _ = get_grid(
            extent,
            resolution=config["evaluation"]["grid_resolution"],
            device=torch.device("cpu"),
        )
        save_landscape_figures(
            log_prob_landscape,
            samples,
            critic_landscape,
            out_path,
            grid=grid.numpy(),
            extent=extent,
            resolution=config["evaluation"]["grid_resolution"],
            file_name="hist.png",
        )
