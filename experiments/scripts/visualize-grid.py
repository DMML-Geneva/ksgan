import argparse
import os
import pickle
import yaml

import matplotlib
from matplotlib import pyplot as plt
import numpy as np

from src.utils import get_grid
from src.data.utils import open_archived_tensorstore

parser = argparse.ArgumentParser(description="Visualize models")
parser.add_argument(
    "--config",
    type=str,
    metavar="CONFIG",
    help="Configuration file for model and training and evaluation and visualization",
)
parser.add_argument(
    "--data-path",
    type=str,
    metavar="DATAPATH",
    help="Path to read raw data",
)
parser.add_argument(
    "--eval-data-path",
    type=str,
    metavar="EVALDATAPATH",
    help="Path to read evaluation results",
)
parser.add_argument(
    "--out-path",
    type=str,
    metavar="OUTPATH",
    help="Path to write figures",
)
parser.add_argument(
    "--group",
    type=str,
    metavar="GROUP",
    help="Name of group to visualize",
)
parser.add_argument(
    "--simulation-budget",
    type=int,
    metavar="SIMBUDGET",
    help="Simulation budget for training",
)
parser.add_argument(
    "--model-seed",
    type=int,
    metavar="MODELSEED",
    help="Seed to initialize model weights",
)


def get_group(groups_list, group_name):
    return next(
        filter(lambda group: group["name"] == group_name, groups_list)
    )["models"]


def main():
    args = parser.parse_args()
    os.makedirs(args.out_path, exist_ok=True)
    with open(args.config, "r") as fin:
        cfg = yaml.load(fin, Loader=yaml.SafeLoader)

    datasets = cfg["datasets"]
    group = get_group(cfg["visualization"]["groups"], args.group)
    hist_colormap = matplotlib.colormaps["viridis"].with_extremes(bad="k")
    fig_hist, axs_hist = plt.subplots(
        nrows=(_nrows := len(datasets)),
        ncols=(_ncols := len(group) + 1),
        figsize=(4 * _ncols + 0.5, 4 * _nrows + 0.5),
        squeeze=False,
    )
    fig_critic, axs_critic = plt.subplots(
        nrows=(_nrows := len(datasets)),
        ncols=(_ncols := len(group)),
        figsize=(4 * _ncols + 0.5, 4 * _nrows + 0.5),
        squeeze=False,
    )
    axs_hist[0, 0].set_title("Data")
    for idx, model in enumerate(group):
        axs_hist[0, idx + 1].set_title(model)
        axs_critic[0, idx].set_title(model)
    for idx, dataset in enumerate(datasets):
        axs_hist[idx, 0].annotate(
            dataset,
            (-0.05, 0.5),
            xycoords="axes fraction",
            rotation=90,
            va="center",
        )
        axs_critic[idx, 0].annotate(
            dataset,
            (-0.05, 0.5),
            xycoords="axes fraction",
            rotation=90,
            va="center",
        )
    for idx_dataset, dataset in enumerate(datasets):
        samples = [
            np.array(
                open_archived_tensorstore(
                    os.path.join(args.data_path, dataset, "test")
                )
            )
        ]
        with open(
            os.path.join(args.data_path, dataset, "extent.pkl"), "rb"
        ) as f:
            extent = pickle.load(f)
        grid = get_grid(
            extent,
            resolution=(resolution := cfg["evaluation"]["grid_resolution"]),
        )[0].numpy()
        grid_x, grid_y = grid[:, 0].reshape(
            resolution,
            resolution,
        ), grid[:, 1].reshape(
            resolution,
            resolution,
        )
        for idx_model, model in enumerate(group):
            path_prefix = (
                args.eval_data_path,
                dataset,
                model,
                str(args.simulation_budget),
                str(args.model_seed),
                "artifacts",
            )
            samples.append(
                np.load(
                    os.path.join(
                        *path_prefix,
                        "samples.npy",
                    )
                )
            )
            if os.path.isfile(
                os.path.join(*path_prefix, "critic_landscape.npy")
            ):
                landscape = np.load(
                    os.path.join(*path_prefix, "critic_landscape.npy")
                )
            else:
                landscape = np.exp(
                    np.load(
                        os.path.join(*path_prefix, "log_prob_landscape.npy")
                    )
                )
            axs_critic[idx_dataset, idx_model].pcolormesh(
                grid_x,
                grid_y,
                landscape.reshape(resolution, resolution),
            )
            axs_critic[idx_dataset, idx_model].set(
                xlim=(extent[0][0], extent[1][0]),
                ylim=(extent[0][1], extent[1][1]),
            )
            axs_critic[idx_dataset, idx_model].axis("off")
            axs_critic[idx_dataset, idx_model].axis("equal")
        range_min, range_max = min(map(np.min, samples)), max(
            map(np.max, samples)
        )
        histograms = tuple(
            np.histogram2d(
                _samples[:, 0],
                _samples[:, 1],
                bins=resolution,
                range=((range_min, range_max), (range_min, range_max)),
            )
            for _samples in samples
        )
        max_count = max(map(lambda hist: hist[0].max(), histograms))
        hist_normalize = matplotlib.colors.Normalize(vmin=0, vmax=max_count)
        for idx, hist in enumerate(histograms):
            axs_hist[idx_dataset, idx].pcolormesh(
                hist[1],
                hist[2],
                np.ma.masked_array(hist[0].T, mask=hist[0].T == 0.0),
                cmap=hist_colormap,
                norm=hist_normalize,
            )
            axs_hist[idx_dataset, idx].set(
                xlim=(range_min, range_max),
                ylim=(range_min, range_max),
            )
            axs_hist[idx_dataset, idx].axis("off")
            axs_hist[idx_dataset, idx].axis("equal")

    fig_hist.savefig(
        os.path.join(args.out_path, "hist.png"),
        bbox_inches="tight",
        dpi=300,
    )
    fig_critic.savefig(
        os.path.join(args.out_path, "critic.png"),
        bbox_inches="tight",
        dpi=300,
    )


if __name__ == "__main__":
    main()
