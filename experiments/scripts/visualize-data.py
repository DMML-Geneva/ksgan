import argparse
import importlib
import os
import pickle
import yaml

import numpy as np
from matplotlib import pyplot as plt

from src.data.utils import open_archived_tensorstore
from src.utils import EPS

parser = argparse.ArgumentParser(description="Visualize test data")
parser.add_argument(
    "--config",
    type=str,
    metavar="CONFIG",
    help="Configuration file for evaluation",
)
parser.add_argument(
    "--simulator",
    type=str,
    metavar="SIM",
    help="Simulator to use",
)
parser.add_argument(
    "--data-path",
    type=str,
    metavar="DATAPATH",
    help="Path to read data",
)
parser.add_argument(
    "--out-path",
    type=str,
    metavar="OUTPATH",
    help="Path to write figures",
)

if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config, "r") as fin:
        cfg = yaml.load(fin, Loader=yaml.SafeLoader)
    os.makedirs(args.out_path, exist_ok=True)

    simulator = importlib.import_module(f"src.simulators.{args.simulator}")
    samples = open_archived_tensorstore(os.path.join(args.data_path, "test"))
    if simulator.shape == (2,):
        samples = np.array(samples)
        smin = samples.min(0)
        smax = samples.max(0)
        extent = (
            smin.tolist(),
            smax.tolist(),
        )
        xmin = ymin = smin.min() - EPS
        xmax = ymax = smax.max() + EPS
        fig, ax = plt.subplots(
            1, 1, figsize=(8, 8), subplot_kw=dict(aspect="equal")
        )
        ax.hist2d(
            samples[:, 0],
            samples[:, 1],
            bins=512,
            range=((xmin, xmax), (ymin, ymax)),
        )
        ax.axis("off")
        ax.axis("equal")
        fig.savefig(
            os.path.join(args.out_path, "hist.png"),
            dpi=300,
            bbox_inches="tight",
            transparent=True,
            pad_inches=0,
        )
    else:
        extent = None
    with open(os.path.join(args.data_path, "extent.pkl"), "wb") as f:
        pickle.dump(extent, f)
