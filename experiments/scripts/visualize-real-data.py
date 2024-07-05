import argparse
import importlib
import os
import pickle
import yaml

import numpy as np
from matplotlib import pyplot as plt

from src.data.utils import open_archived_tensorstore
from src.utils import EPS, set_seed

parser = argparse.ArgumentParser(description="Visualize test data")
parser.add_argument(
    "--config",
    type=str,
    metavar="CONFIG",
    help="Configuration file for evaluation",
)
parser.add_argument(
    "--dataset",
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
parser.add_argument(
    "--seed",
    type=int,
    metavar="SEED",
    help="Seed",
)

if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config, "r") as fin:
        cfg = yaml.load(fin, Loader=yaml.SafeLoader)
    os.makedirs(args.out_path, exist_ok=True)

    dataset = importlib.import_module(f"src.datasets.{args.dataset}")
    samples = open_archived_tensorstore(os.path.join(args.data_path, "test"))
    n_samples = cfg["evaluation"]["n_samples"]
    env_context, nprng = set_seed(args.seed)
    with env_context:
        samples_idx = nprng.choice(samples.shape[0], size=n_samples)
        dataset.save_samples(
            samples[samples_idx].read().result(),
            os.path.join(args.out_path, "data.png"),
        )
