import argparse
import importlib
import os
import pickle
import shutil
import yaml

import numpy as np

from src.utils import set_seed
from src.data.utils import (
    open_tensorstore,
    archive_tensorstore,
    open_archived_tensorstore,
    BIG_ENDIAN_DTYPE_DICT,
    DATA_PRECISION,
)

parser = argparse.ArgumentParser(description="Preprocess data")
parser.add_argument(
    "--simulator",
    type=str,
    metavar="SIM",
    help="Simulator to use",
)
parser.add_argument(
    "--config",
    type=str,
    metavar="CONFIG",
    help="Configuration file",
)
parser.add_argument(
    "--in-path",
    type=str,
    metavar="INPATH",
    help="Path to read data",
)
parser.add_argument(
    "--out-path",
    type=str,
    metavar="OUTPATH",
    help="Path to write data",
)
parser.add_argument(
    "--seed",
    type=int,
    metavar="SEED",
    help="Seed",
)


def main():
    args = parser.parse_args()
    with open(args.config, "r") as fin:
        standardize = yaml.load(fin, Loader=yaml.SafeLoader)[
            "standardization"
        ][args.simulator]
    simulator = importlib.import_module(f"src.simulators.{args.simulator}")
    os.makedirs(args.out_path, exist_ok=True)
    _, numpy_rng = set_seed(args.seed)

    if not standardize:
        mean = np.zeros(
            simulator.shape,
            dtype=BIG_ENDIAN_DTYPE_DICT[DATA_PRECISION],
        )
        std = np.ones(
            simulator.shape,
            dtype=BIG_ENDIAN_DTYPE_DICT[DATA_PRECISION],
        )
        for ds_name in ("validation", "test"):
            shutil.copyfile(
                os.path.join(args.in_path, f"{ds_name}.zip"),
                os.path.join(args.out_path, f"{ds_name}.zip"),
            )
        ds_name = "train"
        raw_ds = dict()
        raw_ds[ds_name] = open_archived_tensorstore(
            os.path.join(args.in_path, ds_name),
        )

        ds = open_tensorstore(
            os.path.join(
                args.out_path,
                ds_name,
            ),
            config={
                "metadata": raw_ds[ds_name].spec().to_json()["metadata"],
                "create": True,
                "delete_existing": True,
            },
        )
        data = np.array(raw_ds[ds_name])
        numpy_rng.shuffle(data)
        ds.write(data).result()
        archive_tensorstore(
            path=os.path.join(args.out_path, ds_name), delete=True
        )
    else:
        raw_ds = dict()
        for ds_name in ("train", "validation", "test"):
            raw_ds[ds_name] = open_archived_tensorstore(
                os.path.join(args.in_path, ds_name),
            )

        mean = np.mean(raw_ds["train"], axis=0)
        std = np.std(raw_ds["train"], ddof=1, axis=0)

        for ds_name in ("train", "validation", "test"):
            ds = open_tensorstore(
                os.path.join(
                    args.out_path,
                    ds_name,
                ),
                config={
                    "metadata": raw_ds[ds_name].spec().to_json()["metadata"],
                    "create": True,
                    "delete_existing": True,
                },
            )
            data = raw_ds[ds_name]
            if ds_name == "train":
                numpy_rng.shuffle(data)
            ds.write((data - mean) / std).result()
            archive_tensorstore(
                path=os.path.join(args.out_path, ds_name), delete=True
            )
    with open(os.path.join(args.out_path, "mean.pkl"), "wb") as f:
        pickle.dump(mean, f)
    with open(os.path.join(args.out_path, "std.pkl"), "wb") as f:
        pickle.dump(std, f)


if __name__ == "__main__":
    main()
