import argparse
import json
import os
import pickle
import yaml

import numpy as np
import pandas as pd

from src.utils import get_grid

parser = argparse.ArgumentParser(description="Visualize models")
parser.add_argument(
    "--config",
    type=str,
    metavar="CONFIG",
    help="Configuration file for model and training and evaluation and visualization",
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
    help="Path to write dataframe",
)
parser.add_argument(
    "--group",
    type=str,
    metavar="GROUP",
    help="Name of group to visualize",
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

    datasets = cfg["real-datasets"]
    group = get_group(cfg["visualization"]["groups"], args.group)
    records = list()
    for dataset in datasets:
        for model in group:
            for seed in cfg["model_seeds"]:
                record = {
                    "dataset": dataset,
                    "model": model,
                    "seed": seed,
                }
                for metric_file in ("metrics-summary-rsample.json",):
                    if (
                        os.stat(
                            file_path := os.path.join(
                                args.eval_data_path,
                                dataset,
                                model,
                                str(seed),
                                metric_file,
                            )
                        ).st_size
                        != 0
                    ):
                        with open(
                            file_path,
                            "r",
                        ) as f:
                            record.update(json.load(f))
                records.append(record)
    pd.DataFrame.from_records(records).to_csv(
        os.path.join(args.out_path, "metrics-real.csv")
    )


if __name__ == "__main__":
    main()
