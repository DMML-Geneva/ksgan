import argparse
from copy import deepcopy
import os
import yaml

from src.evaluation.utils import (
    evaluate_rsample_real,
    model_has_rsample,
)

parser = argparse.ArgumentParser(description="Evaluate model")
parser.add_argument(
    "--model",
    type=str,
    metavar="MODEL",
    help="Model to train",
)
parser.add_argument(
    "--dataset",
    type=str,
    metavar="SIM",
    help="Simulator to use",
)
parser.add_argument(
    "--model-config",
    type=str,
    metavar="MODELCONFIG",
    help="Configuration file for model",
)
parser.add_argument(
    "--config",
    type=str,
    metavar="CONFIG",
    help="Configuration file for evaluation",
)
parser.add_argument(
    "--data-path",
    type=str,
    metavar="DATAPATH",
    help="Path to read data",
)
parser.add_argument(
    "--model-path",
    type=str,
    metavar="MODELPATH",
    help="Path to read model weights",
)
parser.add_argument(
    "--out-path",
    type=str,
    metavar="OUTPATH",
    help="Path to write full evaluation results",
)
parser.add_argument(
    "--metrics-summary-path",
    type=str,
    metavar="METRICSPATH",
    help="Path to write metrics summary",
)
parser.add_argument(
    "--cuda",
    type=bool,
    metavar="CUDA",
    help="CUDA switch",
)
parser.add_argument(
    "--seed",
    type=int,
    metavar="SEED",
    help="Seed",
)


if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.model_config, "r") as fin:
        model_cfg = yaml.load(fin, Loader=yaml.SafeLoader)
    with open(args.config, "r") as fin:
        cfg = yaml.load(fin, Loader=yaml.SafeLoader)
    os.makedirs(args.metrics_summary_path, exist_ok=True)
    if model_has_rsample(args.model):
        evaluate_rsample_real(
            model_name=args.model,
            dataset=args.dataset,
            model_config=deepcopy(model_cfg),
            config=deepcopy(cfg),
            model_path=args.model_path,
            data_path=args.data_path,
            out_path=args.out_path,
            metrics_summary_path=args.metrics_summary_path,
            seed=args.seed,
            cuda=args.cuda,
        )
    else:
        open(
            os.path.join(
                args.metrics_summary_path, "metrics-summary-rsample.json"
            ),
            "w",
        ).close()
