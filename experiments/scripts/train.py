import argparse
import yaml

from src.training.utils import train

parser = argparse.ArgumentParser(description="Train model")
parser.add_argument(
    "--model",
    type=str,
    metavar="MODEL",
    help="Model to train",
)
parser.add_argument(
    "--simulator",
    type=str,
    metavar="SIM",
    help="Simulator to use",
)
parser.add_argument(
    "--simulation-budget",
    type=int,
    metavar="SIMBUDGET",
    help="Simulation budget for training",
)
parser.add_argument(
    "--config",
    type=str,
    metavar="CONFIG",
    help="Configuration file for model and training",
)
parser.add_argument(
    "--config-global",
    type=str,
    metavar="CONFIGGLOB",
    help="Global configuration file for experiments",
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
    help="Path to write model",
)
parser.add_argument(
    "--vis-path",
    type=str,
    metavar="VISPATH",
    help="Path to write visualizations",
)
parser.add_argument(
    "--cuda",
    type=bool,
    metavar="CUDA",
    help="CUDA switch",
)
parser.add_argument(
    "--model-seed",
    type=int,
    metavar="MODELSEED",
    help="Seed to initialize model weights",
)
parser.add_argument(
    "--seed",
    type=int,
    metavar="SEED",
    help="Seed",
)
parser.add_argument(
    "--is-dataset",
    type=bool,
    default=False,
    metavar="ISREALDATA",
    help="Real dataset switch",
)

if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config, "r") as fin:
        cfg = yaml.load(fin, Loader=yaml.SafeLoader)
    with open(args.config_global, "r") as fin:
        cfg_glob = yaml.load(fin, Loader=yaml.SafeLoader)
    train(
        model_name=args.model,
        simulator=args.simulator,
        iterations=cfg["train"]["iterations"],
        config=cfg,
        config_global=cfg_glob,
        simulation_budget=args.simulation_budget,
        data_path=args.data_path,
        out_path=args.out_path,
        vis_path=args.vis_path,
        seed=args.seed,
        model_seed=args.model_seed,
        cuda=args.cuda,
        is_dataset=args.is_dataset,
    )
