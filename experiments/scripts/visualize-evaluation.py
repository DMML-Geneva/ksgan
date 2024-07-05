import argparse
import yaml

from src.visualization.utils import visualize_evaluation

parser = argparse.ArgumentParser(description="Visualize model training")
parser.add_argument(
    "--evaluation-dir",
    type=str,
    metavar="EVAL",
    help="Directory containing evaluation results",
)
parser.add_argument(
    "--out-path",
    type=str,
    metavar="OUTPATH",
    help="Path to write figures",
)
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

if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config, "r") as fin:
        cfg = yaml.load(fin, Loader=yaml.SafeLoader)
    visualize_evaluation(
        evaluation_dir=args.evaluation_dir,
        out_path=args.out_path,
        config=cfg,
        simulator=args.simulator,
        data_path=args.data_path,
    )
