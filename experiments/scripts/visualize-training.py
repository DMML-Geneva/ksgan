import argparse

from src.visualization.utils import visualize_training

parser = argparse.ArgumentParser(description="Visualize model training")
parser.add_argument(
    "--history-file",
    type=str,
    metavar="HIST",
    help="File containing history",
)
parser.add_argument(
    "--out-path",
    type=str,
    metavar="OUTPATH",
    help="Path to write figures",
)

if __name__ == "__main__":
    args = parser.parse_args()
    visualize_training(history_file=args.history_file, out_path=args.out_path)
