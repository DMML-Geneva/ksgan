import argparse
import importlib
import os
import pickle

from tqdm import tqdm

from src.utils import set_seed
from src.data.utils import (
    open_tensorstore,
    archive_tensorstore,
    BIG_ENDIAN_DTYPE_DICT,
    DATA_PRECISION,
)

parser = argparse.ArgumentParser(description="Generate data")
parser.add_argument(
    "--dataset",
    type=str,
    metavar="DATASET",
    help="Dataset to use",
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
    env_context, numpy_rng = set_seed(args.seed)

    dataset = importlib.import_module(f"src.datasets.{args.dataset}").Dataset(
        rng=numpy_rng
    )

    os.makedirs(args.out_path, exist_ok=True)

    with env_context:
        for ds_name in ("train", "validation", "test"):
            ds = open_tensorstore(
                os.path.join(args.out_path, ds_name),
                config={
                    "metadata": {
                        "dtype": BIG_ENDIAN_DTYPE_DICT[DATA_PRECISION],
                        "shape": [
                            dataset.COUNT[ds_name],
                            *dataset.SHAPE,
                        ],
                        "chunks": [1, *dataset.SHAPE],
                    },
                    "create": True,
                    "delete_existing": True,
                },
            )
            futures = []
            data = dataset.get_data(split=ds_name)
            futures.append(ds.write(data))
            for f in tqdm(futures, desc=ds_name):
                f.result()
            archive_tensorstore(
                path=os.path.join(args.out_path, ds_name), delete=True
            )
        with open(os.path.join(args.out_path, "mean.pkl"), "wb") as f:
            pickle.dump(dataset.mean, f)
        with open(os.path.join(args.out_path, "std.pkl"), "wb") as f:
            pickle.dump(dataset.std, f)


if __name__ == "__main__":
    main()
