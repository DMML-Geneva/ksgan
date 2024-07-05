import argparse
import importlib
import os
import yaml

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
    "--simulator",
    type=str,
    metavar="SIM",
    help="Simulator to use",
)
parser.add_argument(
    "--config",
    type=str,
    metavar="CONFIG",
    help="Configuration file for generation",
)
parser.add_argument(
    "--out-path",
    type=str,
    metavar="OUTPATH",
    help="Path to write data",
)
parser.add_argument(
    "--bs",
    type=int,
    metavar="SEED",
    help="Batch size",
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
        cfg = yaml.load(fin, Loader=yaml.SafeLoader)

    env_context, numpy_rng = set_seed(args.seed)
    simulator = importlib.import_module(f"src.simulators.{args.simulator}")
    forward = simulator.Forward()
    bs = args.bs

    os.makedirs(args.out_path, exist_ok=True)

    with env_context:
        for ds_name, n in zip(
            ("train", "validation", "test"),
            (max(cfg["simulation-budgets"]), cfg["validation"], cfg["test"]),
        ):
            ds = open_tensorstore(
                os.path.join(args.out_path, ds_name),
                config={
                    "metadata": {
                        "dtype": BIG_ENDIAN_DTYPE_DICT[DATA_PRECISION],
                        "shape": [n, *simulator.shape],
                        "chunks": [1, *simulator.shape],
                    },
                    "create": True,
                    "delete_existing": True,
                },
            )
            futures = []
            if bs != -1 and bs >= n:
                for i in range(n // bs + (n % bs != 0)):
                    _n = min(bs, n - i * bs)
                    data_batch = forward(n=_n, rng=numpy_rng).astype(
                        BIG_ENDIAN_DTYPE_DICT[DATA_PRECISION]
                    )
                    futures.append(ds[i * bs : i * bs + _n].write(data_batch))
            else:
                data_batch = forward(n=n, rng=numpy_rng).astype(
                    BIG_ENDIAN_DTYPE_DICT[DATA_PRECISION]
                )
                futures.append(ds.write(data_batch))
            for f in tqdm(futures, desc=ds_name):
                f.result()
            archive_tensorstore(
                path=os.path.join(args.out_path, ds_name), delete=True
            )


if __name__ == "__main__":
    main()
