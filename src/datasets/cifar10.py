from operator import itemgetter
from urllib import request
import pickle
import tarfile

from imageio.v3 import imwrite
import numpy as np

from src.data.utils import (
    BIG_ENDIAN_DTYPE_DICT,
    DATA_PRECISION,
)


class Dataset:
    URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    COUNT = {
        "train": 50_000,
        "test": 5_000,
        "validation": 5_000,
    }
    SHAPE = (3, 32, 32)

    def __init__(self, rng=np.random.RandomState()):
        self.train = []
        with tarfile.open(
            fileobj=request.urlopen(self.URL), mode="r|gz"
        ) as tar:
            for member in tar:
                if member.name.endswith("test_batch"):
                    self.validation = pickle.load(
                        tar.extractfile(member), encoding="bytes"
                    )[b"data"]
                elif member.name.find("data_batch_") != -1:
                    self.train.append(
                        pickle.load(tar.extractfile(member), encoding="bytes")[
                            b"data"
                        ]
                    )
        self.train = 2 * (
            (
                np.reshape(
                    np.concatenate(self.train), newshape=(-1, *self.SHAPE)
                ).astype(BIG_ENDIAN_DTYPE_DICT[DATA_PRECISION])
                / 255.0
            )
            - 0.5
        )
        rng.shuffle(self.validation)
        self.test, self.validation = np.vsplit(
            2
            * (
                (
                    np.reshape(
                        self.validation, newshape=(-1, *self.SHAPE)
                    ).astype(BIG_ENDIAN_DTYPE_DICT[DATA_PRECISION])
                    / 255.0
                )
                - 0.5
            ),
            2,
        )
        self.mean = np.zeros(
            self.SHAPE,
            dtype=BIG_ENDIAN_DTYPE_DICT[DATA_PRECISION],
        )
        self.std = np.ones(
            self.SHAPE,
            dtype=BIG_ENDIAN_DTYPE_DICT[DATA_PRECISION],
        )

    def get_data(self, split):
        return getattr(self, split)


shape = Dataset.SHAPE


def save_samples(x, save_path):
    x = ((x + 1.0) * (255.0 / 2)).astype(np.uint8)

    n_samples = x.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1

    nh, nw = rows, n_samples // rows
    x = x.transpose(0, 2, 3, 1)
    h, w = x[0].shape[:2]
    img = np.zeros((h * nh, w * nw, 3), dtype=np.uint8)
    for n, x in enumerate(x):
        j = n // nw
        i = n % nw
        img[j * h : j * h + h, i * w : i * w + w] = x

    imwrite(save_path, img)
