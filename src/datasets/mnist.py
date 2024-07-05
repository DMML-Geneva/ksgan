import gzip
from operator import itemgetter
from urllib import request
import pickle

from imageio.v3 import imwrite
import numpy as np

from src.data.utils import (
    BIG_ENDIAN_DTYPE_DICT,
    DATA_PRECISION,
)


class Dataset:
    URL = "http://www-labs.iro.umontreal.ca/~lisa/deep/data/mnist/mnist_py3k.pkl.gz"
    COUNT = {
        "train": 50_000,
        "test": 10_000,
        "validation": 10_000,
    }
    SHAPE = (28, 28)

    def __init__(self, rng=np.random.RandomState()):
        with gzip.open(request.urlopen(self.URL)) as f:
            self.train, self.validation, self.test = map(
                lambda x: np.reshape(x, newshape=(-1, *self.SHAPE)).astype(
                    BIG_ENDIAN_DTYPE_DICT[DATA_PRECISION]
                ),
                map(itemgetter(0), pickle.load(f)),
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
    x = (255.99 * x).astype("uint8")

    n_samples = x.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1

    nh, nw = rows, n_samples // rows
    if x.ndim == 3:
        h, w = x[0].shape
        img = np.zeros((h * nh, w * nw))
    elif x.shape[1] == 1:
        x = x[:, 0, :, :]
        h, w = x[0].shape
        img = np.zeros((h * nh, w * nw))
    else:
        x = x.transpose(0, 2, 3, 1)
        h, w = x[0].shape[:2]
        img = np.zeros((h * nh, w * nw, 3))
    for n, x in enumerate(x):
        j = n // nw
        i = n % nw
        img[j * h : j * h + h, i * w : i * w + w] = x

    imwrite(save_path, img.astype(np.uint8))
