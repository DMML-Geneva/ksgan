import numpy as np


class Forward:
    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, n, rng=np.random.RandomState(), **kwargs):
        return rng.normal(size=(n, 2), scale=self.scale)


shape = (2,)
