"""Checkerboard data simulator.

Based on the repository of
Will Grathwohl*, Ricky T. Q. Chen*, Jesse Bettencourt, Ilya Sutskever, David Duvenaud. "FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models." International Conference on Learning Representations (2019).

https://github.com/rtqichen/ffjord/blob/994864ad0517db3549717c25170f9b71e96788b1/lib/toy_data.py
"""

import numpy as np


class Forward:
    def __call__(self, n, rng=np.random.RandomState(), **kwargs):
        x1 = rng.rand(n) * 4 - 2
        x2_ = rng.rand(n) - rng.randint(0, 2, n) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        return np.concatenate([x1[:, None], x2[:, None]], 1) * 2


shape = (2,)
