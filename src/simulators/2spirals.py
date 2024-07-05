"""2 spirals data simulator.

Based on the repository of
Will Grathwohl*, Ricky T. Q. Chen*, Jesse Bettencourt, Ilya Sutskever, David Duvenaud. "FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models." International Conference on Learning Representations (2019).

https://github.com/rtqichen/ffjord/blob/994864ad0517db3549717c25170f9b71e96788b1/lib/toy_data.py
"""

import numpy as np


class Forward:
    def __init__(self, noise=0.1):
        self.noise = noise

    def __call__(self, n, rng=np.random.RandomState(), **kwargs):
        n_parameter = np.sqrt(rng.rand(n // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n_parameter) * n_parameter + rng.rand(n // 2, 1) * 0.5
        d1y = np.sin(n_parameter) * n_parameter + rng.rand(n // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += rng.randn(*x.shape) * self.noise
        return x


shape = (2,)
