"""Circles data simulator.

Based on the repository of
Will Grathwohl*, Ricky T. Q. Chen*, Jesse Bettencourt, Ilya Sutskever, David Duvenaud. "FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models." International Conference on Learning Representations (2019).

https://github.com/rtqichen/ffjord/blob/994864ad0517db3549717c25170f9b71e96788b1/lib/toy_data.py
"""

import numpy as np
from sklearn.datasets import make_circles


class Forward:
    def __init__(self, noise=0.08, factor=0.5):
        self.noise = noise
        self.factor = factor

    def __call__(self, n, rng=np.random.RandomState(), **kwargs):
        return (
            make_circles(
                n_samples=n,
                noise=self.noise,
                factor=self.factor,
                random_state=rng,
            )[0]
            * 3
        )


shape = (2,)
