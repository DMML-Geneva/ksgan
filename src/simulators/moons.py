"""Two moons data simulator.

Based on the repository of
Will Grathwohl*, Ricky T. Q. Chen*, Jesse Bettencourt, Ilya Sutskever, David Duvenaud. "FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models." International Conference on Learning Representations (2019).

https://github.com/rtqichen/ffjord/blob/994864ad0517db3549717c25170f9b71e96788b1/lib/toy_data.py
"""

import numpy as np
from sklearn.datasets import make_moons


class Forward:
    def __init__(self, noise=0.1):
        self.noise = noise

    def __call__(self, n, rng=np.random.RandomState(), **kwargs):
        return make_moons(n_samples=n, noise=self.noise, random_state=rng)[
            0
        ] * 2 + np.array([-1, -0.2])


shape = (2,)
