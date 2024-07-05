"""Line data simulator.

Based on the repository of
Will Grathwohl*, Ricky T. Q. Chen*, Jesse Bettencourt, Ilya Sutskever, David Duvenaud. "FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models." International Conference on Learning Representations (2019).

https://github.com/rtqichen/ffjord/blob/994864ad0517db3549717c25170f9b71e96788b1/lib/toy_data.py
"""

import numpy as np


class Forward:
    def __init__(self, scale=5.0, shift=2.5):
        self.scale = scale
        self.shift = shift

    def __call__(self, n, rng=np.random.RandomState(), **kwargs):
        x = rng.rand(n) * self.scale - self.shift
        y = x
        return np.stack((x, y), 1)


shape = (2,)
