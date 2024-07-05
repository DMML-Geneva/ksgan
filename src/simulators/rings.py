"""Rings data simulator.

Based on the repository of
Will Grathwohl*, Ricky T. Q. Chen*, Jesse Bettencourt, Ilya Sutskever, David Duvenaud. "FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models." International Conference on Learning Representations (2019).

https://github.com/rtqichen/ffjord/blob/994864ad0517db3549717c25170f9b71e96788b1/lib/toy_data.py
"""

import numpy as np
from sklearn.utils import shuffle as util_shuffle


class Forward:
    def __init__(self, noise=0.08):
        self.noise = noise

    def __call__(self, n, rng=np.random.RandomState(), **kwargs):
        n_samples4 = n_samples3 = n_samples2 = n // 4
        n_samples1 = n - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, we set endpoint=False
        linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25

        X = (
            np.vstack(
                [
                    np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
                    np.hstack([circ4_y, circ3_y, circ2_y, circ1_y]),
                ]
            ).T
            * 3.0
        )
        X = util_shuffle(X, random_state=rng)

        # Add noise
        X = X + rng.normal(scale=self.noise, size=X.shape)
        return X


shape = (2,)
