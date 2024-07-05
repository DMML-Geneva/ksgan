"""Pinwheel data simulator.

Based on the repository of
Will Grathwohl*, Ricky T. Q. Chen*, Jesse Bettencourt, Ilya Sutskever, David Duvenaud. "FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models." International Conference on Learning Representations (2019).

https://github.com/rtqichen/ffjord/blob/994864ad0517db3549717c25170f9b71e96788b1/lib/toy_data.py
"""

import numpy as np


class Forward:
    def __init__(
        self,
        radial_std=0.3,
        tangential_std=0.1,
        num_classes=5,
        rate=0.25,
    ):
        self.radial_std = radial_std
        self.tangential_std = tangential_std
        self.num_classes = num_classes
        self.rate = rate

    def __call__(self, n, rng=np.random.RandomState(), **kwargs):
        num_per_class = n // self.num_classes

        rads = np.linspace(0, 2 * np.pi, self.num_classes, endpoint=False)

        features = rng.randn(
            self.num_classes * num_per_class + n % self.num_classes, 2
        ) * np.array([self.radial_std, self.tangential_std])
        features[:, 0] += 1.0
        labels = np.concatenate(
            (
                np.repeat(np.arange(self.num_classes), num_per_class),
                rng.randint(
                    low=0, high=self.num_classes, size=n % self.num_classes
                ),
            )
        )

        angles = rads[labels] + self.rate * np.exp(features[:, 0])
        rotations = np.stack(
            [np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)]
        )
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        return 2 * rng.permutation(
            np.einsum("ti,tij->tj", features, rotations)
        )


shape = (2,)
