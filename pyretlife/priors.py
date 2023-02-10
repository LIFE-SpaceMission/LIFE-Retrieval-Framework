"""
Define different types of priors.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Protocol

from scipy import stats

import numpy as np


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

class Prior(Protocol):
    """
    General interface for a prior.

    This is a protocol class, which means that it is not meant to be
    instantiated directly. Instead, it is meant to be subclassed by
    other classes that implement / overwrite the methods defined in
    this class. It is useful for type checking.
    """

    def transform(self, x: float) -> float:
        """
        Transform a cube `x` (in the range [0, 1]) from the sampler
        (PyMultiNest, UltraNest, ...) to a sample from the prior.

        In general, this function needs to implement the inverse CDF
        of the prior distribution.

        Args:
            x: Cube to transform. This is a float in the range [0, 1].

        Returns:
            Transformed cube, that is, a sample from the prior.
        """
        raise NotImplementedError

    def sample(self) -> float:
        """
        Draw a sample from the prior.

        This is a convenience function that first draws a cube from
        Uniform(0, 1) and then applies `transform()` to it.

        Returns:
            Sample from the prior.
        """
        return self.transform(np.random.uniform())


class UniformPrior(Prior):
    """
    Uniform prior.
    """

    def __init__(self, lower: float, upper: float) -> None:
        self.lower = lower
        self.upper = upper
        self.distribution = stats.uniform(lower, upper - lower)

    def transform(self, x: float) -> float:
        return self.distribution.ppf(x)


class GaussianPrior(Prior):
    """
    Gaussian prior.
    """

    def __init__(self, mean: float, std: float) -> None:
        self.mean = mean
        self.std = std
        self.distribution = stats.norm(mean, std)

    def transform(self, x: float) -> float:
        return self.distribution.ppf(x)
