"""
Unit tests for priors.py.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np

from pyretlife.priors import GaussianPrior, UniformPrior


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

def test__uniform_prior() -> None:
    """
    Test `UniformPrior`.
    """

    # Set the random seed
    np.random.seed(42)

    # Create a uniform prior
    uniform_prior = UniformPrior(lower=23, upper=42)

    # Case 1: Transform a given cube
    cube = np.random.uniform(0, 1, size=10_000)
    samples = np.array([uniform_prior.transform(x) for x in cube])
    assert np.all(samples >= 23)
    assert np.all(samples <= 42)

    # Case 2: Use .sample() to draw samples from the prior
    samples = np.array([uniform_prior.sample() for _ in range(10_000)])
    assert np.all(samples >= 23)
    assert np.all(samples <= 42)


def test__gaussian_prior() -> None:
    """
    Test `GaussianPrior`.
    """

    # Set the random seed
    np.random.seed(42)

    # Create a (non-standard) Gaussian prior: N(1, 0.1)
    gaussian_prior = GaussianPrior(mean=1, std=0.1)

    # Case 1: Transform a given cube
    cube = np.random.uniform(0, 1, size=10_000)
    samples = np.array([gaussian_prior.transform(x) for x in cube])
    assert np.isclose(np.mean(samples), 1, atol=0.01)

    # Case 2: Use .sample() to draw samples from the prior
    samples = np.array([gaussian_prior.sample() for _ in range(10_000)])
    assert np.isclose(np.mean(samples), 1, atol=0.01)
