"""
Unit tests for priors_timmy.py.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import pytest

from pyretlife.retrieval.priors_timmy import (
    get_prior_from_string,
    GaussianPrior,
    UniformPrior,
)


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------


def test__get_prior_from_string() -> None:
    """
    Test `get_prior_from_string`.
    """

    # Case 1: Invalid prior string
    for invalid_string in [
        "Z -2 2 T 0",  # Invalid prior type
        "G -2 2 T invalid",  # Invalid ground truth
        "U -2 2",  # Missing ground truth
        "U -2 2 T",  # Missing ground truth
        "U -2 2 T 0 1",  # Too many arguments
    ]:
        with pytest.raises(ValueError) as value_error:
            get_prior_from_string(invalid_string)
        assert "Invalid prior string:" in str(value_error.value)

    # Case 2: Uniform prior
    prior = get_prior_from_string("U -2 2 T 0")
    assert isinstance(prior, UniformPrior)
    assert prior.lower == -2
    assert prior.upper == 2
    assert prior.ground_truth == 0

    # Case 3: Gaussian prior
    prior = get_prior_from_string("G -2 2 T None")
    assert isinstance(prior, GaussianPrior)
    assert prior.mean == -2
    assert prior.std == 2
    assert prior.ground_truth is None


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

    # Case 3: Ground truth is outside the prior range
    with pytest.raises(ValueError) as value_error:
        UniformPrior(lower=23, upper=42, ground_truth=100)
    assert "The ground truth is outside the prior range!" in str(value_error)


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
