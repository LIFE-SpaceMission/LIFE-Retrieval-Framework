"""
Define different types of priors.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Optional, Protocol

import re

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

    def __init__(
        self,
        lower: float,
        upper: float,
        ground_truth: Optional[float] = None,
    ) -> None:
        # Store the parameters
        self.lower = lower
        self.upper = upper
        self.ground_truth = ground_truth

        # Create the distribution object
        self.distribution = stats.uniform(lower, upper - lower)

        # Check that the ground truth is within the prior range
        if (
            ground_truth is not None
            and self.distribution.pdf(ground_truth) == 0
        ):
            raise ValueError("The ground truth is outside the prior range!")

    def transform(self, x: float) -> float:
        return self.distribution.ppf(x)


class GaussianPrior(Prior):
    """
    Gaussian prior.
    """

    def __init__(
        self,
        mean: float,
        std: float,
        ground_truth: Optional[float] = None,
    ) -> None:
        # Store the parameters
        self.mean = mean
        self.std = std
        self.ground_truth = ground_truth

        # Sanity checks on the parameters
        if std <= 0:
            raise ValueError("The standard deviation must be positive!")

        # Create the distribution object
        self.distribution = stats.norm(mean, std)

        # Note: we don't check that the ground truth is within the prior
        # range because the Gaussian prior is unbounded.

    def transform(self, x: float) -> float:
        return self.distribution.ppf(x)


def get_prior_from_string(string: str) -> Prior:
    """
    Parse a prior string and return the corresponding prior object.

    TODO: Maybe we should not rely on a string to define the prior but
      instead use a nested dictionary or something like that? While it
      might be more verbose, it would be more flexible and easier to
      understand and extend. Something like:

        priors = {
            "some_parameter": {
                "kind": "uniform",
                "parameters": {
                    "lower": 0,
                    "upper": 1,
                },
                "ground_truth": 0.5,
            },
            "another_parameter": {
                ...
            }
        }

    TODO: This would be easy to implement with the YAML format.

    Args:
        string: A string that defines a prior. The string must have the
            following format:
                ``[prior_kind] [parameters] T [ground_truth]``
            where:
                - [prior_kind] is a string that defines the prior kind.
                  It can be one of the following:
                      - "U": Uniform prior
                      - "LU": Log-uniform prior
                      - "ULU": Uniform-log-uniform prior
                      - "FU": Flat-uniform prior
                      - "G": Gaussian prior
                      - "LG": Log-Gaussian prior
                - [parameters] is a string that defines the parameters.
                    It can be one of the following:
                      - For a "U" prior: ``[lower] [upper]``
                      - For a "LU" prior: ``[lower] [upper]``
                      - For a "ULU" prior: ``[lower] [upper]``
                      - For a "FU" prior: ``[lower] [upper]``
                      - For a "G" prior: ``[mean] [std]``
                      - For a "LG" prior: ``[mean] [std]``
                - [ground_truth] is a float that defines the ground
                  truth. It is optional and can be set to "None".

    Returns:
        The corresponding prior object.
    """

    # Define a regular expression to parse the prior string
    pattern = (
        r"^(?P<prior_kind>U|LU|ULU|FU|G|LG])\s"  # Prior kind
        r"(?P<parameters>[\-?\d+\.?\d*\s]+)"  # Parameters
        r"T\s(?P<ground_truth>None|\-?\d+\.?\d*)$"  # Ground truth
    )

    # Match the string against the regular expression
    match = re.match(pattern, string.strip())
    if match is None:
        raise ValueError(f"Invalid prior string: {string}")

    # Extract and parse the different parts of the string
    prior_kind = match.group("prior_kind")
    parameters_string = match.group("parameters")
    ground_truth_string = match.group("ground_truth")

    # Parse the parameters string and convert it to a list of floats
    parameters = [float(p) for p in parameters_string.split()]

    # Parse the ground truth string and convert it to a float
    if ground_truth_string == "None":
        ground_truth = None
    else:
        ground_truth = float(ground_truth_string)

    # Create the prior object
    if prior_kind == "U":
        prior = UniformPrior(*parameters, ground_truth)
    elif prior_kind == "G":
        prior = GaussianPrior(*parameters, ground_truth)
    else:
        raise ValueError(f"Invalid prior kind: {prior_kind}")

    return prior
