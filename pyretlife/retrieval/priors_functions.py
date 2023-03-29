"""
PRIORS.py

Here, the unity cube is converted to prior cube by using either an
uninformative, uniform prior, or a Gaussian prior.
The variable cube has length equal to the number of parameters to be
retrieved; the indexing follows the order of the params global list.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import scipy.stats as stat
import numpy as np


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------


def assign_priors(dictionary: dict) -> dict:
    for parameter in dictionary.keys():
        prior_kind = dictionary[parameter]["prior"]["kind"]
        if prior_kind == "uniform":
            dictionary[parameter]["prior"]["function"] = uniform_prior
        elif prior_kind == "log-uniform":
            dictionary[parameter]["prior"]["function"] = log_uniform_prior
        elif prior_kind == "gaussian":
            dictionary[parameter]["prior"]["function"] = gaussian_prior
        elif prior_kind == "log-gaussian":
            dictionary[parameter]["prior"]["function"] = log_gaussian_prior
        else:
            invalid_prior(parameter)
        # TODO Implement ULU, FU, I

    return dictionary


def uniform_prior(r, prior_specs):
    """
    Scales a random number generated in a uniform prior between 0
    and 1 to the respective value corresponding to a uniform prior
    ranged between x1 and x2.

    Called by: FillPriors

    Parameters
    ----------
    r : A random float generated from the uniform prior between [0, 1].
    prior_specs:
    Returns
    -------
    A random number generated from a uniform prior between [x1, x2].
    """
    x1 = prior_specs["lower"]
    x2 = prior_specs["upper"]
    return x1 + r * (x2 - x1)


def gaussian_prior(r, prior_specs):
    """
    Scales a random number generated in a uniform prior between 0
    and 1 to the respective value corresponding to a gaussian prior
    centered at mu and of standard deviation sigma.

    Called by: FillPriors

    Parameters
    ----------
    r : A random float generated from the uniform prior between [0, 1].
    prior_specs:
    Returns
    -------
    A random float generated from a gaussian prior G(mu,sigma).
    """
    # if r < 1e-16 or (1.0 - r) < 1e-16:
    #    return -1.0e32
    # else:
    # return -((r - mu) / sigma)**2 / 2
    mu = prior_specs["mean"]
    sigma = prior_specs["sigma"]
    return stat.norm.ppf(r) * sigma + mu


def log_uniform_prior(r, prior_specs):
    prior_logspace = {"lower": prior_specs["log_lower"], "upper": prior_specs["log_upper"]}
    return np.power(10, uniform_prior(r, prior_logspace))


def log_gaussian_prior(r, prior_specs):
    prior_logspace = {"mean": prior_specs["log_mean"], "sigma": prior_specs["log_sigma"]}
    return np.power(10, gaussian_prior(r, prior_logspace))


def invalid_prior(par):
    # Note: If you are exiting the run with an error, you should not use
    # `sys.exit(0)` because that will produce a return code of 0, which
    # usually means "success" or "no errors". In any case, raising a
    # `ValueError` is probably the best way to go here, because it will
    # give the user a clear error message and traceback, and will also
    # exit the run with a non-zero return code.

    raise ValueError(
        f"{par} does not have a valid prior. Please! choose a valid prior! "
        f"Exiting the run..."
    )
