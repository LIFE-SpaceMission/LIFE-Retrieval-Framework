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
from typing import Union
from pathlib import Path
from numpy import ndarray


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------


def assign_priors(dictionary: dict) -> dict:
    """
    The assign_priors function takes a dictionary of parameters and assigns the appropriate prior function to each
    parameter. The prior functions are defined in the priors module. The assign_priors function is called by the
    read_parameters function, which reads in the parameters from a .yaml file.


    :param dictionary: Pass the dictionary containing all the parameters and their values
    :return:  A dictionary with the prior function assigned to each parameter
    """
    for parameter in dictionary.keys():
        prior_kind = dictionary[parameter]["prior"]["kind"]
        if prior_kind == "uniform":
            dictionary[parameter]["prior"]["function"] = uniform_prior
        elif prior_kind == "log-uniform":
            dictionary[parameter]["prior"]["function"] = log_uniform_prior
        elif prior_kind == "upper-log-uniform":
            dictionary[parameter]["prior"]["function"] = upper_log_uniform_prior
        elif prior_kind == "gaussian":
            dictionary[parameter]["prior"]["function"] = gaussian_prior
        elif prior_kind == "log-gaussian":
            dictionary[parameter]["prior"]["function"] = log_gaussian_prior
        elif prior_kind == "fourth-uniform":
            dictionary[parameter]["prior"][
                "function"
            ] = fourth_power_uniform_prior
        elif prior_kind == "third-uniform":
            dictionary[parameter]["prior"][
                "function"
            ] = third_power_uniform_prior
        elif prior_kind == "second-uniform":
            dictionary[parameter]["prior"][
                "function"
            ] = second_power_uniform_prior
        elif prior_kind == "custom":
            dictionary[parameter]["prior"]["function"] = custom_prior

            dictionary[parameter]["prior"]["prior_specs"][
                "prior_data"
            ] = read_custom_prior(
                dictionary[parameter]["prior"]["prior_specs"]["prior_path"]
            )
        else:
            raise ValueError(
                f"{parameter} does not have a valid prior. Please! choose a valid prior! "
                f"Exiting the run..."
            )

    return dictionary


def read_custom_prior(path: Union[str, Path]) -> ndarray:
    """
    The read_custom_prior function reads in a custom prior from the path specified by the user.
    The function takes one argument, which is a string or Path object specifying where to find the file containing
    the custom prior. The function returns a ndarray containing the data from the custom prior distribution.

    :param path: Specify that the path parameter can be either a string or a path object
    :return:  A dictionary with the prior function assigned to each parameter
    """
    return np.loadtxt(path)


def uniform_prior(r: float, prior_specs: dict) -> float:
    """
    Scales a random number generated in a uniform prior between 0 and 1 to the respective value corresponding to a
    uniform prior ranged between x1 and x2. This function is called by the assign_priors function.

    A random number generated from a uniform prior between [x1, x2].
    :param r: A random float generated from the uniform prior between [0, 1].
    :param prior_specs: A dictionary of prior_specs containing the keywords "lower" and "upper"
    :return: A random number generated from a uniform prior between [x1, x2].
    """
    x1 = prior_specs["lower"]
    x2 = prior_specs["upper"]
    return x1 + r * (x2 - x1)


def gaussian_prior(r: float, prior_specs: dict) -> float:
    """
    Scales a random number generated in a uniform prior between 0 and 1 to the respective value corresponding to a
    gaussian prior centered at mu and of standard deviation sigma. This function is called by the assign_priors
    function.


    :param r: A random float generated from the uniform prior between [0, 1].
    :param prior_specs: A dictionary of prior_specs containing the keywords "mean" and "sigma"
    :return: A random float generated from a gaussian prior G(mu,sigma).
    """

    mu = prior_specs["mean"]
    sigma = prior_specs["sigma"]
    return stat.norm.ppf(r) * sigma + mu


def log_uniform_prior(r: float, prior_specs: dict) -> float:
    """
    Returns a number like 10^x, where x is generated by scaling a random number generated in a uniform prior between
    0 and 1 to the respective value corresponding to a uniform prior ranged between x1 and x2. This function is
    called by the assign_priors function.

    :param r: A random float generated from the uniform prior between [0, 1].
    :param prior_specs: A dictionary of prior_specs containing the keywords "log_lower" and "log_upper"
    :return: A random number whose exponent in base 10 is generated from a uniform prior between [x1, x2].
    """
    prior_logspace = {
        "lower": prior_specs["log_lower"],
        "upper": prior_specs["log_upper"],
    }
    return np.power(10, uniform_prior(r, prior_logspace))


def upper_log_uniform_prior(r: float, prior_specs: dict) -> float:
    """
    Returns a number like (1-10^x), where x is generated by scaling a random number generated in a uniform prior
    between 0 and 1 to the respective value corresponding to a uniform prior ranged between x1 and x2. This function
    is called by the assign_priors function.

    :param r: A random float generated from the uniform prior between [0, 1].
    :param prior_specs: A dictionary of prior_specs containing the keywords "log_lower" and "log_upper"
    :return: A random number like (1-10^x) whose exponent in base 10 is generated from a uniform prior between [x1, x2].
    """

    prior_logspace = {
        "lower": prior_specs["log_lower"],
        "upper": prior_specs["log_upper"],
    }
    return 1 - np.power(10, uniform_prior(r, prior_logspace))


def log_gaussian_prior(r: float, prior_specs: dict) -> float:
    """
    Returns a number like 10^x, where x is generated by scaling a random number generated in a uniform prior between
    0 and 1 to the respective value corresponding to a gaussian prior ranged between x1 and x2. This function is
    called by the assign_priors function.

    :param r: A random float generated from the uniform prior between [0, 1].
    :param prior_specs: A dictionary of prior_specs containing the keywords "log_mean" and "log_sigma"
    :return: A random number whose exponent in base 10 is generated from a gaussian prior G(mu,sigma).
    """

    prior_logspace = {
        "mean": prior_specs["log_mean"],
        "sigma": prior_specs["log_sigma"],
    }
    return np.power(10, gaussian_prior(r, prior_logspace))


def fourth_power_uniform_prior(r: float, prior_specs: dict) -> float:
    """
    The fourth_power_uniform_prior function takes in a random number and prior specifications. It then returns the
    fourth power of a uniform distribution with lower and upper bounds specified by the prior_specs.

    :param r: A random float generated from the uniform prior between [0, 1].
    :param prior_specs: A dictionary of prior_specs containing the keywords "fourth_lower" and "fourth_upper"
    :return: A random number like x^4 where x is generated from a uniform prior between [x1, x2]
    """
    prior_fourth = {
        "lower": prior_specs["fourth_lower"],
        "upper": prior_specs["fourth_upper"],
    }
    return np.sign(uniform_prior(r, prior_fourth))*np.power(uniform_prior(r, prior_fourth), 4)


def third_power_uniform_prior(r: float, prior_specs: dict) -> float:
    """
    The fourth_power_uniform_prior function takes in a random number and prior specifications. It then returns the
    fourth power of a uniform distribution with lower and upper bounds specified by the prior_specs.

    :param r: A random float generated from the uniform prior between [0, 1].
    :param prior_specs: A dictionary of prior_specs containing the keywords "fourth_lower" and "fourth_upper"
    :return: A random number like x^4 where x is generated from a uniform prior between [x1, x2]
    """
    prior_fourth = {
        "lower": prior_specs["third_lower"],
        "upper": prior_specs["third_upper"],
    }
    return np.power(uniform_prior(r, prior_fourth), 3)


def second_power_uniform_prior(r: float, prior_specs: dict) -> float:
    """
    The fourth_power_uniform_prior function takes in a random number and prior specifications. It then returns the
    fourth power of a uniform distribution with lower and upper bounds specified by the prior_specs.

    :param r: A random float generated from the uniform prior between [0, 1].
    :param prior_specs: A dictionary of prior_specs containing the keywords "fourth_lower" and "fourth_upper"
    :return: A random number like x^4 where x is generated from a uniform prior between [x1, x2]
    """
    prior_fourth = {
        "lower": prior_specs["second_lower"],
        "upper": prior_specs["second_upper"],
    }
    return np.sign(uniform_prior(r, prior_fourth))*np.power(uniform_prior(r, prior_fourth), 2)


def custom_prior(r: float, prior_specs: dict) -> float:
    """
    The custom_prior function takes in a random number r and the prior_specs dictionary.
    It then returns the quantile of the data corresponding to that random number.
    This is useful for when you want to use your own data as a prior.

    :param r: A random float generated from the uniform prior between [0, 1].
    :param prior_specs: A dictionary of prior_specs containing the keywords "data"
    :return: The number corresponding to the r-th quantile of the custom prior
    """
    return np.quantile(prior_specs["data"], r, axis=0)


def invalid_prior(par: str):
    """
    The invalid_prior function is used to raise an error if the user provides
    an invalid prior. It takes a single argument, par, which is the name of
    the parameter for which an invalid prior was provided. The function then
    raises a ValueError with a message explaining that the prior provided for
    par is invalid.

    :param par: A string containing the name of the parameter
    :raise ValueError:
    """

    # Note: If you are exiting the run with an error, you should not use
    # `sys.exit(0)` because that will produce a return code of 0, which
    # usually means "success" or "no errors". In any case, raising a
    # `ValueError` is probably the best way to go here, because it will
    # give the user a clear error message and traceback, and will also
    # exit the run with a non-zero return code.
    # TODO: Add functionality for invalid priors

    raise ValueError("The prior provided for " + str(par) + "is invalid.")
