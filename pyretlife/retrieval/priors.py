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
        prior_kind = dictionary[parameter]['prior']['kind']
        if prior_kind == 'uniform':
            dictionary[parameter]['prior']['function'] = uniform_prior
        elif prior_kind == 'log-uniform':
            dictionary[parameter]['prior']['function'] = log_uniform_prior
        elif prior_kind == 'gaussian':
            dictionary[parameter]['prior']['function'] = gaussian_prior
        elif prior_kind == 'log-gaussian':
            dictionary[parameter]['prior']['function'] = log_gaussian_prior
        else:
            invalid_prior(parameter)
        # TODO Implement ULU, FU, I

    return dictionary
    # key = list(self.params.keys()).index(par)
    #
    #
    # switcher = {
    #     "U": priors.uniform_prior(ccube[key], prior[0], prior[1]),
    #     "LU": np.power(
    #         10.0, priors.uniform_prior(ccube[key], prior[0], prior[1])
    #     ),
    #     "ULU": 1
    #            - np.power(
    #         10.0, priors.uniform_prior(ccube[key], prior[0], prior[1])
    #     ),
    #     "FU": np.power(
    #         priors.uniform_prior(ccube[key], prior[0], prior[1]), 4
    #     ),
    #     "G": priors.gaussian_prior(
    #         ccube[key], prior[0], prior[1]
    #     ).astype("float64"),
    #     "LG": np.power(
    #         10.0, priors.gaussian_prior(ccube[key], prior[0], prior[1])
    #     ),
    # }
    #
    # pr = switcher.get(self.params[par]["prior_type"])
    # if pr is None:
    #     priors.invalid_prior(par)
    # ccube[key] = pr


def uniform_prior(r, prior_specs):
    """
    Scales a random number generated in a uniform prior between 0
    and 1 to the respective value corresponding to a uniform prior
    ranged between x1 and x2.

    Called by: FillPriors

    Parameters
    ----------
    r : A random float generated from the uniform prior between [0, 1].
    x1 : The new lower boundary (float).
    x2 : The new upper boundary (floa).

    Returns
    -------
    A random number generated from a uniform prior between [x1, x2].
    """
    x1=prior_specs['lower']
    x2=prior_specs['upper']
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
    mu : The mean of the gaussian distribution (float).
    sigma : The standard deviation of the gaussian distribution (float).

    Returns
    -------
    A random float generated from a gaussian prior G(mu,sigma).
    """
    # if r < 1e-16 or (1.0 - r) < 1e-16:
    #    return -1.0e32
    # else:
    # return -((r - mu) / sigma)**2 / 2
    mu=prior_specs['mean']
    sigma=prior_specs['sigma']
    return stat.norm.ppf(r) * sigma + mu


def log_uniform_prior(r, prior_specs):
    prior_logspace={}
    prior_logspace['lower']=prior_specs['log_lower']
    prior_logspace['upper']=prior_specs['log_upper']
    return np.power(10, uniform_prior(r, prior_logspace))


def log_gaussian_prior(r, prior_specs):
    prior_logspace={}
    prior_logspace['mean']=prior_specs['log_mean']
    prior_logspace['sigma']=prior_specs['log_sigma']
    return np.power(10, gaussian_prior(r,prior_logspace))


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

# def FillPriors(r, str):
#     """
#     For every parameter given in input, it selects the correct prior
#     to convert the random number between 0 and 1 to the corresponding
#     number within the boundaries of the prior.
#
#     Called by: Priors
#
#     Parameters
#     ----------
#     r : A random float generated from the uniform prior between [0, 1].
#     str : The name of the parameter as stored in the global params list.
#     Returns
#     -------
#     A random float within the correct prior.
#     """
#
#     if str == 'a_0':
#         return UniformPrior(r, 0, 500)
#     elif str == 'a_1':
#         return UniformPrior(r, 0, 1000)
#     elif str == 'a_2':
#         return UniformPrior(r, 0, 500)
#     elif str == 'a_3':
#         return UniformPrior(r, 0,100)
#     elif str == 'a_4':
#         return UniformPrior(r, 0.7,1.8)**4 #np.exp(UniformPrior(r, -2, 3))
#     elif str == 'log_delta':
#         return UniformPrior(r, -5.5, 2.5)  # GaussianPrior(r,-5.5,2.5)
#     elif str == 'log_gamma':
#         return UniformPrior(r, 0, 2)  # GaussianPrior(r,0,2)
#     elif str == 't_int':
#         return UniformPrior(r, 0, 1500)
#     elif str == 't_equ':
#         return UniformPrior(r, 0, 1400)
#     elif str == 'log_p_trans':
#         return UniformPrior(r, -3, 3)  # GaussianPrior(r,-3,3)
#     elif str == 'alpha':
#         return UniformPrior(r, 0.25, 0.4)  # GaussianPrior(r,0.25,0.4)
#     elif str == 'log_g':
#         return UniformPrior(r, 0.4, 4.7)
#     elif str == 'g':
#         return UniformPrior(r, 2, 2e3)
#     elif str == 'log_P0':
#         return UniformPrior(r, -2, 2)
#     elif str == 'P0':
#         return UniformPrior(r, 0.5,2)**4
#     elif str == 'R_pl':
#         return GaussianPrior(
#             r,
#             float(g.knowns['R_pl_initial']),
#             float(g.knowns['dR_pl_initial'])
#         )
#     elif str == 'M_pl':
#         return 10**GaussianPrior(
#             r,
#             float(g.knowns['log_Mmedian_pl']),
#             float(g.knowns['d_log_Mmedian_pl']),
#         )
#     elif str == 'd_syst':
#             return UniformPrior(r, 0, 100000)
#     elif str in g.opacities: #All the folders in the opacity lines folder
#         return UniformPrior(r, -15, 0) #    UniformPrior(r, 0, 1)
#     elif str in ['N2']:
#         return UniformPrior(r, -15, 0)
#     elif str in ['O2']:
#         return UniformPrior(r, -15, 0)
#     elif str == 'T0':
#         return UniformPrior(r, 0, 1000)
#     else:
#         print('Missing this prior!', str)
#         return -np.inf
