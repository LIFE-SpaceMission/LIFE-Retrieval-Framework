"""
Unit tests for priors_timmy.py.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import pytest
from pyretlife.retrieval.priors import assign_priors, uniform_prior, gaussian_prior, log_uniform_prior, \
    upper_log_uniform_prior, log_gaussian_prior


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prior_kind",
    [
        "uniform",
        "log-uniform",
        "upper-log-uniform",
        "gaussian",
        "log-gaussian",
        "fourth-uniform",
        # "custom",
        "invalid"
    ],
)
def test__assign_priors(prior_kind: str):
    test_dict = {"parameter": {"prior": {"kind": prior_kind}}}
    if prior_kind is not "invalid":
        updated_dict = assign_priors(test_dict)

        assert "function" in updated_dict['parameter']['prior'].keys()
        assert callable(updated_dict['parameter']['prior']['function']) #returns True if it is a function or a class (it's callable)
    else:
        with pytest.raises(ValueError) as E:
            assign_priors(test_dict)
        assert "does not have a valid prior" in str(E)

def test__uniform_prior():
    prior_specs={'upper':2, 'lower':-2}
    assert uniform_prior(0, prior_specs) == -2
    assert uniform_prior(1, prior_specs) == 2

def test__gaussian_prior():
    prior_specs = {'mean': 1, 'sigma': 0.5}
    assert np.isclose(gaussian_prior(0.5, prior_specs), 1)
    # assert np.isclose(gaussian_prior(0.42, prior_specs),0.8990532604290745)
    # assert np.isinf(gaussian_prior(0, prior_specs))
    # assert np.isinf(gaussian_prior(1, prior_specs))

def test__log_uniform_prior():
    r=0.5
    prior_specs={'log_upper':2, 'log_lower':-2}
    assert np.isclose(log_uniform_prior(r,prior_specs),1)

def test__upper_log_uniform_prior():
    r=0.5
    prior_specs={'log_upper':2, 'log_lower':-2}
    assert np.isclose(upper_log_uniform_prior(r,prior_specs),0)

def test__log_gaussian_prior():
    r=0.5
    prior_specs={'log_mean':1, 'log_sigma':0.5}
    assert np.isclose(log_gaussian_prior(r,prior_specs),10) #10**1 is 10