__author__ = "Konrad"
__copyright__ = "Copyright 2022, Konrad"
__maintainer__ = "Björn S. Konrad"
__email__ = "konradb@phys.ethz.ch"
__status__ = "Development"

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

def logistic_function(x, L, k, hm):
    return L / (1 + np.exp(-k * (x - hm)))


def inverse_logistic_function(y, L, k, hm):
    return hm - np.log(L / y - 1) / k
