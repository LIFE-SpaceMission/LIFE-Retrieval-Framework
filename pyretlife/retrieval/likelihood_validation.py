import numpy as np
from numpy import ndarray


def validate_pt_profile(
    settings: dict, temp_vars: dict, phys_vars: dict
) -> bool:
    # Check: Return -inf if pressures for madhuseager models are not
    # monotonically increasing.
    result = False
    if settings["parameterization"] == "madhuseager":
        if (
            not settings["log_top_pressure"]
            < temp_vars["log_P1"]
            < temp_vars["log_P2"]
            < temp_vars["log_P3"]
            < phys_vars["log_P0"]
        ):
            result = True
    if settings["parameterization"] == "mod_madhuseager":
        if (
            not settings["log_top_pressure"]
            < temp_vars["log_P1"]
            < temp_vars["log_P2"]
            < phys_vars["log_P0"]
        ):
            result = True

    # Check: Return -inf if parameters for the Guillot model are bad
    if settings["parameterization"] == "guillot":
        if temp_vars["alpha"] < -1:
            result = True

    return result


def validate_sum_of_cube(cube: list) -> bool:
    if np.isfinite(cube).all():
        return True


def validate_positive_temperatures(temp: ndarray) -> bool:
    if any((temp < 0).tolist()):
        return True


def validate_sum_of_abundances(chem_vars: dict) -> bool:
    if sum(chem_vars.values()) > 1:
        return True


def validate_spectrum_goodness(flux: ndarray) -> bool:
    if np.sum(np.isnan(flux)) > 0:
        return True
