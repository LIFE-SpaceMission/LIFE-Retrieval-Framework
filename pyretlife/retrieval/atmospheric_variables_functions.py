from molmass import Formula
import astropy.constants
import numpy as np
from typing import Tuple
from numpy import ndarray


def calculate_gravity(phys_vars: dict) -> dict:
    """
    Function to check if the surface gravity is provided or can
    be calculated from the provided parameters.
    """

    # Calculate the surface gravity g given M_Pl and R_pl or log_g.
    # If in knowns already, skip
    if "g" not in phys_vars.keys():
        if "log_g" in phys_vars.keys():
            phys_vars["g"] = 10 ** phys_vars["log_g"]
        else:
            phys_vars["g"] = (
                astropy.constants.G.cgs.value * phys_vars["M_pl"] / (phys_vars["R_pl"]) ** 2
            )
    return phys_vars


def calculate_polynomial_profile(P: ndarray, temp_vars: dict) -> ndarray:
    return np.array(
        np.polyval(
            np.array(
                [
                    temp_vars["a_" + str(len(temp_vars) - 1 - i)]
                    for i in range(len(temp_vars))
                ]
            ),
            np.log10(P),
        )
    )


# TODO typeset vae_pt
def calculate_vae_profile(P: ndarray, vae_pt, temp_vars: dict) -> ndarray:
    return vae_pt.get_temperatures(
        z=np.array(
            [temp_vars["z_" + str(i + 1)] for i in range(len(temp_vars))]
        ),
        log_p=np.log10(P),
    )


def calculate_guillot_profile(P: ndarray, pRT, temp_vars: dict) -> ndarray:
    return pRT.nat_cst.guillot_modif(
        P,
        1e1 ** temp_vars["log_delta"],
        1e1 ** temp_vars["log_gamma"],
        temp_vars["t_int"],
        temp_vars["t_equ"],
        1e1 ** temp_vars["log_p_trans"],
        temp_vars["alpha"],
    )


def calculate_isothermal_profile(P: ndarray, temp_vars: dict) -> ndarray:
    return temp_vars["T_eq"] * np.ones_like(P)


def calculate_madhuseager_profile(P: ndarray, temp_vars: dict) -> ndarray:
    import scipy.ndimage as sci

    beta1 = 0.5
    beta2 = 0.5

    def T_P(P_m, P_i, T_i, alpha, beta):
        return (np.log(P_m / P_i) / alpha) ** (1 / beta) + T_i

    P0, P1, P2, P3 = (
        10 ** P[0],  # log_top_pressure by definition
        10 ** temp_vars["log_P1"],
        10 ** temp_vars["log_P2"],
        10 ** temp_vars["log_P3"],
    )

    T = np.zeros_like(P)

    T2 = (
        temp_vars["T0"]
        + (np.log(P1 / P0) / temp_vars["alpha1"]) ** (1 / beta1)
        - (np.log(P1 / P2) / temp_vars["alpha2"]) ** (1 / beta2)
    )
    T3 = T_P(P3, P2, T2, temp_vars["alpha2"], beta2)

    for i in range(np.size(P)):
        if P[i] < P1:
            T[i] = T_P(
                P[i],
                P0,
                temp_vars["T0"],
                temp_vars["alpha1"],
                beta1,
            )
        elif P1 < P[i] < P3:
            T[i] = T_P(P[i], P2, T2, temp_vars["alpha2"], beta2)
        elif P[i] > P3:
            T[i] = T3

    T = sci.gaussian_filter1d(T, 20.0, mode="nearest")
    return T

def calculate_mod_madhuseager_profile(P: ndarray, temp_vars: dict) -> ndarray:
    beta1 = 0.5
    beta2 = 0.5

    def T_P(P_m, P_i, T_i, alpha, beta):
        return (np.log(P_m / P_i) / alpha) ** (1 / beta) + T_i

    P0, P1, P2 = (
        10 ** P[0],
        10 ** temp_vars["log_P1"],
        10 ** temp_vars["log_P2"],
    )

    T = np.zeros_like(P)

    T2 = (
        temp_vars["T0"]
        + (np.log(P1 / P0) / temp_vars["alpha1"]) ** (1 / beta1)
        - (np.log(P1 / P2) / temp_vars["alpha2"]) ** (1 / beta2)
    )

    for i in range(np.size(P)):
        if P[i] < P1:
            T[i] = T_P(
                P[i],
                P0,
                temp_vars["T0"],
                temp_vars["alpha1"],
                beta1,
            )
        elif P1 < P[i]:
            T[i] = T_P(P[i], P2, T2, temp_vars["alpha2"], beta2)
    return T


def calculate_abundances(chem_vars: dict, press: ndarray) -> dict:
    abundances = {}

    for name in chem_vars.keys():
        abundances[name] = np.ones_like(press) * chem_vars[name]
    return abundances


def assign_cloud_parameters(
    abundances: dict, cloud_vars: dict, press: ndarray
) -> Tuple[dict, dict, dict, int]:
    # TODO test that it works
    cloud_radii = {}
    cloud_lnorm = 0
    for cloud in cloud_vars.keys():
        abundances[cloud.split("_")[0]][
            np.where(
                (press < cloud_vars[cloud]["bottom_pressure"])
                & (press > cloud_vars[cloud]["top_pressure"])
            )
        ] = cloud_vars[cloud]["abundance"]
        cloud_vars[cloud]["bottom_pressure"] = (
            cloud_vars[cloud]["top_pressure"] + cloud_vars[cloud]["thickness"]
        )
        cloud_radii[cloud.split("_")[0]] = cloud_vars[cloud]["particle_radius"]
        # TODO is it the same for all clouds then?
        cloud_lnorm = cloud_vars[cloud]["sigma_lnorm"]
    return abundances, cloud_vars, cloud_radii, cloud_lnorm


def calc_MMW(abundances: dict, settings: dict, inert: ndarray) -> ndarray:
    mmw = np.zeros_like(range(settings['n_layers']),dtype=float)
    for layer in range(settings['n_layers']):
        for key in abundances.keys():
            mmw[layer] = mmw[layer]+ abundances[key][layer] * getMM(key)

        if "mmw_inert" in settings.keys():
             mmw[layer] =mmw[layer]+ inert[layer] * float(settings["mmw_inert"])
    return mmw


def getMM(species):
    """
    Get the molecular mass of a given species.

    This function uses the molmass package to
    calculate the mass number for the standard
    isotope of an input species. If all_iso
    is part of the input, it will return the
    mean molar mass.

    Args:
        species : string
            The chemical formula of the compound. ie C2H2 or H2O
    Returns:
        The molar mass of the compound in atomic mass units.
    """
    name = species.split("_")[0]
    name = name.split(",")[0]
    f = Formula(name)
    if "all_iso" in species:
        return f.mass
    return f.isotope.massnumber
