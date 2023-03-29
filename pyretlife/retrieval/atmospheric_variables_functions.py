from molmass import Formula
import astropy.constants as const
import numpy as np
from typing import Union, Tuple
from pyretlife.retrieval import pt_vae as vae
from numpy import ndarray, poly1d


def calculate_gravity(phys_vars:dict) -> dict:
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
                const.G.cgs.value
                * phys_vars["M_pl"]
                / (phys_vars["R_pl"]) ** 2
            )
    return phys_vars

def calculate_log_ground_pressure(phys_vars:dict) -> dict:
    """
    Function to check if the surface gravity is provided or can
    be calculated from the provided parameters.
    """

    # Calculate the surface gravity g given M_Pl and R_pl or log_g.
    # If in knowns already, skip
    if "log_P0" not in phys_vars.keys():
        if "P0" in phys_vars.keys():
            phys_vars["log_P0"] = np.log10(phys_vars['P0'])
    return phys_vars

def calculate_polynomial_profile(P: ndarray,temp_vars:dict) -> ndarray:

    return np.array(np.polyval(
    np.array(
        [
            temp_vars["a_" + str(len(temp_vars) - 1 - i)]
            for i in range(len(temp_vars))
        ]
    ),
    np.log10(P),
    ))

#TODO typeset vae_pt
def calculate_vae_profile(P:ndarray, vae_pt, temp_vars:dict) -> ndarray:
    return vae_pt.get_temperatures(
    z=np.array(
        [
            temp_vars["z_" + str(i + 1)]
            for i in range(len(temp_vars))
        ]
    ),
    log_p=np.log10(P))


def calculate_guillot_profile(P: ndarray, pRT, temp_vars:dict) ->ndarray:
    return pRT.nat_cst.guillot_modif(
        P,
        1e1 ** temp_vars["log_delta"],
        1e1 ** temp_vars["log_gamma"],
        temp_vars["t_int"],
        temp_vars["t_equ"],
        1e1 ** temp_vars["log_p_trans"],
        temp_vars["alpha"],
    )

def calculate_isothermal_profile(P:ndarray, temp_vars:dict)-> ndarray:
    return temp_vars["T_eq"] * np.ones_like(P)


def calculate_madhuseager_profile(P:ndarray,temp_vars:dict) -> ndarray:

    import scipy.ndimage as sci
    beta1 = 0.5
    beta2 = 0.5

    def T_P(P_m, P_i, T_i, alpha, beta):
        return (np.log(P_m / P_i) / alpha) ** (1 / beta) + T_i

    P0, P1, P2, P3 = (
        10 ** P[0], #log_top_pressure by definition
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
            T[i] = T_P(
                P[i], P2, T2, temp_vars["alpha2"], beta2
            )
        elif P[i] > P3:
            T[i] = T3

    T = sci.gaussian_filter1d(T, 20.0, mode="nearest")

def calculate_mod_madhuseager_profile(P:ndarray,temp_vars:dict) -> ndarray:

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
            T[i] = T_P(
                P[i], P2, T2, temp_vars["alpha2"], beta2
            )
    return T


def get_MMW_from_nfrac(n_frac):
    """
    Calculate the mean molecular weight from a number fraction

    Args:
        n_fracs : dict
            A dictionary of number fractions
    """
    mass = 0.0
    for key,value in n_frac.items():
        spec = key.split("_R_")[0]
        mass += value*getMM(spec)
    return mass


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
    name = name.split(',')[0]
    f = Formula(name)
    if "all_iso" in species:
        return f.mass
    return f.isotope.massnumber