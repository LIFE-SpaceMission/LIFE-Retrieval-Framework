from typing import Tuple
import scipy.ndimage as sci
import numpy as np
from astropy.constants import G
from molmass import Formula
from numpy import ndarray


def calculate_gravity(phys_vars: dict, config: dict) -> dict:
    """
    The calculate_gravity function calculates the surface gravity of a planet given its mass and radius.
        If the surface gravity is already provided, it will skip this step.

    :param phys_vars: The dictionary of all the physical variables
    :param config:  The dictionary containing all the config file settings/known values/parameters.
    :return: The updated phys_vars dictionary
    """

    # Calculate the surface gravity g given M_Pl and R_pl or log_g.
    # If in knowns already, skip
    if "g" not in config["PHYSICAL PARAMETERS"].keys():
        if "log_g" in config["PHYSICAL PARAMETERS"].keys():
            phys_vars["g"] = 10 ** phys_vars["log_g"]
        else:
            phys_vars["g"] = (
                G.cgs.value * phys_vars["M_pl"] / (phys_vars["R_pl"]) ** 2
            )
    return phys_vars


def set_log_ground_pressure(
    phys_vars: dict, config: dict, knowns: dict, use_truth: bool = False
) -> dict:
    """
    The set_log_ground_pressure function checks if the surface pressure is provided or can be calculated from the
    provided parameters and brings it to the correct format for petitRADTRANS. A specific calculation is made in the
    case of settings_clouds='opaque'.

    :param phys_vars: The dictionary of all the physical variables
    :param config: The dictionary containing all the config file settings/known values/parameters.
    :param knowns: The dictionary containing all known quantities.
    :param use_truth: bool: A boolean that allows to use the true value of the pressure rather than 10^{-4}.
        Used for plotting, for opaque cloudy cases only.
    :return: The updated phys_vars dictionary
    """

    # Case dependant setting of the surface pressure
    if "CLOUD PARAMETERS" in config.keys():
        if config["CLOUD PARAMETERS"]["settings_clouds"] == "opaque":
            # Choose a surface pressure below the lower cloud deck
            if not (
                ("log_P0" in config["PHYSICAL PARAMETERS"].keys())
                or ("P0" in config["PHYSICAL PARAMETERS"].keys())
            ):
                phys_vars["log_P0"] = 4
            else:
                if ("log_P0" in knowns) or ("P0" in knowns):
                    if use_truth:
                        if "P0" in knowns:
                            phys_vars["log_P0"] = np.log10(
                                knowns["P0"]["truth"]
                            )
                    else:
                        phys_vars["log_P0"] = 4
                else:
                    raise RuntimeError(
                        "ERROR! For opaque cloud models, the surface pressure "
                        "P0 is not retrievable!"
                    )
        else:
            if "log_P0" not in config["PHYSICAL PARAMETERS"].keys():
                if "P0" in config["PHYSICAL PARAMETERS"].keys():
                    phys_vars["log_P0"] = np.log10(phys_vars["P0"])
                else:
                    raise RuntimeError("ERROR! Either log_P0 or P0 is needed!")
    else:
        if "log_P0" not in config["PHYSICAL PARAMETERS"].keys():
            if "P0" in config["PHYSICAL PARAMETERS"].keys():
                phys_vars["log_P0"] = np.log10(phys_vars["P0"])
            else:
                raise RuntimeError("ERROR! Either log_P0 or P0 is needed!")

    return phys_vars


def calculate_polynomial_profile(pressure: ndarray, temp_vars: dict) -> ndarray:
    """
    The calculate_polynomial_profile function takes in a pressure array and a dictionary of temperature variables,
    and returns an array of temperatures corresponding to the pressures. The function uses the numpy polyval function
    to calculate these temperatures.

    :param pressure: The pressure values
    :param temp_vars: The dictionary of a_i coefficients to calculate the polynomial.
    :return: The log10 of the pressure profile
    """
    return np.array(
        np.polyval(
            np.array(
                [
                    temp_vars["a_" + str(len(temp_vars) - 1 - i)]
                    for i in range(len(temp_vars))
                ]
            ),
            np.log10(pressure),
        )
    )


# TODO typeset vae_pt
def calculate_vae_profile(
    pressure: ndarray, vae_pt, temp_vars: dict
) -> ndarray:
    """
    TBD
    The calculate_vae_profile function takes in a pressure array and the vae_pt object,
    and returns an array of temperatures corresponding to each pressure. The function
    also takes in a dictionary of temperature variables that are used to calculate the
    temperatures.

    :param pressure: The pressure values
    :param vae_pt: Access the get_temperatures function from the vae_pt object
    :param temp_vars: The dictionary of temperature coefficients
    :return: The temperature profile for the given pressure levels
    """
    return vae_pt.get_temperatures(
        z=np.array(
            [temp_vars["z_" + str(i + 1)] for i in range(len(temp_vars))]
        ),
        log_p=np.log10(pressure),
    )


def calculate_guillot_profile(
    pressure: ndarray, prt_instance, temp_vars: dict
) -> ndarray:
    """
    The calculate_guillot_profile function calculates the Guillot profile for a given set of parameters.

    :param pressure: The pressure array
    :param prt_instance: The instance of petitRADTRANS
    :param temp_vars: The dictionary of temperature parameters to calculate the Guillot profile.
    :return: The temperature profile for the given pressure levels
    """
    return prt_instance.nat_cst.guillot_modif(
        pressure,
        1e1 ** temp_vars["log_delta"],
        1e1 ** temp_vars["log_gamma"],
        temp_vars["t_int"],
        temp_vars["t_equ"],
        1e1 ** temp_vars["log_p_trans"],
        temp_vars["alpha"],
    )


def calculate_isothermal_profile(pressure: ndarray, temp_vars: dict) -> ndarray:
    """
    The calculate_isothermal_profile function calculates the temperature profile for an isothermal atmosphere.

    :param pressure: The pressure array
    :param temp_vars: The dictionary of temperature parameters to calculate an isothermal profile.
    :return: The temperature profile for the given pressure levels
    """
    return temp_vars["T_eq"] * np.ones_like(pressure)


def madhuseager_temperature_calculator(
    pressure_m: float,
    pressure_i: float,
    temperature_i: float,
    alpha: float,
    beta: float,
) -> float:
    """
    CHECK The madhuseager_temperature_calculator function calculates the temperature of a parcel of air at a given
    pressure using the Madhusudhan-Seager equation. The function takes in four parameters: pressure_m, pressure_i,
    temperature_i, and alpha. Pressure_m is the measured (or desired) pressure for which we want to calculate the
    temperature. Pressure_i is the initial (or reference) atmospheric level from which we are calculating our new
    value for T(p). Temperature_i is the initial (or reference) atmospheric level's corresponding temperature value
    at p = p(initial). Alpha and beta are constants that

    :param pressure_m:  Calculate the temperature at a given pressure
    :param pressure_i:  Set the initial pressure
    :param temperature_i:  Set the initial temperature
    :param alpha:  Calculate the temperature gradient, and beta is used to calculate the adiabatic lapse rate
    :param beta:  Calculate the temperature of a gas at a given pressure
    :return: The temperature at the pressure level m
    """
    return (np.log(pressure_m / pressure_i) / alpha) ** (
        1 / beta
    ) + temperature_i


def calculate_madhuseager_profile(
    pressure: ndarray, temp_vars: dict
) -> ndarray:
    """
    The calculate_madhuseager_profile function calculates the temperature profile of a planet using the
    Madhusudhan&Seager (2009) model.

    :param pressure: The pressure array
    :param temp_vars: The dictionary of temperature parameters to calculate the Madhusudhan&Seager profile.
    :return: The temperature profile for the given pressure levels
    """

    beta1 = 0.5
    beta2 = 0.5

    pressure_0, pressure_1, pressure_2, pressure_3 = (
        pressure[0],  # log_top_pressure by definition
        10 ** temp_vars["log_P1"],
        10 ** temp_vars["log_P2"],
        10 ** temp_vars["log_P3"],
    )

    temperature = np.zeros_like(pressure)

    temperature_2 = (
        temp_vars["T0"]
        + (np.log(pressure_1 / pressure_0) / temp_vars["alpha1"]) ** (1 / beta1)
        - (np.log(pressure_1 / pressure_2) / temp_vars["alpha2"]) ** (1 / beta2)
    )
    temperature_3 = madhuseager_temperature_calculator(
        pressure_3, pressure_2, temperature_2, temp_vars["alpha2"], beta2
    )

    for i in range(np.size(pressure)):
        if pressure[i] < pressure_1:
            temperature[i] = madhuseager_temperature_calculator(
                pressure[i],
                pressure_0,
                temp_vars["T0"],
                temp_vars["alpha1"],
                beta1,
            )
        elif pressure_1 < pressure[i] < pressure_3:
            temperature[i] = madhuseager_temperature_calculator(
                pressure[i],
                pressure_2,
                temperature_2,
                temp_vars["alpha2"],
                beta2,
            )
        elif pressure[i] > pressure_3:
            temperature[i] = temperature_3

    temperature = sci.gaussian_filter1d(temperature, 20.0, mode="nearest")
    return temperature


def calculate_mod_madhuseager_profile(
    pressure: ndarray, temp_vars: dict
) -> ndarray:
    """
    The calculate_mod_madhuseager_profile function calculates the temperature profile of a planet using the modified
    Madhusudhan-Seager model. The function takes in an array of pressures and a dictionary containing all relevant
    variables for calculating the temperature profile. It returns an array of temperatures corresponding to each
    pressure value.

    :param pressure: The pressure array
    :param temp_vars: The dictionary of temperature values to calculate the modified Madhusudhan-Seager model.
    :return: The temperature array
    """
    beta1 = 0.5
    beta2 = 0.5

    pressure_0, pressure_1, pressure_2 = (
        pressure[0],
        10 ** temp_vars["log_P1"],
        10 ** temp_vars["log_P2"],
    )

    temperature = np.zeros_like(pressure)

    temperature_2 = (
        temp_vars["T0"]
        + (np.log(pressure_1 / pressure_0) / temp_vars["alpha1"]) ** (1 / beta1)
        - (np.log(pressure_1 / pressure_2) / temp_vars["alpha2"]) ** (1 / beta2)
    )

    for i in range(np.size(pressure)):
        if pressure[i] < pressure_1:
            temperature[i] = madhuseager_temperature_calculator(
                pressure[i],
                pressure_0,
                temp_vars["T0"],
                temp_vars["alpha1"],
                beta1,
            )
        elif pressure_1 < pressure[i]:
            temperature[i] = madhuseager_temperature_calculator(
                pressure[i],
                pressure_2,
                temperature_2,
                temp_vars["alpha2"],
                beta2,
            )
    return temperature


def calculate_abundances(chem_vars: dict, press: ndarray) -> dict:
    """
    TBD
    The calculate_abundances function takes a dictionary of chemical variables and an array of pressures,
    and returns a dictionary with the abundances for each molecule.
    CHECK HOW IT WORKS FOR SLOPES

    :param chem_vars: The dictionary of chemical variables.
    :param press: The array of pressures.
    :return: A dictionary of updated abundances.
    """
    abundances = {}

    slopes = [
        parameter for parameter in chem_vars.keys() if "Slope" in parameter
    ]
    molecules = [
        parameter for parameter in chem_vars.keys() if "Slope" not in parameter
    ]

    for molecule in molecules:
        if "Slope_" + molecule in slopes:
            abundances[molecule] = 10 ** (
                np.log10(press) * chem_vars["Slope_" + molecule]
                + np.log10(chem_vars[molecule])
            )
        else:
            abundances[molecule] = np.ones_like(press) * chem_vars[molecule]

    return abundances


def assign_cloud_parameters(
    abundances: dict, cloud_vars: dict, press: ndarray
) -> Tuple[dict, dict, dict, int]:
    # TODO test that it works
    """
    CHECK The assign_cloud_parameters function takes in the abundances dictionary, cloud_vars dictionary,
    and pressure array. It then calculates the bottom pressure of each cloud layer by adding its top pressure to its
    thickness. It sets all abundance values outside a given cloud layer to 0 (i.e., it removes them from
    consideration). It also creates a new cloud_radii dictionary that contains only the particle radii for each type
    of condensate (i.e., no more information about the clouds themselves). Finally, it returns these three
    dictionaries as well as an integer representing lnorm.

    :param abundances: Store the abundances of each element in a dictionary
    :param cloud_vars: Set the cloud parameters
    :param press: Set the abundance outside the cloud layer to 0
    :return: The abundances, cloud_vars, cloud_radii and cloud_lnorm
    """
    cloud_radii = {}
    cloud_lnorm = 0
    for cloud in cloud_vars.keys():
        cloud_vars[cloud]["bottom_pressure"] = (
            cloud_vars[cloud]["top_pressure"] + cloud_vars[cloud]["thickness"]
        )

        # set the abundance outside the cloud layer to 0
        abundances[cloud.split("_")[0]][
            np.where(press > cloud_vars[cloud]["bottom_pressure"])
        ] = 0
        abundances[cloud.split("_")[0]][
            np.where(press < cloud_vars[cloud]["top_pressure"])
        ] = 0

        cloud_radii[cloud.split("_")[0]] = cloud_vars[cloud]["particle_radius"]
        # TODO is it the same for all clouds then?
        cloud_lnorm = cloud_vars[cloud]["sigma_lnorm"]
    return abundances, cloud_vars, cloud_radii, cloud_lnorm


def calc_mmw(abundances: dict, settings: dict, inert: ndarray) -> ndarray:
    """
    The calc_mmw function calculates the mean molecular weight of each layer in the atmosphere.

    :param abundances: The abundances dictionary
    :param settings: The settings dictionary
    :param inert: The weight of the inert gas
    :return: The mean molecular weight of the gas in each layer
    """
    mmw = np.zeros_like(range(settings["n_layers"]), dtype=float)
    for layer in range(settings["n_layers"]):
        for key in abundances.keys():
            mmw[layer] = mmw[layer] + abundances[key][layer] * get_mm(key)

        if "mmw_inert" in settings.keys():
            mmw[layer] = mmw[layer] + inert[layer] * float(
                settings["mmw_inert"]
            )
    return mmw


def get_mm(species: str) -> float:
    """
    Get the molecular mass of a given species.

    This function uses the molmass package to calculate the mass number for the standard isotope of an input species.
    If all_iso is part of the input, it will return the mean molar mass.

    :param species: string: The chemical formula of the compound. ie C2H2 or H2O
    :return: The molar mass of the compound in atomic mass units.
    """
    name = species.split("_")[0]
    name = name.split(",")[0]
    name = name.replace("(c)", "")
    f = Formula(name)
    if "all_iso" in species:
        return f.mass
    return f.isotope.massnumber
