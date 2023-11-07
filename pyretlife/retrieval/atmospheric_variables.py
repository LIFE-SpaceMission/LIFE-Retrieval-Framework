from typing import Tuple
import scipy.ndimage as sci
import scipy as scp
import numpy as np
from astropy.constants import G
from molmass import Formula
from numpy import ndarray
import sys


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
    temperature = np.array(
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
    return temperature # np.maximum(temperature,0.01)


def calculate_spline_profile(pressure: ndarray, temp_vars: dict, phys_vars: dict, settings: dict) -> ndarray:
    """
    The calculate_spline_profile function takes in a pressure array and a dictionary of temperature variables,
    and returns an array of temperatures corresponding to the pressures. The function uses the scipy.interpolate.make_interp_spline
    function to calculate these temperatures.

    :param pressure: The pressure values
    :param temp_vars: The dictionary of a_i coefficients to calculate the polynomial.
    :return: The temperature profile for the given pressure levels
    """
    
    pressure_points = [phys_vars['log_P0']]
    for i in range(1,settings['spline_points']-1):
        pressure_points += [pressure_points[i-1] + temp_vars['Position_P'+str(i)] * (np.log10(pressure[0]) - pressure_points[i-1])]
    pressure_points += [np.log10(pressure[0])]

    temperature_points  = [temp_vars['T'+str(i)] for i in range(settings['spline_points'])]

    spline = scp.interpolate.make_interp_spline(pressure_points[::-1],temperature_points[::-1],k=settings['spline_degree_k'])

    if 'spline_smoothing' in temp_vars.keys():
        if len(pressure) == settings['n_layers']:
            temperature = scp.ndimage.gaussian_filter1d(spline(np.log10(pressure)),temp_vars['spline_smoothing'],mode='nearest')
        else:
            true_inds = np.where(np.log10(pressure)<=phys_vars['log_P0'])
            smoothing_factor = len(true_inds[0])/settings['n_layers']
            temperature = spline(np.log10(pressure))
            temperature[np.where(np.log10(pressure)>phys_vars['log_P0'])]=temp_vars['T0']
            temperature = scp.ndimage.gaussian_filter1d(temperature,temp_vars['spline_smoothing']*smoothing_factor,mode='nearest')

    else:
        temperature = spline(np.log10(pressure))

    return temperature


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


def calculate_abundances(chem_vars: dict, press: ndarray, settings: dict) -> dict:
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

    for molecule in chem_vars.keys():
        if 'Drying' not in molecule:
            abundances[molecule] = np.ones_like(press) * chem_vars[molecule]

    if settings['abundance_units']=='MMR':
        inert = calculate_inert(abundances)
        mmw = calculate_mmw_MMR(abundances, settings, inert)
        abundances, inert = convert_MMR_to_VMR(abundances, settings, inert, mmw)
        return abundances
    
    else:
        return abundances



def calculate_inert(abundances: dict):
    total = np.zeros_like(abundances[list(abundances.keys())[0]])
    for key in abundances.keys():
            total = total + abundances[key]
    return 1.0 - total



def water_ice_vapor_pressure(T,T_ST=373.15,p_ST=1.01325,T_0=273.16,p_i0=6.1173*1e-3):
    
    #Calculated using the Goff–Gratch equation (see wikipedia)
    if (T >= T_0):
        # Goff–Gratch equation for Water
        term1 = -7.90298*(T_ST/T-1.)
        term2 = 5.02808*np.log10(T_ST/T)
        term3 = -1.3816e-7*(10**(11.344*(1.-T/T_ST))-1.)
        term4 = 8.1328*10**(-3)*(10**(-3.49149*(T_ST/T-1.))-1.)
        term5 = np.log10(p_ST)
        p_vp = 10.**(term1+term2+term3+term4+term5)
    else:
        # Goff–Gratch equation for Ice
        term1 = -9.09718*(T_0/T-1)
        term2 = -3.56654*np.log10(T_0/T)
        term3 = 0.876793*(1-T/T_0)
        term4 = np.log10(p_i0)
        p_vp = 10**(term1+term2+term3+term4)
    return p_vp


    
def condense_water(abundances_VMR,pressure,temperature,phys_vars,settings,drying=0.0):

    # Convert mass mixing ration to volume mixing ratio
    # and calculate the partial pressure of water
    PP_Water = abundances_VMR['H2O']*pressure

    # Calculate the vapor pressure of water at all pressures
    VP_Water = np.zeros_like(pressure)
    for index in range(len(VP_Water)):
        VP_Water[index]=water_ice_vapor_pressure(temperature[index])

    # Calculation of the variable water partial pressure profile.
    # If water partial pressure exceeds vapor pressure the water condenses 
    # condensation_pressures stores the layers wher condensation occurrs
    condensation_pressures = []
    if len(pressure) == settings['n_layers']:
        above_surface = range(settings['n_layers'])
        below_surface = None
    else:
        above_surface = np.where(np.log10(pressure) <= phys_vars['log_P0'])[0]
        below_surface = np.where(np.log10(pressure) >  phys_vars['log_P0'])[0]
        drying = drying*settings['n_layers']/len(above_surface)
    for index in above_surface:
        if PP_Water[index] >= VP_Water[index]:
            PP_Water[index:] = (VP_Water[index]/pressure[index])*pressure[index:]
            VP_Water = 10**(np.log10(VP_Water)-drying)
            condensation_pressures += [pressure[index]]

    # Calculate theVMR of the Water in the atmosphere
    VMR_Water = PP_Water/pressure
    if below_surface is not None:
        VMR_Water[below_surface] = VMR_Water[above_surface[0]]

    # retrun the new abundances
    abundances_VMR['H2O'] = VMR_Water[::-1]

    if len(condensation_pressures) == 0:
        median_cond_pressure = None
    else: 
        median_cond_pressure = np.median(condensation_pressures)

    return abundances_VMR, median_cond_pressure



def assign_cloud_parameters(
    abundances_VMR: dict, cloud_vars: dict, press: ndarray,phys_vars: dict, median_cond_pressure: float,
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
    cloud_Pcloud = None
    cloud_fraction = None
    for cloud in cloud_vars.keys():
        if cloud =='cloud_fraction':
            cloud_fraction = cloud_vars['cloud_fraction']
            if not (('Pcloud' in cloud_vars.keys()) or ('Position_Pcloud' in cloud_vars.keys())):
                cloud_Pcloud = median_cond_pressure
        elif cloud =='Pcloud':
            cloud_Pcloud = cloud_vars['Pcloud']
        elif cloud == 'Position_Pcloud':
            cloud_Pcloud = 10**(phys_vars['log_P0'] + cloud_vars['Position_Pcloud'] * (np.log10(press[0]) - phys_vars['log_P0']))
        elif ('_cloud_top' in cloud):
            ind_cltp = np.argmin(np.abs(np.log10(press)-np.log10(cloud_vars['Pcloud'])))
            ind_surf = len(press)-1
            if ind_surf == ind_cltp:
                ind_cltp -= 1
            slope = (np.log10(abundances_VMR[cloud.split('_cloud_top')[0]][0]) - np.log10(cloud_vars[cloud]))/(ind_surf-ind_cltp)
            abundances_VMR[cloud.split('_cloud_top')[0]]=10 ** (
                (ind_surf-np.arange(0,len(press),1)) * slope
                + np.log10(abundances_VMR[cloud.split('_cloud_top')[0]][0])
            )
            abundances_VMR[cloud.split('_cloud_top')[0]][press<cloud_vars['Pcloud']]=cloud_vars[cloud]
        else:
            cloud_vars[cloud]["bottom_pressure"] = (
                cloud_vars[cloud]["top_pressure"] + cloud_vars[cloud]["thickness"]
            )

            # set the abundance outside the cloud layer to 0
            abundances_VMR[cloud.split("_")[0]][
                np.where(press > cloud_vars[cloud]["bottom_pressure"])
            ] = 0
            abundances_VMR[cloud.split("_")[0]][
                np.where(press < cloud_vars[cloud]["top_pressure"])
            ] = 0

            cloud_radii[cloud.split("_")[0]] = cloud_vars[cloud]["particle_radius"]
            # TODO is it the same for all clouds then?
            cloud_lnorm = cloud_vars[cloud]["sigma_lnorm"]
    return abundances_VMR, cloud_vars, cloud_radii, cloud_lnorm, cloud_Pcloud, cloud_fraction


def calculate_mmw_VMR(abundances_VMR: dict, settings: dict, inert: ndarray) -> ndarray:
    """
    The calc_mmw function calculates the mean molecular weight of each layer in the atmosphere.

    :param abundances: The abundances dictionary
    :param settings: The settings dictionary
    :param inert: The weight of the inert gas
    :return: The mean molecular weight of the gas in each layer
    """
    mm = {key: get_mm(key) for key in abundances_VMR.keys()}
    size = np.size(abundances_VMR[list(abundances_VMR.keys())[0]])
    mmw = np.zeros_like(range(size), dtype=float)
    for layer in range(size):
        for key in abundances_VMR.keys():
            mmw[layer] = mmw[layer] + abundances_VMR[key][layer] * mm[key]

        if "mmw_inert" in settings.keys():
            mmw[layer] = mmw[layer] + inert[layer] * float(
                settings["mmw_inert"]
            )
    return mmw

def calculate_mmw_MMR(abundances_MMR: dict, settings: dict, inert: ndarray) -> ndarray:
    """
    The calc_mmw function calculates the mean molecular weight of each layer in the atmosphere.

    :param abundances: The abundances dictionary
    :param settings: The settings dictionary
    :param inert: The weight of the inert gas
    :return: The mean molecular weight of the gas in each layer
    """
    mm = {key: get_mm(key) for key in abundances_MMR.keys()}
    size = np.size(abundances_MMR[list(abundances_MMR.keys())[0]])
    mmw = np.zeros_like(range(size), dtype=float)
    for layer in range(size):
        for key in abundances_MMR.keys():
            mmw[layer] = mmw[layer] + abundances_MMR[key][layer] / mm[key]

        if "mmw_inert" in settings.keys():
            mmw[layer] = mmw[layer] + inert[layer] / float(
                settings["mmw_inert"]
            )
    return 1.0 / mmw

def convert_VMR_to_MMR(abundances_VMR: dict, settings: dict, inert_VMR: ndarray, mmw: ndarray) -> Tuple[dict, ndarray]:
    """
    The calc_mmw function calculates the mean molecular weight of each layer in the atmosphere.

    :param abundances: The abundances dictionary
    :param settings: The settings dictionary
    :param inert: The weight of the inert gas
    :return: The mean molecular weight of the gas in each layer
    """
    abundances_MMR = {}#copy.deepcopy(abundances_VMR)

    for key in abundances_VMR.keys():
        abundances_MMR[key] = abundances_VMR[key]*get_mm(key)/mmw

    if "mmw_inert" in settings.keys():
        inert_MMR = inert_VMR*settings["mmw_inert"]/mmw
    
    return abundances_MMR, inert_MMR



def convert_MMR_to_VMR(abundances_MMR: dict, settings: dict, inert_MMR: ndarray, mmw: ndarray) ->  Tuple[dict, ndarray]:
    """
    The calc_mmw function calculates the mean molecular weight of each layer in the atmosphere.

    :param abundances: The abundances dictionary
    :param settings: The settings dictionary
    :param inert: The weight of the inert gas
    :return: The mean molecular weight of the gas in each layer
    """
    abundances_VMR = {}#abundances_MMR.copy()

    for key in abundances_VMR.keys():
        abundances_VMR[key] = abundances_MMR[key]/get_mm(key)*mmw

    if "mmw_inert" in settings.keys():
        inert_VMR = inert_MMR/settings["mmw_inert"]*mmw
    
    return abundances_VMR, inert_VMR



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
