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
    Calculate the surface gravity of a planet.

    This function calculates the surface gravity of a planet given its mass and radius.
    If the surface gravity is already provided, it will skip this step.

    :param phys_vars: Dictionary of all the physical variables
    :type phys_vars: dict
    :param config: Dictionary containing all the config file settings/known values/parameters
    :type config: dict
    :return: Updated phys_vars dictionary
    :rtype: dict
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
    Set the log of ground pressure.

    This function checks if the surface pressure is provided or can be calculated from the
    provided parameters and brings it to the correct format for petitRADTRANS. A specific
    calculation is made in the case of settings_clouds='opaque'.

    :param phys_vars: Dictionary of all the physical variables
    :type phys_vars: dict
    :param config: Dictionary containing all the config file settings/known values/parameters
    :type config: dict
    :param knowns: Dictionary containing all known quantities
    :type knowns: dict
    :param use_truth: Boolean that allows to use the true value of the pressure rather than 10^{-4}
    :type use_truth: bool
    :return: Updated phys_vars dictionary
    :rtype: dict
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
    Calculate a polynomial temperature profile.

    This function takes in a pressure array and a dictionary of temperature variables,
    and returns an array of temperatures corresponding to the pressures using a polynomial function.

    :param pressure: The pressure values
    :type pressure: ndarray
    :param temp_vars: The dictionary of a_i coefficients to calculate the polynomial
    :type temp_vars: dict
    :return: The temperature profile for the given pressure levels
    :rtype: ndarray
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
    Calculate a spline temperature profile.

    This function takes in a pressure array and dictionaries of temperature variables, physical variables, and settings,
    and returns an array of temperatures corresponding to the pressures using a spline interpolation.

    :param pressure: The pressure values
    :type pressure: ndarray
    :param temp_vars: The dictionary of temperature variables
    :type temp_vars: dict
    :param phys_vars: The dictionary of physical variables
    :type phys_vars: dict
    :param settings: The dictionary of settings
    :type settings: dict
    :return: The temperature profile for the given pressure levels
    :rtype: ndarray
    """
    pressure_points = [phys_vars['log_P0']]
    for i in range(1,settings['spline_points']-1):
        pressure_points += [pressure_points[i-1] + temp_vars['Position_P'+str(i)] * (np.log10(pressure[0]) - pressure_points[i-1])]
    pressure_points += [np.log10(pressure[0])]

    try:
        temperature_points  = [temp_vars['T'+str(i)] for i in range(settings['spline_points'])]
    except:
        temperature_points  = [temp_vars['T'+str(i)] for i in range(settings['spline_points']-1)]+[temp_vars['T'+str(settings['spline_points']-2)]]

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


def calculate_adiabat_profile(pressure: ndarray, temp_vars: dict, phys_vars: dict) -> ndarray:
    """
    Calculate an adiabatic temperature profile.

    This function calculates the temperature profile assuming an adiabatic atmosphere.

    :param pressure: The pressure values
    :type pressure: ndarray
    :param temp_vars: The dictionary of temperature variables
    :type temp_vars: dict
    :param phys_vars: The dictionary of physical variables
    :type phys_vars: dict
    :return: The adiabatic temperature profile for the given pressure levels
    :rtype: ndarray
    """
    beta = 2.0/(temp_vars['Gas_Deg_Freedom']+2.0)

    temperature = temp_vars['Constant_Factor']**(1-beta) * pressure**beta

    P_inv = 10**(np.log10(pressure[0]) -  temp_vars['Position_P_Switch']*(np.log10(pressure[0])-phys_vars['log_P0']))
    
    ind = np.where(pressure <= P_inv)
    slope = ((temp_vars['Constant_Factor']**(1-beta) * P_inv**beta)-temp_vars['T_Top'])/(np.log10(P_inv)-np.log10(pressure[0]))
    
    temperature[ind] = temp_vars['Constant_Factor']**(1-beta) * P_inv**beta - slope * (np.log10(P_inv)-np.log10(pressure[ind]))

    if 'spline_smoothing' in temp_vars.keys():
        return scp.ndimage.gaussian_filter1d(temperature,temp_vars['adiabat_smoothing'],mode='nearest')
    else:
        return temperature


# TODO typeset vae_pt
def calculate_vae_profile(
    pressure: ndarray, vae_pt, temp_vars: dict
) -> ndarray:
    """
    Calculate a temperature profile using a variational autoencoder (VAE).

    This function takes in a pressure array, a VAE object, and a dictionary of temperature variables,
    and returns an array of temperatures corresponding to each pressure.

    :param pressure: The pressure values
    :type pressure: ndarray
    :param vae_pt: The VAE object to access the get_temperatures function
    :type vae_pt: object
    :param temp_vars: The dictionary of temperature coefficients
    :type temp_vars: dict
    :return: The temperature profile for the given pressure levels
    :rtype: ndarray
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
    Calculate a Guillot temperature profile.

    This function calculates the Guillot profile for a given set of parameters.

    :param pressure: The pressure array
    :type pressure: ndarray
    :param prt_instance: The instance of petitRADTRANS
    :type prt_instance: object
    :param temp_vars: The dictionary of temperature parameters to calculate the Guillot profile
    :type temp_vars: dict
    :return: The temperature profile for the given pressure levels
    :rtype: ndarray
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
    Calculate an isothermal temperature profile.

    This function returns a constant temperature profile for all pressure levels,
    representing an isothermal atmosphere.

    :param pressure: The pressure array (not used in this function, but included for consistency)
    :param temp_vars: A dictionary containing temperature variables. Should include 'T_eq' key.
    :return: An array of constant temperatures (T_eq) for all pressure levels
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
    Calculate the temperature of a parcel of air using the Madhusudhan-Seager equation.

    This function calculates the temperature at a given pressure using the Madhusudhan-Seager equation.

    :param pressure_m: The measured (or desired) pressure for which we want to calculate the temperature
    :type pressure_m: float
    :param pressure_i: The initial (or reference) atmospheric pressure level
    :type pressure_i: float
    :param temperature_i: The initial (or reference) atmospheric temperature
    :type temperature_i: float
    :param alpha: A constant used to calculate the temperature gradient
    :type alpha: float
    :param beta: A constant used to calculate the adiabatic lapse rate
    :type beta: float
    :return: The temperature at the pressure level m
    :rtype: float
    """
    return (np.log(pressure_m / pressure_i) / alpha) ** (
        1 / beta
    ) + temperature_i


def calculate_madhuseager_profile(
    pressure: ndarray, temp_vars: dict
) -> ndarray:
    """
    Calculate the temperature profile using the Madhusudhan & Seager (2009) model.

    This function calculates the temperature profile of a planet using the Madhusudhan & Seager (2009) model.

    :param pressure: The pressure array
    :type pressure: ndarray
    :param temp_vars: The dictionary of temperature parameters to calculate the Madhusudhan & Seager profile
    :type temp_vars: dict
    :return: The temperature profile for the given pressure levels
    :rtype: ndarray
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
    Calculate the temperature profile using a modified Madhusudhan-Seager model.

    This function calculates the temperature profile of a planet using a modified version 
    of the Madhusudhan-Seager model.

    :param pressure: The pressure array
    :type pressure: ndarray
    :param temp_vars: The dictionary of temperature values to calculate the modified Madhusudhan-Seager model
    :type temp_vars: dict
    :return: The temperature array
    :rtype: ndarray
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


def xi(gamma: float, tau: ndarray) -> ndarray:
    """
    Calculate Equation (14) of Line et al. (2013) Apj 775, 137

    :param gamma: Visible-to-thermal stream Planck mean opacity ratio
    :type gamma: float
    :param tau: Gray IR optical depth
    :type tau: ndarray
    :return: The calculated xi value
    :rtype: ndarray

    .. note::
       2014-12-10  patricio : Initial implementation.
    """
    return (2.0/3) * \
            (1 + (1./gamma) * (1 + (0.5*gamma*tau-1)*np.exp(-gamma*tau)) +
            gamma*(1 - 0.5*tau**2) * scp.special.expn(2, gamma*tau)              )


def calculate_line_profile(pressure: ndarray, temp_vars: dict, phys_vars: dict,
                R_star = 6.995e8, T_star = 5780.0, T_int = 0.0, sma = 1.0 * scp.constants.au):
    """
    Generate a PT profile based on input free parameters and pressure array.

    If no inputs are provided, it will run in demo mode, using free
    parameters given by the Line 2013 paper and some dummy pressure
    parameters.

    :param pressure: Array of pressure values in bars
    :type pressure: ndarray
    :param temp_vars: Dictionary containing temperature variables
    :type temp_vars: dict
    :param phys_vars: Dictionary containing physical variables
    :type phys_vars: dict
    :param R_star: Stellar radius (in meters), defaults to 6.995e8
    :type R_star: float, optional
    :param T_star: Stellar effective temperature (in Kelvin degrees), defaults to 5780.0
    :type T_star: float, optional
    :param T_int: Planetary internal heat flux (in Kelvin degrees), defaults to 0.0
    :type T_int: float, optional
    :param sma: Semi-major axis (in meters), defaults to 1.0 * scp.constants.au
    :type sma: float, optional
    :return: Temperature array
    :rtype: ndarray

    .. note::
        The `temp_vars` dictionary should contain:
        - kappa: Planck thermal IR opacity in units cm^2/gr (in log10)
        - gamma1: Visible-to-thermal stream Planck mean opacity ratio (in log10)
        - gamma2: Visible-to-thermal stream Planck mean opacity ratio (in log10)
        - alpha: Visible-stream partition (0.0--1.0)
        - beta: A 'catch-all' for albedo, emissivity, and day-night redistribution (on the order of unity)

        The `phys_vars` dictionary should contain:
        - grav: Planetary surface gravity (at 1 bar) in cm/second^2

    .. warning::
        The `T_int_type` parameter mentioned in the original docstring is not used in the function signature.

    .. note::
        **Developers:**

        Madison Stemm     : astromaddie@gmail.com

        Patricio Cubillos : pcubillos@fulbrightmail.org

        **Modification History:**

        2014-09-12  Madison  : Initial version, adapted from equations (13)-(16) in Line et al. (2013), Apj, 775, 137.

        2014-12-10  patricio : Reviewed and updated code.

        2015-01-22  patricio : Receive log10 of free parameters now.

        2019-02-13  mhimes   : Replaced `params` arg with each parameter for consistency with other PT models

        2019-09-10  mhimes   : Added T_int calculation from Thorngren et al. (2019)

      """

    # Stellar input temperature (at top of atmosphere):
    T_irr = temp_vars['beta'] * (R_star / (2.0*sma))**0.5 * T_star

    # Gray IR optical depth:
    tau = temp_vars['kappa'] * (pressure*1e6) / phys_vars['g'] # Convert bars to barye (CGS)

    xi1 = xi(temp_vars['gamma1'], tau)
    xi2 = xi(temp_vars['gamma2'], tau)

    # Temperature profile (Eq. 13 of Line et al. 2013):
    temperature = (0.75 * (T_int**4 * (2.0/3.0 + tau) +
                            T_irr**4 * (1-temp_vars['alpha']) * xi1 +
                            T_irr**4 * temp_vars['alpha']     * xi2 ) )**0.25

    return temperature


def calculate_abundances(chem_vars: dict, press: ndarray, settings: dict) -> Tuple[dict,dict]:
    """
    Calculate the abundances of chemical species in an atmosphere.

    This function takes chemical variables, pressure levels, and settings to compute
    the abundances of various molecules in an atmosphere. It can handle both
    Mass Mixing Ratio (MMR) and Volume Mixing Ratio (VMR) units.

    :param chem_vars: A dictionary containing the chemical variables for each molecule
    :type chem_vars: dict
    :param press: An array of pressure levels in the atmosphere
    :type press: ndarray
    :param settings: A dictionary containing configuration settings, including 'abundance_units'
    :type settings: dict
    :return: A tuple containing two dictionaries: The first dictionary contains the calculated abundances for each molecule; The second dictionary contains the chemical variables in VMR units (if conversion was needed)
    :rtype: Tuple[dict, dict]
    
    .. note::
        TBD CHECK HOW IT WORKS FOR SLOPES
    """
    abundances = {}

    for molecule in chem_vars.keys():
        if 'Drying' not in molecule:
            abundances[molecule] = np.ones_like(press) * chem_vars[molecule]

    if settings['abundance_units']=='MMR':
        inert = calculate_inert(abundances)
        mmw = calculate_mmw_MMR(abundances, settings, inert)
        abundances_VMR, inert = convert_MMR_to_VMR(abundances, settings, inert, mmw)
        chem_vars_VMR = {key: abundances_VMR[key][0] for key in abundances_VMR.keys()}
        return abundances_VMR, chem_vars_VMR
    
    else:
        return abundances, chem_vars


def calculate_inert(abundances: dict) -> ndarray:
    """
    Calculate the abundance of inert gases in the atmosphere.

    This function computes the amount of inert gases present in the atmosphere
    by subtracting the sum of all known abundances from 1.

    :param abundances: A dictionary containing the abundances of various chemical species.
                       Each key is a species name, and the corresponding value is an array
                       of abundances at different atmospheric levels.
    :type abundances: dict
    :return: An array representing the abundance of inert gases at each atmospheric level.
    :rtype: ndarray
    """
    total = np.zeros_like(abundances[list(abundances.keys())[0]])
    for key in abundances.keys():
            total = total + abundances[key]
    return 1.0 - total


def water_ice_vapor_pressure(T, T_ST=373.15, p_ST=1.01325, T_0=273.16, p_i0=6.1173*1e-3):
    """
    Calculate the vapor pressure of water or ice using the Goff-Gratch equation.

    This function computes the vapor pressure of water (for T >= T_0) or ice (for T < T_0)
    based on the given temperature using the Goff-Gratch equation.

    :param T: Temperature in Kelvin
    :type T: float
    :param T_ST: Steam point temperature in Kelvin, defaults to 373.15 K (100°C)
    :type T_ST: float, optional
    :param p_ST: Steam point pressure in bars, defaults to 1.01325 bar (1 atm)
    :type p_ST: float, optional
    :param T_0: Triple point temperature of water in Kelvin, defaults to 273.16 K (0.01°C)
    :type T_0: float, optional
    :param p_i0: Vapor pressure of ice at the triple point in bars, defaults to 6.1173e-3 bar
    :type p_i0: float, optional
    :return: Vapor pressure of water or ice in bars
    :rtype: float

    .. note::
        The Goff-Gratch equation is used for its accuracy over a wide range of temperatures.
        For T >= T_0, the equation for water is used. For T < T_0, the equation for ice is used.
    """
    
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



def condense_water_old(abundances_VMR, pressure, temperature, phys_vars, settings, drying=0.0):
    """
    Calculate the condensation of water in an atmosphere.

    This function computes the condensation of water vapor in an atmosphere
    based on the given abundances, pressure, temperature, and other parameters.

    :param abundances_VMR: Dictionary of volume mixing ratios for atmospheric constituents
    :type abundances_VMR: dict
    :param pressure: Array of pressure levels in the atmosphere
    :type pressure: ndarray
    :param temperature: Array of temperature values corresponding to pressure levels
    :type temperature: ndarray
    :param phys_vars: Dictionary of physical variables
    :type phys_vars: dict
    :param settings: Dictionary of settings for the calculation
    :type settings: dict
    :param drying: Drying factor, defaults to 0.0
    :type drying: float, optional
    :return: Tuple containing updated abundances_VMR and condensation pressures
    :rtype: tuple

    .. note::
        This function calculates the condensation of water vapor by comparing
        the partial pressure of water to its vapor pressure at each atmospheric level.
    """

    # Convert mass mixing ration to volume mixing ratio
    # and calculate the partial pressure of water
    PP_Water = abundances_VMR['H2O']*pressure

    # Calculate the vapor pressure of water at all pressures
    VP_Water = np.zeros_like(pressure)
    for index in range(len(VP_Water)):
        VP_Water[index] = water_ice_vapor_pressure(temperature[index])

    # Calculation of the variable water partial pressure profile.
    # If water partial pressure exceeds vapor pressure the water condenses 
    # condensation_pressures stores the layers wher condensation occurrs
    #layer_thickness = np.log10(pressure[0])-np.log10(pressure[1])
    condensation_pressures = []
    if len(pressure) == settings['n_layers']:
        above_surface = range(settings['n_layers'])
        below_surface = None
    else:
        above_surface = np.where(np.log10(pressure) <= phys_vars['log_P0'])[0]
        below_surface = np.where(np.log10(pressure) >  phys_vars['log_P0'])[0]
        drying = drying*settings['n_layers']/len(above_surface)
    drying_factor = 1
    for index in above_surface:
        if PP_Water[index] >= VP_Water[index]:
            PP_Water[index:] = (VP_Water[index]/pressure[index])*pressure[index:]
            VP_Water = VP_Water*(1-drying)
            drying_factor *= (1-drying)
            if drying_factor >= 0.1:
                condensation_pressures += [pressure[index]]

    # Calculate theVMR of the Water in the atmosphere
    VMR_Water = PP_Water/pressure
    if below_surface is not None:
        VMR_Water[below_surface] = VMR_Water[above_surface[0]]

    # retrun the new abundances
    abundances_VMR['H2O'] = VMR_Water[::-1]

    #if len(condensation_pressures) == 0:
    #    median_cond_pressure = None
    #else: 
    #    median_cond_pressure = np.median(condensation_pressures)

    return abundances_VMR, condensation_pressures


def condense_water(abundances_VMR, pressure, temperature, phys_vars, settings, drying=0.0):
    """
    Calculate the condensation of water in an atmosphere.

    This function computes the condensation of water vapor in an atmosphere
    based on the given abundances, pressure, temperature, and other parameters.

    :param abundances_VMR: Dictionary of volume mixing ratios for atmospheric constituents
    :type abundances_VMR: dict
    :param pressure: Array of pressure levels in the atmosphere
    :type pressure: ndarray
    :param temperature: Array of temperature values corresponding to pressure levels
    :type temperature: ndarray
    :param phys_vars: Dictionary of physical variables
    :type phys_vars: dict
    :param settings: Dictionary of settings for the calculation
    :type settings: dict
    :param drying: Drying factor, defaults to 0.0
    :type drying: float, optional
    :return: Tuple containing updated abundances_VMR and condensation pressures
    :rtype: tuple

    .. note::
        This function calculates the condensation of water vapor by comparing
        the partial pressure of water to its vapor pressure at each atmospheric level.
    """

    # Convert mass mixing ration to volume mixing ratio
    # and calculate the partial pressure of water
    PP_Water = abundances_VMR['H2O']*pressure

    # Calculate the vapor pressure of water at all pressures
    VP_Water = np.zeros_like(pressure)
    for index in range(len(VP_Water)):
        VP_Water[index] = water_ice_vapor_pressure(temperature[index]) #10**(np.log10(relative_humidity)-index*drying)*

    # Calculation of the variable water partial pressure profile.
    # If water partial pressure exceeds vapor pressure the water condenses 
    # condensation_pressures stores the layers wher condensation occurrs
    #layer_thickness = np.log10(pressure[0])-np.log10(pressure[1])
    if len(pressure) == settings['n_layers']:
        above_surface = range(settings['n_layers'])
        below_surface = None
    else:
        above_surface = np.where(np.log10(pressure) <= phys_vars['log_P0'])[0]
        below_surface = np.where(np.log10(pressure) >  phys_vars['log_P0'])[0]
        #drying = drying*settings['n_layers']/len(above_surface)
    condensation_levels = []
    for index in above_surface:
        if PP_Water[index] >= VP_Water[index]:
            PP_Water[index:] = (VP_Water[index]/pressure[index])*pressure[index:]
            condensation_levels += [len(pressure)-1-index]

    # Calculate theVMR of the Water in the atmosphere
    VMR_Water = PP_Water/pressure
    if below_surface is not None:
        VMR_Water[below_surface] = VMR_Water[above_surface[0]]
    abundances_VMR['H2O'] = VMR_Water[::-1]

    # Calculate theVMR of the Water in the atmosphere
    layer_drying = np.log10(drying)*abundances_VMR['H2O'][0]/abundances_VMR['H2O']
    abundances_VMR['H2O'] = 10**(np.log10(abundances_VMR['H2O'])+layer_drying)

    # Get the layers where cloud formation is possible. If no condensation set clouds to surface
    condensation_pressures = [pressure[::-1][i] for i in set(condensation_levels).intersection(np.where(layer_drying >= -1)[0])]
    if condensation_pressures == []:
        if condensation_pressures == []:
            condensation_pressures = [pressure[0]]
        else:
            condensation_pressures = pressure[::-1][max(condensation_levels)]

    return abundances_VMR, condensation_pressures


def assign_cloud_parameters(
    abundances_VMR: dict, cloud_vars: dict, press: ndarray,phys_vars: dict, condensation_pressures: float,
) -> Tuple[dict, dict, dict, int]:
    """
    Assign cloud parameters and update abundances.

    This function takes in the abundances dictionary, cloud_vars dictionary, and pressure array. 
    It calculates the bottom pressure of each cloud layer by adding its top pressure to its thickness. 
    It sets all abundance values outside a given cloud layer to 0 (i.e., it removes them from consideration). 
    It also creates a new cloud_radii dictionary that contains only the particle radii for each type of condensate.

    :param abundances_VMR: Dictionary of volume mixing ratios for each species
    :type abundances_VMR: dict
    :param cloud_vars: Dictionary containing cloud parameters
    :type cloud_vars: dict
    :param press: Array of pressure levels
    :type press: ndarray
    :param phys_vars: Dictionary of physical variables
    :type phys_vars: dict
    :param condensation_pressures: Pressures at which condensation occurs
    :type condensation_pressures: float
    :return: A tuple containing updated abundances, cloud variables, cloud radii, cloud lognormal distribution parameter, cloud pressure, and cloud fraction
    :rtype: Tuple[dict, dict, dict, int, float, float]

    """

    cloud_radii = {}
    cloud_lnorm = 0
    cloud_Pcloud = None
    cloud_fraction = None
    for cloud in cloud_vars.keys():
        if cloud =='cloud_fraction':
            cloud_fraction = cloud_vars['cloud_fraction']
            if not (('Pcloud' in cloud_vars.keys()) or ('Position_Pcloud' in cloud_vars.keys())):
                if condensation_pressures == []:
                    cloud_Pcloud = 10**phys_vars['log_P0']
                else:
                    cloud_Pcloud = np.median(condensation_pressures)
        elif cloud =='Pcloud':
            cloud_Pcloud = cloud_vars['Pcloud']
        elif cloud == 'Position_Pcloud':
            if condensation_pressures == None:
                cloud_Pcloud = 10**(phys_vars['log_P0'] + cloud_vars['Position_Pcloud'] * (np.log10(press[0]) - phys_vars['log_P0']))
            elif condensation_pressures == []:
                cloud_Pcloud = 10**phys_vars['log_P0']
            else:
                cloud_Pcloud = 10**(np.log10(condensation_pressures[0]) + cloud_vars['Position_Pcloud'] * (np.log10(condensation_pressures[-1]) - np.log10(condensation_pressures[0])))
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
    Calculate the mean molecular weight of each layer in the atmosphere.

    This function calculates the mean molecular weight (MMW) for each layer of the atmosphere
    based on the provided volume mixing ratios (VMR) of different species, settings, and
    the weight of inert gas present.

    :param abundances_VMR: Dictionary containing the volume mixing ratios of different species
    :type abundances_VMR: dict
    :param settings: Dictionary containing various settings for the calculation
    :type settings: dict
    :param inert: Array representing the weight of the inert gas in each layer
    :type inert: ndarray
    :return: Array of mean molecular weights for each atmospheric layer
    :rtype: ndarray

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
    Calculate the mean molecular weight of each layer in the atmosphere.

    This function computes the mean molecular weight for each atmospheric layer
    based on the provided mass mixing ratios (MMR) of different species, settings,
    and the weight of inert gas present.

    :param abundances_MMR: Dictionary containing the mass mixing ratios of different species
    :type abundances_MMR: dict
    :param settings: Dictionary containing various settings for the calculation
    :type settings: dict
    :param inert: Array representing the weight of the inert gas in each layer
    :type inert: ndarray
    :return: Array of mean molecular weights for each atmospheric layer
    :rtype: ndarray
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
    Convert Volume Mixing Ratio (VMR) to Mass Mixing Ratio (MMR) for atmospheric species.

    This function converts the volume mixing ratios of various atmospheric species to
    their corresponding mass mixing ratios. It also handles the conversion of inert
    gases if present.

    :param abundances_VMR: Dictionary containing volume mixing ratios of atmospheric species
    :type abundances_VMR: dict
    :param settings: Dictionary containing various settings for the calculation
    :type settings: dict
    :param inert_VMR: Array representing the volume mixing ratio of inert gases in each layer
    :type inert_VMR: ndarray
    :param mmw: Array of mean molecular weights for each atmospheric layer
    :type mmw: ndarray
    :return: A tuple containing the dictionary of mass mixing ratios and the array of inert gas MMR
    :rtype: Tuple[dict, ndarray]
    """
    abundances_MMR = {}#copy.deepcopy(abundances_VMR)

    for key in abundances_VMR.keys():
        abundances_MMR[key] = abundances_VMR[key]*get_mm(key)/mmw

    if "mmw_inert" in settings.keys():
        inert_MMR = inert_VMR*settings["mmw_inert"]/mmw
    
    return abundances_MMR, inert_MMR


def convert_MMR_to_VMR(abundances_MMR: dict, settings: dict, inert_MMR: ndarray, mmw: ndarray) ->  Tuple[dict, ndarray]:
    """
    Convert Mass Mixing Ratio (MMR) to Volume Mixing Ratio (VMR) for atmospheric species.

    This function converts the mass mixing ratios of various atmospheric species to
    their corresponding volume mixing ratios. It also handles the conversion of inert
    gases if present.

    :param abundances_MMR: Dictionary containing mass mixing ratios of atmospheric species
    :type abundances_MMR: dict
    :param settings: Dictionary containing various settings for the calculation
    :type settings: dict
    :param inert_MMR: Array representing the mass mixing ratio of inert gases in each layer
    :type inert_MMR: ndarray
    :param mmw: Array of mean molecular weights for each atmospheric layer
    :type mmw: ndarray
    :return: A tuple containing the dictionary of volume mixing ratios and the array of inert gas VMR
    :rtype: Tuple[dict, ndarray]
    """
    abundances_VMR = {}#abundances_MMR.copy()

    for key in abundances_MMR.keys():
        abundances_VMR[key] = abundances_MMR[key]/get_mm(key)*mmw

    if "mmw_inert" in settings.keys():
        inert_VMR = inert_MMR/settings["mmw_inert"]*mmw
    
    return abundances_VMR, inert_VMR


def get_mm(species: str) -> float:
    """
    Get the molecular mass of a given species.

    This function uses the molmass package to calculate the mass number for the standard isotope
    of an input species. If 'all_iso' is part of the input, it will return the mean molar mass.

    :param species: The chemical formula of the compound, e.g., C2H2 or H2O
    :type species: str
    :return: The molar mass of the compound in atomic mass units
    :rtype: float
    """
    name = species.split("_")[0]
    name = name.split(",")[0]
    name = name.replace("(c)", "")
    f = Formula(name)
    if "all_iso" in species:
        return f.mass
    return f.isotope.massnumber
