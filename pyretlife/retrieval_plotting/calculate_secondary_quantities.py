__author__ = "Konrad, Alei, Molliere, Quanz"
__copyright__ = "Copyright 2022, Konrad, Alei, Molliere, Quanz"
__maintainer__ = "Björn S. Konrad, Eleonora Alei"
__email__ = "konradb@ethz.ch, elalei@phys.ethz.ch"
__status__ = "Development"

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
import sys
import numpy as np
import time as t
import contextlib
import astropy.constants as apc
import astropy.units as u
import scipy as sp

from pyretlife.retrieval.atmospheric_variables import calculate_gravity
from pyretlife.retrieval.radiative_transfer import scale_flux_to_distance



# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

def parallel_pt_profile_calculation(rp_object,parameter_samples,skip = 1,layers=500,p_surf=4,n_processes=None,process=None):
    '''
    Function to calculate the PT profiles corresponding to the retrieved posterior distributions
    for subsequent plotting in the flux PT plotting functions.
    '''

    # Iterate over the equal weighted posterior and Split up the jobs onto the multiple processes
    dimension = np.shape(parameter_samples)[0]//skip
    process,ind_start,ind_end = task_assignment('PT-Profile',n_processes,dimension,process)

    # Print status of calculation
    if process == 0:
        print('Starting PT-profile calculation.')
        print('\t0.00 % of PT-profiles calculated.', end = "\r")

    pt_profiles = {}
    t_start = t.time()
    for i in range(ind_start,ind_end):
        ind = skip*i

        # Fetch the known parameters and a sample of retrieved
        # parameters from the posteriors
        rp_object.assign_cube_to_parameters(parameter_samples[ind,:])
        #self.P0_test(ind=i)

        if "log_P0" not in rp_object.parameters.keys():
            rp_object.phys_vars["log_P0"] = np.log10(rp_object.phys_vars["P0"])

        # Calculate the cloud bottom pressure from the cloud thickness parameter
        cloud_tops = []
        cloud_bottoms = []
        for key in rp_object.cloud_vars.keys():
            cloud_bottoms += [rp_object.cloud_vars[key]['top_pressure']+rp_object.cloud_vars[key]['thickness']]
            cloud_tops += [rp_object.cloud_vars[key]['top_pressure']]
            rp_object.calculate_pt_profile(
                    parameterization=rp_object.settings["parameterization"],
                    log_ground_pressure=p_surf,
                    log_top_pressure=np.log10(np.min(cloud_tops)),
                    layers=layers,
                    )
            pressure_cloud_top = rp_object.press[0]
            temperature_cloud_top = rp_object.temp[0]

        # Extrapolate the retrieved P-T profile to higher pressures
        rp_object.calculate_pt_profile(
                parameterization=rp_object.settings["parameterization"],
                log_ground_pressure=p_surf,
                log_top_pressure=rp_object.settings["log_top_pressure"],
                layers=layers,
                )
        pressure_extrapolated = rp_object.press
        temperature_extrapolated = rp_object.temp
        ind = np.where(rp_object.press > 10**rp_object.phys_vars['log_P0'])

        # Calculate the pressure temperature profile corresponding to the set of parameters
        rp_object.calculate_pt_profile(
                parameterization=rp_object.settings["parameterization"],
                log_ground_pressure=rp_object.phys_vars["log_P0"],
                log_top_pressure=rp_object.settings["log_top_pressure"],
                layers=rp_object.settings["n_layers"],
                )

        # Initialize the arrays for storage
        if i==ind_start:
            size = ind_end-ind_start
            pt_profiles['pressures'] = np.zeros((size,len(rp_object.press)))
            pt_profiles['temperatures'] = np.zeros((size,len(rp_object.temp)))
            pt_profiles['pressures_extrapolated'] = np.zeros((size,len(pressure_extrapolated)))
            pt_profiles['temperatures_extrapolated'] = np.zeros((size,len(temperature_extrapolated)))
            if len(rp_object.cloud_vars) != 0:
                pt_profiles['pressures_cloud_top'] = np.zeros((size,len([pressure_cloud_top])))
                pt_profiles['temperatures_cloud_top'] = np.zeros((size,len([temperature_cloud_top])))

        # Save the results
        save = i-ind_start
        pt_profiles['pressures'][save,:] = rp_object.press
        pt_profiles['temperatures'][save,:] = rp_object.temp
        pt_profiles['pressures_extrapolated'][save,:] = pressure_extrapolated
        pt_profiles['temperatures_extrapolated'][save,:] = temperature_extrapolated
        if len(rp_object.cloud_vars) != 0:
            pt_profiles['pressures_cloud_top'][save,:] = pressure_cloud_top
            pt_profiles['temperatures_cloud_top'][save,:] = temperature_cloud_top

        # Print status of calculation
        if process == 0:
            t_end = t.time()
            remain_time = (t_end-t_start)/((i+1)/(ind_end-ind_start))-(t_end-t_start)
            print('\t'+str(np.round((i+1)/(ind_end-ind_start)*100,2))+' % of PT-profiles calculated. Estimated time remaining: '+str(remain_time//3600)+
                        ' h, '+str((remain_time%3600)//60)+' min.        ', end = "\r")

    # Print status of calculation
    if process == 0:
        print('\nPT-profile calculation completed.')

    return pt_profiles



def parallel_spectrum_calculation(rp_object,parameter_samples,skip = 1,n_processes=None,process=None):
    '''
    Function to calculate the fluxes corresponding to the retrieved posterior distributions
    for subsequent plotting in the flux plotting functions.
    '''

    # Iterate over the equal weighted posterior and Split up the jobs onto the multiple processes
    dimension = np.shape(parameter_samples)[0]//skip
    process,ind_start,ind_end = task_assignment('Spectrum',n_processes,dimension,process)

    # Initialize the RT object and read the data
    with contextlib.redirect_stdout(None):
        rp_object.petitRADTRANS_initialization()

    # Print status of calculation
    if process == 0:
        print('Starting spectrum calculation.')
        print('\t0.00 % of spectra calculated.', end = "\r")

    spectra = {}
    t_start = t.time()
    for i in range(ind_start,ind_end):
        ind = skip*i

        # Fetch the known parameters and a sample of retrieved
        # parameters from the posteriors
        rp_object.assign_cube_to_parameters(parameter_samples[ind,:])
        rp_object.phys_vars = calculate_gravity(rp_object.phys_vars,rp_object.config)
            
        if "log_P0" not in rp_object.parameters.keys():
            rp_object.phys_vars["log_P0"] = np.log10(rp_object.phys_vars["P0"])

        # Test the values of P0 and g and change to required values if necessary
        #self.g_test()
        #self.P0_test()

        # Calculate the pressure temperature profile corresponding to the set of parameters
        rp_object.calculate_pt_profile(
                parameterization=rp_object.settings["parameterization"],
                log_ground_pressure=rp_object.phys_vars["log_P0"],
                log_top_pressure=rp_object.settings["log_top_pressure"],
                layers=rp_object.settings["n_layers"],
                )

        rp_object.calculate_spectrum()
        rp_object.distance_scale_spectrum()
         
        # Store the calculated spectra
        if i == 0:
            spectra['wavelengths'] = rp_object.rt_object.wavelength
        if i==ind_start:
            size = ind_end-ind_start

            # Initialize the arrays for storage
            spectra['fluxes'] = np.zeros((size,len(rp_object.rt_object.flux)))
            spectra['emission_contribution'] = np.zeros((size,np.shape(rp_object.rt_object.contr_em)[0],np.shape(rp_object.rt_object.contr_em)[1]))
            if rp_object.settings['include_moon'] == 'True':
                spectra['moon_fluxes'][save,:] = np.zeros((size,len(rp_object.rt_object.flux)))

        # Save the calculated spectra
        save = i-ind_start
        spectra['fluxes'][save,:] = rp_object.rt_object.flux
        spectra['emission_contribution'][save,:,:] = rp_object.rt_object.contr_em
        if rp_object.settings['include_moon'] == 'True':
            spectra['moon_fluxes'][save,:] = rp_object.moon_flux
            
        # Print status of calculation
        if process == 0:
            t_end = t.time()
            remain_time = (t_end-t_start)/((i+1)/(ind_end-ind_start))-(t_end-t_start)
            print('\t'+str(np.round((i+1)/(ind_end-ind_start)*100,2))+' % of spectra calculated. Estimated time remaining: '+str(remain_time//3600)+
                    ' h, '+str((remain_time%3600)//60)+' min.            ', end = "\r")

    # Print status of calculation
    if process == 0:
        print('\nSpectrum calculation completed.')

    #return the calculated results
    return spectra



# Assignment of Tasks to different processes for parallel computation
def task_assignment(calculation_type,n_processes,dimension,process):
    if (n_processes is not None) and (process is not None):
        if process == 0:
            # Print the task assignment
            print_task_assignment(calculation_type,n_processes,dimension)
        ind_start = process*dimension//n_processes
        ind_end = min(dimension,(process+1)*dimension//n_processes)
    else:
        ind_start = 0
        ind_end = dimension
        process = 0
    return process,ind_start,ind_end



# Printing function for parallel computation
def print_task_assignment(calculation_type,n_processes,dimension):
    print('\n-----------------------------------------------------')
    print('\n    '+str(calculation_type)+' calculation on multiple CPUs:')
    print('')
    print('    Number of calculations:\t'+str(dimension))
    print('    Number of processes:\t'+str(n_processes))
    print('')
    print('    Distribution of tasks:')
    for proc_ind in range(n_processes):
        print('\tProcess '+str(proc_ind)+':\t'+str(calculation_type)+':\t'+str(proc_ind*dimension//n_processes+1)+'-'+str(min(dimension,(proc_ind+1)*dimension//n_processes)))
    print('\n-----------------------------------------------------\n')



# Routine that calculates the bond albedo of a planet
def bond_albedo_calculation(rp_object,
                            stellar_luminosity,
                            error_stellar_luminosity,
                            planet_star_separation,
                            error_planet_star_separation):

    wavelengths = rp_object.retrieved_wavelengths*rp_object.units.retrieval_units['wavelength']
    planet_radius = rp_object.posteriors['R_pl'].to_numpy()*rp_object.units.retrieval_units['R_pl']
    posterior_dimension = rp_object.posteriors['R_pl'].size

    # Generating random data for the stellar luminosity and the panet separation
    stellar_luminosities = stellar_luminosity*apc.L_sun + error_stellar_luminosity*np.random.randn(posterior_dimension)*apc.L_sun
    planet_star_separations = planet_star_separation*apc.au + error_planet_star_separation*np.random.randn(posterior_dimension)*apc.au

    # initializing arrays to store the calculation results
    retrieved_equilibrium_temperature = np.zeros(posterior_dimension)
    retrieved_bond_albedo = np.zeros(posterior_dimension)

    # function to calculate the BB spectrum given a temperature
    def blackbody_lam(x, temperature):
        exponent = apc.h*apc.c/(wavelengths.to(u.m)*apc.k_B*temperature*u.K)
        BB_flux = (2*np.pi*apc.h*apc.c**2/(wavelengths.to(u.m)**5*(np.exp(exponent) - 1)))
        return [sp.integrate.simps(BB_flux.value,wavelengths.to(u.m).value)]

    # calculate the equilibrium temperature for all points in the posterior
    for i in range(np.size(retrieved_equilibrium_temperature)):
        rescaled_retrieved_flux = scale_flux_to_distance(rp_object.retrieved_fluxes[i,:],rp_object.knowns['d_syst']['truth'],planet_radius[i].to(u.m).value)*rp_object.units.retrieval_units['flux']
        converted_retrieved_flux = rescaled_retrieved_flux.to(u.J/(u.m**3*u.s),equivalencies=u.spectral_density(wavelengths))
        integrated_retrieved_flux = sp.integrate.simps(converted_retrieved_flux.value,wavelengths.to(u.m).value)

        retrieved_equilibrium_temperature[i], cov = sp.optimize.curve_fit(blackbody_lam, [1], integrated_retrieved_flux,p0=[300])
        retrieved_bond_albedo[i] = 1 - 16*np.pi*planet_star_separations[i]**2*apc.sigma_sb*(retrieved_equilibrium_temperature[i]*u.K)**4/stellar_luminosities[i]

    return retrieved_equilibrium_temperature, retrieved_bond_albedo

"""

    wavelengths = rp_object.retrieved_wavelengths*rp_object.units.retrieval_units['wavelength']
    planet_radius = rp_object.posteriors['R_pl'].to_numpy()*rp_object.units.retrieval_units['R_pl']

    retrieved_equilibrium_temperature = np.zeros_like(planet_radius.value)#*u.K

    def blackbody_lam(x,temperature):

        exponent = apc.h*apc.c/(wavelengths.to(u.m)*apc.k_B*temperature*u.K)
        BB_flux = 2*np.pi*apc.h*apc.c**2 / (wavelengths.to(u.m)**5 * (np.exp(exponent) - 1)) # calculate the BB flux

        return [sp.integrate.simps(BB_flux.value,wavelengths.to(u.m))]

    print(rp_object.knowns)

    for index in range(np.size(planet_radius)):
        factor = 1e7/1e6/(rp_object.petitRADTRANS.nat_cst.c/rp_object.retrieved_wavelengths*1e4)*1e6*rp_object.retrieved_wavelengths*1e-6*(planet_radius[index].value*1e2)**2/(rp_object.knowns['d_syst']['truth'])**2
        int_f = sp.integrate.simps(np.ndarray.flatten(rp_object.retrieved_fluxes[index])/factor,wavelengths.to(u.m))


        retrieved_equilibrium_temperature[index], cov = sp.optimize.curve_fit(blackbody_lam, [1], int_f,p0=[300])#*u.K
        print(retrieved_equilibrium_temperature[index])
    sys.exit()

            self.ret_opaque_T[index], cov = sp.optimize.curve_fit(blackbody_lam, [1], np.sum(np.ndarray.flatten(self.retrieved_fluxes[i])/factor),p0=[300])








        # Generating random data for the stellar luminosity and the panet separation
        L_star_data = L_star + sigma_L_star*np.random.randn(*self.ret_opaque_T.shape)
        sep_planet_data = sep_planet + sigma_sep_planet*np.random.randn(*self.ret_opaque_T.shape)





        # Defining constants needed for the calculations
        L_sun = 3.826*1e26
        AU = 1.495978707*1e11
        sigma_SBoltzmann = 5.670374419*1e-8

        # Converting stellar luminosity and planet separation to SI
        L_star_data_SI = L_star_data * L_sun
        sep_planet_data_SI = sep_planet_data * AU

        # Calculate the bond albedo
        self.A_Bond_ret = 1 - 16*np.pi*sep_planet_data_SI**2*sigma_SBoltzmann*self.ret_opaque_T**4/L_star_data_SI
        A_Bond_true = 1 - 16*np.pi*(sep_planet * AU)**2*sigma_SBoltzmann*T_equ_true**4/(L_star*L_sun)

        return A_Bond_true, T_equ_true
"""


