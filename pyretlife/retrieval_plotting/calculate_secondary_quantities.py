__author__ = "Konrad, Alei, Molliere, Quanz"
__copyright__ = "Copyright 2022, Konrad, Alei, Molliere, Quanz"
__maintainer__ = "Björn S. Konrad, Eleonora Alei"
__email__ = "konradb@ethz.ch, elalei@phys.ethz.ch"
__status__ = "Development"

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import time as t
import contextlib

from pyretlife.retrieval.atmospheric_variables import calculate_gravity



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