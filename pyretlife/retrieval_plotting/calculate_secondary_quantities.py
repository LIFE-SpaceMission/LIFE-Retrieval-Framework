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
import matplotlib.pyplot as plt
import copy

from pyretlife.retrieval.atmospheric_variables import (
    calculate_gravity,
    set_log_ground_pressure)

from pyretlife.retrieval.radiative_transfer import scale_flux_to_distance

from pyretlife.retrieval_plotting.color_handling import (
    generate_quantile_color_levels,
    generate_color_map_from_levels,
)

from pyretlife.retrieval.atmospheric_variables import (
    calculate_gravity,
    set_log_ground_pressure,
    calculate_mmw_VMR,
    convert_VMR_to_MMR,
)


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

def parallel_pt_profile_calculation(rp_object,parameter_samples,skip = 1,layers=500,p_surf=4,n_processes=None,process=None,use_truth=False):
    '''
    Function to calculate the PT profiles corresponding to the retrieved posterior distributions
    for subsequent plotting in the flux PT plotting functions.
    '''

    # Iterate over the equal weighted posterior and Split up the jobs onto the multiple processes
    dimension = np.shape(parameter_samples)[0]//skip
    process,ind_start,ind_end = task_assignment('PT-Profile',n_processes,dimension,process)

    # Initialize the arrays for storage
    size = ind_end-ind_start
    pt_profiles = {}
    pt_profiles['pressures'] = np.zeros((size,rp_object.settings["n_layers"]))
    pt_profiles['temperatures'] = np.zeros((size,rp_object.settings["n_layers"]))
    pt_profiles['pressures_extrapolated'] = np.zeros((size,layers))
    pt_profiles['temperatures_extrapolated'] = np.zeros((size,layers))
    if len([key for key in rp_object.parameters.keys() if 'top_pressure' in key]) > 0:
        pt_profiles['pressures_cloud_top'] = np.zeros(size)
        pt_profiles['temperatures_cloud_top'] = np.zeros(size)

    # Print status of calculation
    if process == 0:
        print('Starting PT-profile calculation.')
        print('\t0.00 % of PT-profiles calculated.', end = "\r")

    t_start = t.time()
    for i in range(ind_start,ind_end):
        ind = skip*i

        # Fetch the known parameters and a sample of retrieved
        # parameters from the posteriors
        rp_object.assign_cube_to_parameters(parameter_samples[ind,:])
        rp_object.phys_vars = set_log_ground_pressure(rp_object.phys_vars, rp_object.config, rp_object.knowns, use_truth = use_truth)
        rp_object.phys_vars = calculate_gravity(rp_object.phys_vars,rp_object.config)

        # Calculate the cloud bottom pressure from the cloud thickness parameter
        cloud_tops = []
        for key in rp_object.cloud_vars.keys():
            if not ((key in ['cloud_fraction','Pcloud','Position_Pcloud']) or ('_cloud_top' in key)):
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

        # Save the results
        save = i-ind_start
        pt_profiles['pressures'][save,:] = rp_object.press
        pt_profiles['temperatures'][save,:] = rp_object.temp
        pt_profiles['pressures_extrapolated'][save,:] = pressure_extrapolated
        pt_profiles['temperatures_extrapolated'][save,:] = temperature_extrapolated
        if len(cloud_tops) != 0:
            pt_profiles['pressures_cloud_top'][save] = pressure_cloud_top
            pt_profiles['temperatures_cloud_top'][save] = temperature_cloud_top

        # Print status of calculation
        if process == 0:
            t_end = t.time()
            remain_time = (t_end-t_start)/((i+1)/(size))-(t_end-t_start)
            print('\t'+str(np.round((i+1)/(size)*100,2))+' % of PT-profiles calculated. Estimated time remaining: '+str(remain_time//3600)+
                        ' h, '+str((remain_time%3600)//60)+' min.        ', end = "\r")

    # Print status of calculation
    if process == 0:
        print('\nPT-profile calculation completed.')

    return pt_profiles



def parallel_spectrum_calculation(rp_object,parameter_samples,skip = 1,n_processes=None,process=None,use_truth=False,emission_contribution=True):
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

    # Initialize the arrays for storage
    size = ind_end-ind_start
    spectra = {}
    spectra['gravity'] = np.zeros(size)
    spectra['fluxes'] = np.zeros((size,len(rp_object.rt_object.freq)))
    if emission_contribution:
        spectra['emission_contribution'] = np.zeros((size,rp_object.settings["n_layers"],len(rp_object.rt_object.freq)))
    if rp_object.settings['include_moon'] == 'True':
        spectra['moon_fluxes'][save,:] = np.zeros((size,len(rp_object.rt_object.freq)))

    # Print status of calculation
    if process == 0:
        print('Starting spectrum calculation.')
        print('\t0.00 % of spectra calculated.', end = "\r")

    t_start = t.time()
    for i in range(ind_start,ind_end):
        ind = skip*i

        # Fetch the known parameters and a sample of retrieved
        # parameters from the posteriors
        rp_object.assign_cube_to_parameters(parameter_samples[ind,:])
        rp_object.phys_vars = calculate_gravity(rp_object.phys_vars,rp_object.config)
        rp_object.phys_vars = set_log_ground_pressure(rp_object.phys_vars, rp_object.config, rp_object.knowns, use_truth = use_truth)

        # Calculate the pressure temperature profile corresponding to the set of parameters
        rp_object.calculate_pt_profile(
                parameterization=rp_object.settings["parameterization"],
                log_ground_pressure=rp_object.phys_vars["log_P0"],
                log_top_pressure=rp_object.settings["log_top_pressure"],
                layers=rp_object.settings["n_layers"],
                )

        # Abundance and spectrum calculation
        rp_object.calculate_abundances()
        rp_object.rt_object.wavelength, rp_object.rt_object.flux, rp_object.rt_object.contr_em = rp_object.calculate_spectrum(em_contr=True)
        rp_object.distance_scale_spectrum()
         
        # Store the calculated spectra
        save = i-ind_start
        if i == 0:
            spectra['wavelengths'] = rp_object.rt_object.wavelength
        spectra['gravity'][save] = rp_object.phys_vars["g"]
        spectra['fluxes'][save,:] = rp_object.rt_object.flux
        if emission_contribution:
            spectra['emission_contribution'][save,:,:] = rp_object.rt_object.contr_em
        if rp_object.settings['include_moon'] == 'True':
            spectra['moon_fluxes'][save,:] = rp_object.moon_flux
            
        # Print status of calculation
        if process == 0:
            t_end = t.time()
            remain_time = (t_end-t_start)/((i+1)/(size))-(t_end-t_start)
            print('\t'+str(np.round((i+1)/(size)*100,2))+' % of spectra calculated. Estimated time remaining: '+str(remain_time//3600)+
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



def add_secondary_parameter_to_parameters(parameters,name,unit,title,true_value=None):
        # add the bond albedo to the parameters
        parameters[name] = {'input_unit':unit,
                            'unit':unit,
                            'input_truth':true_value,
                            'truth':true_value,
                            'type':'SECONDARY PARAMETERS',
                            'title':title,
                            'prior':{'kind':None}}
        
        return parameters



def abundance_profile_calculation(rp_object,layers=500,p_surf=4,use_truth=False):
    '''
    Function to calculate the abundance profiles corresponding to the retrieved
    posterior distributions for subsequent plotting.
    '''

    #Get the Data and the dimension of the data
    parameter_samples = rp_object.posteriors.to_numpy()[:,:-1]
    dimension = np.shape(parameter_samples)[0]

    # Initialize the dictionary for storage
    abundance_profiles = {}

    # Print status of calculation
    t_start = t.time()
    print('Starting abundance profile calculation.')
    print('\t0.00 % of PT-profiles calculated.', end = "\r")

    for i in range(dimension):
        # Fetch the known parameters and a sample of retrieved
        # parameters from the posteriors
        rp_object.assign_cube_to_parameters(parameter_samples[i,:])
        rp_object.phys_vars = calculate_gravity(rp_object.phys_vars,rp_object.config)
        rp_object.phys_vars = set_log_ground_pressure(rp_object.phys_vars, rp_object.config, rp_object.knowns, use_truth = use_truth)

        # Calculate the pressure temperature profile corresponding to the set of parameters
        rp_object.calculate_pt_profile(
                parameterization=rp_object.settings["parameterization"],
                log_ground_pressure=rp_object.phys_vars["log_P0"],
                log_top_pressure=rp_object.settings["log_top_pressure"],
                layers=rp_object.settings["n_layers"],
                )
        
        # Calculate the abundances profiles and store results
        rp_object.calculate_abundances()
        if i == 0:
            abundance_profiles['pressures'] = np.zeros((dimension,rp_object.settings["n_layers"]))
            for key in rp_object.abundances_VMR.keys():
                abundance_profiles[key+'_VMR'] = np.zeros((dimension,rp_object.settings["n_layers"]))
                abundance_profiles[key+'_MMR'] = np.zeros((dimension,rp_object.settings["n_layers"]))
        abundances_MMR = VMR_to_MMR(rp_object.abundances_VMR,rp_object.settings)
        abundance_profiles['pressures'][i,:] = rp_object.press
        for key in rp_object.abundances_VMR.keys():
            abundance_profiles[key+'_VMR'][i,:]  = rp_object.abundances_VMR[key]
            abundance_profiles[key+'_MMR'][i,:]  = abundances_MMR[key]
        
        # Extrapolate the retrieved P-T profile to higher pressures
        rp_object.calculate_pt_profile(
                parameterization=rp_object.settings["parameterization"],
                log_ground_pressure=p_surf,
                log_top_pressure=rp_object.settings["log_top_pressure"],
                layers=layers,
                )   

        # Calculate the abundances profiles and store results
        rp_object.calculate_abundances()
        if i == 0:
            abundance_profiles['pressures_extrapolated'] = np.zeros((dimension,layers))
            for key in rp_object.abundances_VMR.keys():
                abundance_profiles[key+'_VMR_extrapolated'] = np.zeros((dimension,layers))
                abundance_profiles[key+'_MMR_extrapolated'] = np.zeros((dimension,layers))
        for key in rp_object.abundances_VMR.keys():
            rp_object.abundances_VMR[key][np.where(rp_object.abundances_VMR[key]>abundance_profiles[key+'_VMR'][i,-1])]=abundance_profiles[key+'_VMR'][i,-1]
        abundances_MMR = VMR_to_MMR(rp_object.abundances_VMR,rp_object.settings)
        abundance_profiles['pressures_extrapolated'][i,:] = rp_object.press
        for key in rp_object.abundances_VMR.keys():
            abundance_profiles[key+'_VMR_extrapolated'][i,:] = rp_object.abundances_VMR[key]
            abundance_profiles[key+'_MMR_extrapolated'][i,:] = abundances_MMR[key]

        # Print the status of the calcuation
        if i%100 == 0:
            t_end = t.time()
            remain_time = (t_end-t_start)/((i+1)/(dimension))-(t_end-t_start)
            print('\t'+str(np.round((i+1)/(dimension)*100,2))+' % of abundance profiles calculated. Estimated time remaining: '+str(remain_time//3600)+
                            ' h, '+str((remain_time%3600)//60)+' min.        ', end = "\r")

    # Print status of calculation
    print('Abundance profile calculation completed.                                                    ')

    #return the calculated results
    return abundance_profiles



def VMR_to_MMR(abundances_VMR,settings):
    
    # Calculate the inert mass of the atmosphere
    total = np.zeros_like(abundances_VMR[list(abundances_VMR.keys())[0]])
    for abundance_VMR in abundances_VMR.keys():
        total = total + abundances_VMR[abundance_VMR]
    inert = (np.ones_like(total) - total)

    # Calculate the mean molecular weight of the atmosphere
    MMW = calculate_mmw_VMR(abundances_VMR, settings, inert)

    # Convert VMR to MMR and return the result
    abundances_MMR, inert_MMR = convert_VMR_to_MMR(abundances_VMR, settings, inert, MMW)
    return abundances_MMR



def calculate_profile_contours(rp_object,pressures_extrapolated,pressures,parameters_extrapolated,parameters,quantiles,plot_residual=False,smoothing=4,ax=None):
    '''
    Function to calculate the contours that correspond to a set of provided quantiles
    for given a set of profiles (e.g. retrieved P-T profiles).
    '''
        
    # Find the quantiles for the different pressures and temperatures of the extrapolated profiles
    pressure_layers_quantiles = [np.nanquantile(pressures_extrapolated,q,axis=0) for q in quantiles]
    if plot_residual:
        parameter_layers_quantiles = [np.nanquantile(parameters_extrapolated,q,axis=0)-np.nanquantile(parameters_extrapolated,0.5,axis=0) for q in quantiles]
    else:
        parameter_layers_quantiles = [np.nanquantile(parameters_extrapolated,q,axis=0) for q in quantiles]

    # Merge the P-T profile quantiles with the surface pressure if retrieved
    color_levels, level_thresholds, N_levels = generate_quantile_color_levels('k',quantiles)
    pressure_max = 1e6
    pressure_layers_bottom = len(quantiles)//2*[[]]
    parameter_layers_bottom = len(quantiles)//2*[[]]
    if not rp_object.settings['include_scattering']['clouds'] == True:

        # Special condition for residula plotting
        if plot_residual:
            mean_S_parameters = np.median(parameters[:,-1])
        else:
            mean_S_parameters = 0

        # Define limits and make a 2d histogram of the surface pressures and temperatures
        parameter_eb = np.quantile(parameters[:,-1],[0.16,0.5,0.84])
        parameter_lim = [parameter_eb[1]-4*(parameter_eb[1]-parameter_eb[0]),parameter_eb[1]+4*(parameter_eb[2]-parameter_eb[1])]
        pressure_eb = np.quantile(np.log10(pressures[:,-1]),[0.16,0.5,0.84])
        pressure_lim = [pressure_eb[1]-4*(pressure_eb[1]-pressure_eb[0]),pressure_eb[1]+4*(pressure_eb[2]-pressure_eb[1])]
        #parameter_lim = np.quantile(parameters[:,-1],[0.01,0.99])-mean_S_parameters
        parameter_range = parameter_lim[1]-parameter_lim[0]
        #pressure_lim  = np.quantile(pressures[:,-1], [0.01,0.99])
        pressure_range = pressure_lim[1]-pressure_lim[0]

        # Calculate Contours for the surface pressure
        Z,X,Y=np.histogram2d(parameters[:,-1]-mean_S_parameters,np.log10(pressures[:,-1]),bins=55,
                        range = [[parameter_lim[0]-0.1*parameter_range,parameter_lim[1]+0.1*parameter_range],[pressure_lim[0]-0.1*pressure_range,pressure_lim[1]+0.1*pressure_range]])
        Z = sp.ndimage.filters.gaussian_filter(Z, [2.9,2.9], mode='constant')
        map, norm, levels = generate_color_map_from_levels(Z,color_levels,level_thresholds)
        contour = plt.contour((X[:-1]+X[1:])/2,10**((Y[:-1]+Y[1:])/2),Z.T,levels=np.array(levels)/3.0,alpha=0,zorder=0).allsegs[:-1]
        pressure_max = np.max(contour[0][0][:,1])
        if ax is None:
            plt.clf()

        # Iterate over all contours and merge the surface with the perofile contour
        for i in range(len(contour)):
            # Calculate the distance between the contour and the P-T profile quantiles
            dist  = sp.spatial.distance.cdist(np.array([contour[i][0][:,0]/1000,(np.log10(contour[i][0][:,1])+6)/10]).T,
                                                np.array([parameter_layers_quantiles[-(i+1)]/1000,(np.log10(pressure_layers_quantiles[-(i+1)])+6)/10]).T)
            dist2 = sp.spatial.distance.cdist(np.array([contour[i][0][:,0]/1000,(np.log10(contour[i][0][:,1])+6)/10]).T,
                                                np.array([parameter_layers_quantiles[i]/1000,(np.log10(pressure_layers_quantiles[i])+6)/10]).T)

            # Find the points of minimal distance on the contour (use 6 points to get bot minimas)
            num = 6
            s  = np.shape(dist)
            s2 = np.shape(dist2)
            ind  = np.array([[i//s[1] ,i%s[1] ] for i in np.argsort(dist , axis=None)[:num]])
            ind2 = np.array([[i//s2[1],i%s2[1]] for i in np.argsort(dist2, axis=None)[:num]])
            ind  = [ind[np.argmax(pressure_layers_quantiles[-(i+1)][ind[:,1]])],ind[np.argmin(pressure_layers_quantiles[-(i+1)][ind[:,1]])]]
            ind2 = [ind2[np.argmax(pressure_layers_quantiles[i][ind2[:,1]])],   ind2[np.argmin(pressure_layers_quantiles[i][ind2[:,1]])]]

            # Save the segments of the contours for later plotting
            pressure_layers_bottom[i] = contour[i][0][ind[0][0]:ind2[0][0],1]
            parameter_layers_bottom[i] = contour[i][0][ind[0][0]:ind2[0][0],0]

            # Reject P-T quantiles with pressures higher than the surface pressure
            pressure_layers_quantiles[-(i+1)] = pressure_layers_quantiles[-(i+1)][:ind[0][1]]
            parameter_layers_quantiles[-(i+1)]= parameter_layers_quantiles[-(i+1)][:ind[0][1]]
            pressure_layers_quantiles[i]      = pressure_layers_quantiles[i][:ind2[0][1]]
            parameter_layers_quantiles[i]     = parameter_layers_quantiles[i][:ind2[0][1]]

    # Generate the quantile contours of the profiles
    pressure_contours = []
    parameter_contours = []
    for i in range(N_levels):
        pressure_contours +=  [np.append(np.append(pressure_layers_quantiles[i],
                                                   pressure_layers_bottom[i][::-1]),
                                         np.flip(pressure_layers_quantiles[-i-1]))]
        parameter_contours += [sp.ndimage.filters.gaussian_filter1d(np.append(np.append(parameter_layers_quantiles[i],
                                                                                        parameter_layers_bottom[i][::-1]),
                                                                    np.flip(parameter_layers_quantiles[-i-1])), smoothing, mode='nearest')]

    # Return the calculated contour range
    return pressure_contours,parameter_contours,pressure_max



def calculate_profile_contours_new(data_extrapolated,
                                   pressures_extrapolated,
                                   pressures,
                                   volume_percentages = [0.30,0.50,0.70,0.90],
                                   smoothing = 3,
                                   data_lim = None,
                                   log_data = True,
                                   log_pressure = True,
                                   bins_data = None):
    
    # Define the log_switcher
    log_switcher = {True: (lambda x: np.log10(x)),
                    False: (lambda x: x)}
    inverse_log_switcher = {True: (lambda x: 10**x),
                            False: (lambda x: x)}
    
    # If needed get the limits of the curves
    if data_lim is None:
        data_lim = [max(np.min(data_extrapolated),1e-50),
                    min(np.max(data_extrapolated),1e50)]
    data_lim = log_switcher[log_data](data_lim)
    pressure_lim = log_switcher[log_pressure]([max(np.min(pressures_extrapolated),1e-50),
                                               min(np.max(pressures_extrapolated),1e50)])

    # Get the sizes of different arrays and copy necessary arrays
    n_profiles = len(pressures_extrapolated[:,0])
    n_layers = len(pressures_extrapolated[0,:])
    n_levels = len(volume_percentages)
    data_extrapolated = copy.deepcopy(data_extrapolated)
    if bins_data is None:
        bins_data = n_layers

    # Set all values below the retrieved ground pressure to np.nan
    # TODO: Only do this if clouds are not opaque
    for i in range(n_profiles):
        data_extrapolated[i,np.where(pressures[i,-1]<pressures_extrapolated[i,:])] = np.nan
    n_true_nan_layer = np.sum((np.sum(np.isnan(data_extrapolated),axis = 0)).reshape(n_layers,1),axis = 1)
    
    # Set all values outside of boundaries to nan
    data_extrapolated[np.where(log_switcher[log_data](data_extrapolated) < data_lim[0])] = np.nan
    data_extrapolated[np.where(log_switcher[log_data](data_extrapolated) > data_lim[1])] = np.nan
    data_extrapolated[np.where(log_switcher[log_pressure](pressures_extrapolated) < pressure_lim[0])] = np.nan
    data_extrapolated[np.where(log_switcher[log_pressure](pressures_extrapolated) > pressure_lim[1])] = np.nan
    edge_nan = np.sum((np.sum(np.isnan(data_extrapolated),axis = 0)).reshape(n_layers,1),axis = 1) - n_true_nan_layer

    # Calculate the volume scaling factor (i.e. the percentage of profiles 
    # that have non-nan point in a given atmosphere layer)
    volume_scaling_factor = (n_profiles-n_true_nan_layer-edge_nan)/(n_profiles-edge_nan)
    
    # Calculate the 2d histogram of the profiles
    histogram, data_edges, pressure_edges = np.histogram2d(log_switcher[log_data](data_extrapolated).flatten(),
                                                           log_switcher[log_pressure](pressures_extrapolated).flatten(),
                                                           bins=[bins_data,n_layers],
                                                           range=[data_lim,pressure_lim],)
    data_centres = (data_edges[:-1]+data_edges[1:])/2
    pressure_centres = (pressure_edges[:-1]+pressure_edges[1:])/2
    smoothed_histogram = sp.ndimage.gaussian_filter(histogram,(smoothing,smoothing))
    sorted_histogram = np.sort(smoothed_histogram,axis = 0)

    # Define the listss to store the calculated percentage envelopes
    percentage_envelope_data = [[] for _ in range(2*n_levels)]
    percentage_envelope_pressure = [[] for _ in range(2*n_levels)]

    # Iterate over all pressure layers of the posteerior
    for layer in range(n_layers):
        sorted_layer = sorted_histogram[::-1,layer]
        volume_layer = np.sum(smoothed_histogram[:,layer])

        # Calculate the levels at which contain 
        # percentages of posterior volume
        percentage_ind = 0
        percentage_levels  = []
        posterior_volume = 0
        for ind in range(len(sorted_layer)):
            if posterior_volume + sorted_layer[ind] >= volume_percentages[percentage_ind]*volume_layer:
                percentage_levels += [(sorted_layer[ind]+sorted_layer[ind+1])/2]
                percentage_ind += 1
                if percentage_ind == n_levels:
                    break
            posterior_volume += sorted_layer[ind]

        # Find all bins that lie above the percentage layers
        for level in range(n_levels):
            # Find where the layer histogram switches from above to below the
            # level by seaching for a change in the sign
            crossings = np.argwhere(np.diff(np.sign(percentage_levels[level]
                                                    -volume_scaling_factor[layer]
                                                    *smoothed_histogram[:,layer]
                                                    ))).flatten()
            
            # If ther is at least one such intersection store the point
            if len(crossings)>=1:
                if len(crossings)==1:
                    percentage_envelope_data[level] += [data_centres[0]]
                else:
                    percentage_envelope_data[level] += [data_centres[crossings[0]]]
                percentage_envelope_data[-(level+1)] += [data_centres[crossings[-1]]]
                percentage_envelope_pressure[level] += [pressure_centres[layer]]
                percentage_envelope_pressure[-(level+1)] += [pressure_centres[layer]]

    # Calculate the contours and return
    contours = {}
    for i in range(n_levels):
        contours[str(np.round(volume_percentages[i],3))] = {'data_values':inverse_log_switcher[log_data](sp.ndimage.gaussian_filter(percentage_envelope_data[i]+percentage_envelope_data[-(i+1)][::-1],smoothing)),
                                                            'pressure_values':inverse_log_switcher[log_pressure](sp.ndimage.gaussian_filter(percentage_envelope_pressure[i]+percentage_envelope_pressure[-(i+1)][::-1],smoothing))}
    return contours