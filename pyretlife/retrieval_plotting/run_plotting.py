"""
This module contains the `RetrievalPlottingObject` class, which is the main
class used to generate plots of the pyretlife retrievals.
"""

__author__ = "Konrad, Alei, Molliere, Quanz"
__copyright__ = "Copyright 2022, Konrad, Alei, Molliere, Quanz"
__maintainer__ = "Björn S. Konrad, Eleonora Alei"
__email__ = "konradb@ethz.ch, elalei@phys.ethz.ch"
__status__ = "Development"

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import importlib
import json
import os
import sys
from typing import Union
from pathlib import Path
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pymultinest as nest
import spectres as spectres
import pandas as pd
import time as t
import astropy.units as u
import pickle
import contextlib

from pyretlife.retrieval.atmospheric_variables import (
    calculate_gravity,
    calculate_polynomial_profile,
    calculate_vae_profile,
    calculate_guillot_profile,
    calculate_isothermal_profile,
    calculate_madhuseager_profile,
    calculate_mod_madhuseager_profile,
    calculate_abundances,
    assign_cloud_parameters,
    calc_mmw,
)
from pyretlife.retrieval.configuration_ingestion import (
    read_config_file,
    check_if_configs_match,
    populate_dictionaries,
    make_output_folder,
    load_data,
    get_check_opacity_path,
    get_check_prt_path,
    get_retrieval_path,
    set_prt_opacity,
)
from pyretlife.retrieval.likelihood_validation import (
    validate_pt_profile,
    validate_cube_finite,
    validate_positive_temperatures,
    validate_sum_of_abundances,
    validate_spectrum_goodness,
)
from pyretlife.retrieval.priors import assign_priors
from pyretlife.retrieval.radiative_transfer import (
    define_linelists,
    calculate_moon_flux,
    assign_reflectance_emissivity,
    calculate_emission_flux,
    scale_flux_to_distance,
    rebin_spectrum,
)
from pyretlife.retrieval.units import (UnitsUtil,
                                       convert_spectrum,
                                       convert_knowns_and_parameters,
                                       )

from pyretlife.retrieval.run import RetrievalObject

from pyretlife.retrieval_plotting.parallel_computation import (
    parallel,
)
from pyretlife.retrieval_plotting.calculate_secondary_quantities import(
    parallel_pt_profile_calculation,
    parallel_spectrum_calculation,
    bond_albedo_calculation,
    add_secondary_parameter_to_parameters
)

from pyretlife.retrieval_plotting.color_handling import (
    generate_quantile_color_levels,
    generate_color_map_from_levels,
)

from pyretlife.retrieval_plotting.custom_matplotlib_handles import (
    MulticolorPatch,
    MulticolorPatchHandler,
    Handles
)
from pyretlife.retrieval_plotting.posterior_plotting import (
    Generate_Parameter_Titles,
    Scale_Posteriors,
    Posterior_Plot,
    Corner_Plot
)
from pyretlife.retrieval_plotting.inlay_plot import (
    add_inlay_plot,
    add_inlay_plot_labels
)

# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------
# Create a new class that inherits functions from globals

class retrieval_plotting_object(RetrievalObject):
    """
    This class binds together all the different parts of the retrieval.

    Args:
        run_retrieval:

    Attributes:
        config: The configuration (i.e., the contents of the YAML or
            INI file for a given retrieval) as a dictionary.
        input_prt_path: Path to the petitRADTRANS installation.
        input_opacity_path: Path to the o
    """

    def __init__(self,results_directory: Union[Path, str]):
        """
        This function reads the input.ini file as well as the retrieval
        results files of interest to us and stores the read in data
        in order to generate the retrieval plots of interest to us.
        """

        # Execute RetrievalObject initialization
        super().__init__(run_retrieval=False,)

        # Store additional required constructor arguments
        self.results_directory = results_directory
        self.posteriors = None
        self.log_evidence = None

        # Generate Directory to store the plots
        if not os.path.exists(self.results_directory + "/Plots_New/"):
            os.makedirs(self.results_directory + "/Plots_New/")

        # Read the config file and populate the dictionary
        self.load_configuration(config_file=self.results_directory+'/input_new.yaml')
        self.unit_conversion()
        self.assign_knowns()
        self.load_posteriors()


        truths = np.array([[self.parameters[parameter]['truth'] for parameter in self.parameters.keys()]])



    def load_posteriors(self):
        # Load the posterior distribution with pymultinest and save it to a pandas DataFrame
        data = nest.Analyzer(len(self.parameters.keys()),outputfiles_basename = self.results_directory)
        posterior_parameter_names = list(self.parameters.keys())+['likelihood']

        self.posteriors = pd.DataFrame(data.get_equal_weighted_posterior(),columns=posterior_parameter_names)

        #pd.set_option('display.max_columns', 500)
        #print(self.posteriors)
        self.log_evidence = [data.get_stats()['global evidence'],data.get_stats()['global evidence error']]





    """
    #################################################################################
    #                                                                               #
    #   Routines for calculating spectra and PT profiles form posteriors.           #
    #                                                                               #
    #################################################################################
    """

    

    def calculate_posterior_spectrum(self,skip=1,n_processes=50,reevaluate_spectra=False):
        '''
        gets the spectra corresponding to the parameter values
        of the equal weighted posteriors.
        '''

        # If not yet done calculate the data corresponding
        # to the retrieved posterior distributions of partameters
        if not hasattr(self, 'retrieved_fluxes'):
            function_args = {'parameter_samples':self.posteriors.to_numpy()[:,:-1],'skip':skip,'n_processes':n_processes}
            self.__evaluate_posteriors(data_type='Spec',data_name='spectra',function_name='parallel_spectrum_calculation',function_args=function_args,force_evaluate=reevaluate_spectra)



    def calculate_true_spectrum(self):
        '''
        calculates the spectrum from the provided true values
        '''
                
        # If not yet done calculate the data corresponding
        # to the retrieved posterior distributions of partameters
        if not hasattr(self, 'true_fluxes'):
            truths = np.array([[self.parameters[parameter]['truth'] for parameter in self.parameters.keys()]])

            try:
                print('Calculating spectrum from provided true values.')
                with contextlib.redirect_stdout(None):
                    spectrum = parallel_spectrum_calculation(self,truths,use_truth=True)
                print('Spectrum successfully calculated from true values.\n')
                for key in spectrum.keys():
                    setattr(self,'true_'+key,spectrum[key])
            except:
                print('Warning: Error when calcuculating spectrum due to missing true values.\n')



    def calculate_posterior_pt_profile(self,skip=1,n_processes=50,reevaluate_PT=False,layers=500,p_surf=4):
        '''
        gets the PT profiles corresponding to the parameter values
        of the equal weighted posteriors.
        '''

        # If not yet done calculate the data corresponding
        # to the retrieved posterior distributions of partameters
        if not hasattr(self, 'retrieved_pressures'):
            function_args = {'parameter_samples':self.posteriors.to_numpy()[:,:-1],'skip':skip,'layers':layers,'p_surf':p_surf,'n_processes':n_processes}
            self.__evaluate_posteriors(data_type='PT',data_name='PT profiles',function_name='parallel_pt_profile_calculation',function_args=function_args,force_evaluate=reevaluate_PT)



    def calculate_true_pt_profile(self,layers=500,p_surf=4):
        '''
        calculates the PT profile from the provided true values
        '''

        # If not yet done calculate the data corresponding
        # to the retrieved posterior distributions of partameters
        if not hasattr(self, 'true_pressures'):
            truths = np.array([[self.parameters[parameter]['truth'] for parameter in self.parameters.keys()]])
            
            try:
                print('Calculating PT profile from provided true values.')
                with contextlib.redirect_stdout(None):
                    pt_profile = parallel_pt_profile_calculation(self,truths,layers=layers,p_surf=p_surf,use_truth=True)
                print('PT profile successfully calculated from true values.\n')
                for key in pt_profile.keys():
                    setattr(self,'true_'+key,pt_profile[key])
            except:
                print('Warning: Error when calcuculating PT profile due to missing true values.\n')



    def __evaluate_posteriors(self,data_type,data_name,function_name,function_args,force_evaluate=False):
        '''
        gets the data corresponding to the parameter values
        of the equal weighted posteriors.
        '''

        # check if the data for the specified skip
        # values are already calculated
        try:
            if force_evaluate:
                raise ValueError('Forced recalculation of ' + data_type + '.')

            # Try loading previously calculated data
            load_file = open(self.results_directory+'Plots_New/Ret_'+data_type+'_Skip_'+str(function_args['skip'])+'.pkl', "rb")
            loaded_data = pickle.load(load_file)
            load_file.close()

            # Initialize class attributes for the data from the different processes
            for key in loaded_data.keys():
                setattr(self,key,loaded_data[key])

            print('Loaded previously calculated '+data_name+' from '+self.results_directory+'Plots_New/Ret_'+data_type+'_Skip_'+str(function_args['skip'])+'.pkl.\n')

        # If not calculated or the revaluation is desired
        # Calculate from scratch
        except:
            print('Calculating retrieved '+data_name+' from scratch.')

            # Check that we do not use too many CPUs
            if (np.shape(self.posteriors)[0]//function_args['skip'])//function_args['n_processes'] < 3:
                print('Not enough jobs for the specified number of processes!')
                while (np.shape(self.posteriors)[0]//function_args['skip'])//function_args['n_processes'] < 3:
                    function_args['n_processes'] -= 1
                print('I lowered n_processes to '+str(function_args['n_processes'])+'.')

            # Start the paralel calculation
            parallel_calculation = parallel(function_args['n_processes'])
            result_process = parallel_calculation.calculate(self.results_directory,function_name,function_args)

            # Combine the data from the different processes
            print('Combining all data.')
            result_combined = {}
            for key in result_process[0].keys():
                combined = result_process[0][key].copy()
                for process in range(1,function_args['n_processes']):
                    try:
                        combined = np.append(combined,result_process[process][key],axis = 0)
                    except:
                        pass
                result_combined['retrieved_'+key] = combined

                # Initialize class attributes for the data from the different processes
                setattr(self,'retrieved_'+key,combined)
            print('Done combining all data.')

            # Save the calculated data in a pickle file for later reloading to save time
            save_file = open(self.results_directory+'Plots_New/Ret_'+data_type+'_Skip_'+str(function_args['skip'])+'.pkl', "wb")
            pickle.dump(result_combined, save_file, protocol=4)
            save_file.close()
            print('Saved calculated '+data_name+' in '+self.results_directory+'Plots_New/Ret_'+data_type+'_Skip_'+str(function_args['skip'])+'.pkl.\n')



    def deduce_bond_albedo(self,stellar_luminosity,error_stellar_luminosity,planet_star_separation,error_planet_star_separation,true_equilibrium_temperature = None,true_bond_albedo = None, reevaluate_bond_albedo=False):

        # check if the data for the specified skip
        # values are already calculated
        try:
            if reevaluate_bond_albedo:
                raise ValueError('Forced recalculation of Bond albedos.')

            # Try loading previously calculated data
            load_file = open(self.results_directory+'Plots_New/Ret_Bond_Albedo.pkl', "rb")
            results = pickle.load(load_file)
            load_file.close()

            print('Loaded previously calculated Bond albedos from '+self.results_directory+'Plots_New/Ret_bond_albedo.pkl.\n')

        # If not calculated or the revaluation is desired
        # Calculate from scratch
        except:
            print('Calculating retrieved Bond albedos from scratch.')

            # Start the calculation
            equilibrium_temperatures, bond_albedos = bond_albedo_calculation(self,stellar_luminosity,error_stellar_luminosity,planet_star_separation,error_planet_star_separation)
            results = {'T_eq':equilibrium_temperatures,'A_b':bond_albedos}

            # Save the calculated data in a pickle file for later reloading to save time
            save_file = open(self.results_directory+'Plots_New/Ret_Bond_Albedo.pkl', "wb")
            pickle.dump(results, save_file, protocol=4)
            save_file.close()
            print('Saved calculated Bond albedos in '+self.results_directory+'Plots_New/Ret_Bond_Albedo.pkl.\n')

        # add the bond albedo to the parameters and the posterior
        self.posteriors['T_eq'], self.posteriors['A_b'] = results['T_eq'], results['A_b'] 
        self.parameters = add_secondary_parameter_to_parameters(self.parameters,name='T_eq',unit=u.K,title='$\mathrm{T_{eq,\,Planet}}$',true_value=true_equilibrium_temperature)
        self.parameters = add_secondary_parameter_to_parameters(self.parameters,name='A_b',unit=u.dimensionless_unscaled,title='$\mathrm{A_{B,\,Planet}}$',true_value=true_bond_albedo)



    def deduce_gravity(self, true_gravity = None):

        # take the surface gravityfrom the spectrum calculation
        self.posteriors['g'] = self.retrieved_gravity

        # add the bond albedo to the parameters
        self.parameters = add_secondary_parameter_to_parameters(self.parameters,name='g',unit=u.cm/u.s/u.s,title='$\mathrm{g}$',true_value=true_gravity)



    def deduce_surface_temperature(self, true_surface_temperature = None):

        # take the temperature in the lowest layer of the retrieved P-T profiles
        self.posteriors['T_0'] = self.retrieved_temperatures[:,-1]

        # add the bond albedo to the parameters
        self.parameters = add_secondary_parameter_to_parameters(self.parameters,name='T_0',unit=u.K,title='$\mathrm{T_{0}}$',true_value=true_surface_temperature)



    """
    #################################################################################
    #                                                                               #
    #   Routines for generating cornerplots.                                        #
    #                                                                               #
    #################################################################################
    """



    def Posteriors(self, save=False, plot_corner=True, log_pressures=True, log_mass=True, log_abundances=True, log_particle_radii=True, plot_pt=True, plot_physparam=True,
                    plot_clouds=True,plot_chemcomp=True,plot_scatt=True,plot_moon=False,plot_secondary_parameters=True, bins=20, quantiles1d=[0.16, 0.5, 0.84],
                    color='k',add_table=False,color_truth='C3',ULU_lim=[-0.15,0.75],parameter_units='input',custom_unit_titles={},custom_parameter_titles={}):
        '''
        This function generates a corner plot for the retrieved parameters.
        '''

        # get the indices of all parameters shown in the corner plot
        parameters_plotted = []
        for parameter in self.parameters:
            if (self.parameters[parameter]['type'] == 'TEMPERATURE PARAMETERS') and plot_pt:
                parameters_plotted += [parameter]
            elif (self.parameters[parameter]['type'] == 'PHYSICAL PARAMETERS') and plot_physparam:
                parameters_plotted += [parameter]
            elif (self.parameters[parameter]['type'] == 'CHEMICAL COMPOSITION PARAMETERS') and plot_chemcomp:        
                parameters_plotted += [parameter]
            elif (self.parameters[parameter]['type'] == 'CLOUD PARAMETERS') and plot_clouds:
                parameters_plotted += [parameter]
            elif (self.parameters[parameter]['type'] == 'SCATTERING PARAMETERS') and plot_scatt:
                parameters_plotted += [parameter]
            elif (self.parameters[parameter]['type'] == 'SECONDARY PARAMETERS') and plot_secondary_parameters:
                parameters_plotted += [parameter]
            elif (self.parameters[parameter]['type'] == 'MOON PARAMETERS') and plot_moon:
                parameters_plotted += [parameter]

        # Copy the relevant data
        local_post = self.posteriors.copy()
        local_truths = {parameter:self.parameters[parameter]['truth'] for parameter in parameters_plotted}

        # Generate the titles
        Generate_Parameter_Titles(self)
        local_titles = {i:self.parameters[i]['title'] for i in parameters_plotted}
        for parameter in parameters_plotted:
            if parameter in custom_parameter_titles:
                local_titles[parameter] = custom_parameter_titles[parameter]

        # Unit conversions for plotting if units=None retrieval units are plotted
        # if units='input' the units in the input.ini file are plotted
        retrieval_unit =  {i:self.parameters[i]['unit'] for i in parameters_plotted}
        if parameter_units == 'input':
            local_units = {i:self.parameters[i]['input_unit'] for i in parameters_plotted}
        else:
            local_units = retrieval_unit.copy()
            for parameter in parameters_plotted:
                if parameter in parameter_units:
                    local_units[parameter] = parameter_units[parameter]

        # Add the units to the titles
        for parameter in parameters_plotted:
            if not f"{local_units[parameter]:latex}" == '$\\mathrm{}$':
                unit = '\\left['+f"{local_units[parameter]:latex}"[1:-1]+'\\right]'
            else:
                unit = ''
            if parameter in custom_unit_titles:
                unit = '\\left['+custom_unit_titles[parameter][1:-1]+'\\right]'
            local_titles[parameter] = local_titles[parameter][:-1]+unit+'$'

        # Convert the units of the posterior and the true value
        for parameter in parameters_plotted:
            local_post[parameter]   = self.units.truth_unit_conversion(parameter,retrieval_unit[parameter],local_units[parameter],local_post[parameter].to_numpy(),printing=False)
            local_truths[parameter] = self.units.truth_unit_conversion(parameter,retrieval_unit[parameter],local_units[parameter],local_truths[parameter],printing=False)

        # Adust the local copy of the posteriors according to the users desires
        local_post, local_truths, local_titles = Scale_Posteriors(self,local_post, local_truths, local_titles, parameters_plotted,
                                                                  log_pressures=log_pressures, log_mass=log_mass,
                                                                  log_abundances=log_abundances, log_particle_radii=log_particle_radii)

        # Check if there were ULU posteriors
        ULU = [parameter for parameter in parameters_plotted if self.parameters[parameter]['prior']['kind'] == 'upper-log-uniform']

        if plot_corner:
            fig, axs = Corner_Plot(parameters_plotted,local_post,local_titles,local_truths,quantiles1d=quantiles1d,bins=bins,color=color,
                                            add_table=add_table,color_truth=color_truth,ULU=ULU if ULU != [] else None,ULU_lim=ULU_lim)
            if save:
                plt.savefig(self.results_directory+'Plots_New/plot_corner.pdf', bbox_inches='tight')
            else:
                return fig, axs
        else:
            if not os.path.exists(self.results_directory + 'Plots_New/Posteriors/'):
                os.makedirs(self.results_directory + 'Plots_New/Posteriors/')
            for parameter in parameters_plotted:
                fig, axs = Posterior_Plot(local_post[parameter],local_titles[parameter],local_truths[parameter],
                                    quantiles1d=quantiles1d,bins=bins,color=color,ULU=(parameter in ULU),ULU_lim=ULU_lim)

                if save:
                    plt.savefig(self.results_directory+'/Plots_New/Posteriors/'+parameter+'.pdf', bbox_inches='tight')
                else:
                    return fig, axs





    """
    #################################################################################
    #                                                                               #
    #   Routine for generating Spectrum plots.                                      #
    #                                                                               #
    #################################################################################
    """



    def plot_retrieved_flux(self,  quantiles = [0.05,0.15,0.25,0.35,0.65,0.75,0.85,0.95],
                    quantiles_title = None, ax = None, color='C2', case_identifier = None, plot_noise = False, plot_true_spectrum = False, plot_datapoints = False,
                    
                    figsize=(12,2),
                    legend_loc = 'best',
                    x_lim = None,
                    y_lim = None,
                    
                    noise_title = 'Observation Noise',  

                    
                    plot_instruments_separately=False,     
                    plot_residual = False,
                    plot_log_wavelength=False,
                    plot_log_flux=False,
                    plot_unit_wavelength=None,
                    plot_unit_flux=None,
                    plot_retrieved_median=False):
        '''
        This Function creates a plot that visualizes the absolute uncertainty on the
        retrieval results in comparison with the input spectrum for the retrieval.
        '''

        # Unit conversions for the x and y scales of the graph
        retrieval_units = {'x_unit':self.units.retrieval_units['wavelength'], 'y_unit':self.units.retrieval_units['flux']}
        plot_units = {'x_unit':retrieval_units['x_unit'] if plot_unit_wavelength is None else plot_unit_wavelength,
                       'y_unit':retrieval_units['y_unit'] if plot_unit_flux is None else plot_unit_flux}
        unit_titles = {i:'$\\left['+f"{plot_units[i]:latex}"[1:-1]+'\\right]$' for i in plot_units}

        # Convert the fluxes to the desired units
        local_instrument = self.instrument.copy()
        for instrument in local_instrument.keys():
            converted_spectrum = self.units.unit_spectrum_conversion('spec',[retrieval_units['x_unit'],retrieval_units['y_unit']],[plot_units['x_unit'],plot_units['y_unit']],
                                            np.array([self.instrument[instrument]['wavelength'],self.instrument[instrument]['flux'],self.instrument[instrument]['error']]).T,printing=False)
            local_instrument[instrument]['wavelength'], local_instrument[instrument]['flux'], local_instrument[instrument]['error'] = converted_spectrum[:, 0], converted_spectrum[:, 1], converted_spectrum[:, 2]
        local_wavelengths, local_fluxes = self.units.unit_spectrum_cube([retrieval_units['x_unit'],retrieval_units['y_unit']],[plot_units['x_unit'],plot_units['y_unit']],
                                        self.retrieved_wavelengths,self.retrieved_fluxes)

        # Find the quantiles for the retrieved spectra
        median_all_wlavelengths = np.quantile(local_fluxes,0.5,axis=0)
        quantiles_all_wlavelengths = [np.quantile(local_fluxes,q,axis=0) for q in quantiles]

        # Define factors depening on wether residual is plotted or not
        factor_input_spectrum = 1 if plot_residual else 0
        factor_percentage = {instrument:(100/local_instrument[instrument]['flux'] if plot_residual else 1) for instrument in local_instrument.keys()}

        # If necessary rebin the quantiles, calculate the residuals, and split up into instruments
        retrieved_instrument_median = {}
        retrieved_instrument_quantiles = {}
        retrieved_instrument_wavelengths = {}
        if plot_instruments_separately or plot_residual:
            for instrument in local_instrument.keys():
                # Rebin the spectrum according to the input spectrum if wavelenths differ strongly
                if not np.array([(np.round(local_instrument[instrument]['wavelength'],10)==np.round(local_wavelengths,10))]).all():
                    retrieved_instrument_median[instrument] = (spectres.spectres(local_instrument[instrument]['wavelength'],local_wavelengths,median_all_wlavelengths)-local_instrument[instrument]['flux']*factor_input_spectrum)*factor_percentage[instrument]
                    retrieved_instrument_quantiles[instrument] = [(spectres.spectres(local_instrument[instrument]['wavelength'],local_wavelengths,quantiles_all_wlavelengths[q])-local_instrument[instrument]['flux']*factor_input_spectrum)*factor_percentage[instrument] for q in range(len(quantiles))]
                else:
                    retrieved_instrument_median[instrument] = (median_all_wlavelengths-local_instrument[instrument]['flux']*factor_input_spectrum)*factor_percentage[instrument]
                    retrieved_instrument_quantiles[instrument] = [(quantiles_all_wlavelengths[q]-local_instrument[instrument]['flux']*factor_input_spectrum)*factor_percentage[instrument] for q in range(len(quantiles))]
                retrieved_instrument_wavelengths[instrument] = local_instrument[instrument]['wavelength']
        else:
            retrieved_instrument_median['full_range'] = median_all_wlavelengths
            retrieved_instrument_quantiles['full_range'] = quantiles_all_wlavelengths
            retrieved_instrument_wavelengths['full_range'] = local_wavelengths

        # Generate colorlevels for the different quantiles
        color_levels, level_thresholds, N_levels = generate_quantile_color_levels(color,quantiles)

        if plot_instruments_separately:
            instrument_plots = [[instrument] for instrument in retrieved_instrument_wavelengths.keys()]
        else:
            instrument_plots = [list(retrieved_instrument_wavelengths.keys())]

        ax_arg = ax

        # Generate plots for different instrument configurations
        for instrument_plot in instrument_plots:

            figure = plt.figure(figsize=figsize)
            ax = figure.add_axes([0.1, 0.1, 0.8, 0.8])
                
            # Plotting the retrieved Spectra
            for instrument in instrument_plot:
                for i in range(N_levels):
                    ax.fill(np.append(retrieved_instrument_wavelengths[instrument],np.flip(retrieved_instrument_wavelengths[instrument])),
                            np.append(retrieved_instrument_quantiles[instrument][i],np.flip(retrieved_instrument_quantiles[instrument][-i-1])),color = tuple(color_levels[i, :]),lw = 0,clip_box=True,zorder=1)
                if plot_retrieved_median:
                    ax.plot(retrieved_instrument_wavelengths[instrument],retrieved_instrument_median[instrument],color=color,lw = 0.5, label = 'Best Fit',zorder=2)
                        
            # Plotting the input spectrum
            input_instruments = local_instrument.keys() if 'full_range' in instrument_plot else instrument_plot

            for instrument in input_instruments:

                # Plot the noise for the input spectrum
                if plot_noise:
                    ax.fill(np.append(local_instrument[instrument]['wavelength'],np.flip(local_instrument[instrument]['wavelength'])),np.append((local_instrument[instrument]['flux']*abs(factor_input_spectrum-1)+local_instrument[instrument]['error'])*factor_percentage[instrument],
                            np.flip((local_instrument[instrument]['flux']*abs(factor_input_spectrum-1)-local_instrument[instrument]['error'])*factor_percentage[instrument])),color = (0.8,0.8,0.8,1),lw = 0,clip_box=True,zorder=0)
                if plot_true_spectrum:
                    label = None if plot_residual else 'Input Spectrum'
                    ls = ':' if plot_residual else '-'
                    lw = 2 if plot_retrieved_median else 1.5
                    ax.plot(local_instrument[instrument]['wavelength'],local_instrument[instrument]['flux']*abs(factor_input_spectrum-1),color = 'black',ls=ls,label=label,lw=lw,zorder=2)
                if plot_datapoints:
                    ax.errorbar(local_instrument[instrument]['wavelength'],local_instrument[instrument]['flux']*abs(factor_input_spectrum-1),yerr=local_instrument[instrument]['error']*factor_percentage[instrument],color = 'k',ms = 3,marker='o',ls='',label = 'Input Spectrum',zorder=2)

            # If it is a single plot show the axes titles
            if ax_arg is None:
                if plot_residual:
                    ax.set_ylabel(r'Residual $\left[\%\right]$')
                else:
                    #ax.set_ylabel(r'Flux at 10 pc $\left[\mathrm{\frac{erg}{s\,Hz\,m^2}}\right]$')
                    ax.set_ylabel('Flux at 10 pc '+unit_titles['y_unit'])
                ax.set_xlabel('Wavelength '+unit_titles['x_unit'])

            # Set the limits for the plot axes and the scaling
            if plot_log_wavelength:
                ax.set_xscale('log')
            if x_lim is None:
                x_lim_loc = [min([retrieved_instrument_wavelengths[instrument][0] for instrument in instrument_plot]),
                             max([retrieved_instrument_wavelengths[instrument][-1] for instrument in instrument_plot])]
            else:
                x_lim_loc = x_lim
            ax.set_xlim(x_lim_loc)

            if plot_log_flux:
                ax.set_yscale('log')
            if y_lim is None:
                if plot_residual:
                    y_lim_loc = [-58,58]
                else:
                    y_lim_loc = [0,list(ax.get_ylim())[1]]
            else:
                y_lim_loc = y_lim
            ax.set_ylim(y_lim_loc)

            # Print the case identifier
            if case_identifier is not None:
                if plot_residual:
                    ax.annotate(case_identifier,[x_lim_loc[1]-0.025*(x_lim_loc[1]-x_lim_loc[0]),y_lim_loc[0]+0.1*(y_lim_loc[1]-y_lim_loc[0])],ha='right',va='bottom',weight='bold')
                else:
                    ax.annotate(case_identifier,[x_lim_loc[1]-0.05*(x_lim_loc[1]-x_lim_loc[0]),y_lim_loc[0]+0.05*(y_lim_loc[1]-y_lim_loc[0])],ha='right',va='bottom',weight='bold')

            # Legend cosmetics (adding custom patches to the legend)
            handles, labels = ax.get_legend_handles_labels()
            patch_handles = []
            patch_labels = []
            patch_handles = [MulticolorPatch([tuple(color_levels[i, :])],[1]) for i in range(N_levels)]
            if quantiles_title is None:
                patch_labels = [str(quantiles[i])+'-'+str(quantiles[-i-1]) for i in range(N_levels)]
            else:
                patch_labels = quantiles_title
            if plot_noise:
                patch_handles = [MulticolorPatch([(0.8,0.8,0.8)],[1])]+patch_handles
                patch_labels = [noise_title]+patch_labels

            # Add the legend
            ncol= len(labels+patch_labels) if plot_residual else 1
            lgd = ax.legend(handles+patch_handles,labels+patch_labels,
                        handler_map={str:  Handles(), MulticolorPatch:  MulticolorPatchHandler()}, ncol=ncol,loc=legend_loc,frameon=False)
            
            # Save or pass back the figure
            if ax_arg is not None:
                return
            else:
                filename = 'full_range' if not plot_instruments_separately else instrument_plot[0]
                if plot_residual:
                    filename = 'residual_' + filename
                plt.savefig(self.results_directory+'Plots_New/plot_spectrum_'+filename+'.pdf', bbox_inches='tight',bbox_extra_artists=(lgd,), transparent=True)
                plt.close()





    """
    #################################################################################
    #                                                                               #
    #   Routines for generating PT profile plots.                                   #
    #                                                                               #
    #################################################################################
    """

    

    def plot_retrieved_pt_profile(self, save=False,  x_lim =[0,1000], y_lim = [1e-6,1e4], quantiles=[0.05,0.15,0.25,0.35,0.65,0.75,0.85,0.95],
                    quantiles_title = None, inlay_loc='upper right', bins_inlay = 20,x_lim_inlay =None, y_lim_inlay = None, figure = None, ax = None, color='C2', case_identifier = '',
                    legend_n_col = 2, legend_loc = 'best',n_processes=50,figsize=(6.4, 4.8),h_cover=0.45,reevaluate_PT = False,
                    
                    true_cloud_top=[None,None],

                    plot_residual = False,
                    plot_clouds = False,
                    plot_unit_temperature=None,
                    plot_unit_pressure=None,):
        '''
        This Function creates a plot that visualizes the absolute uncertainty on the
        retrieval results in comparison with the input PT profile for the retrieval.
        '''

        # Unit conversions for the x and y scales of the graph
        retrieval_units = {'x_unit':u.K, 'y_unit':u.bar}
        local_units = {'x_unit':retrieval_units['x_unit'] if plot_unit_temperature is None else plot_unit_temperature,
                       'y_unit':retrieval_units['y_unit'] if plot_unit_pressure is None else plot_unit_pressure}
        unit_titles = {i:'$\\left['+f"{local_units[i]:latex}"[1:-1]+'\\right]$' for i in local_units}

        # Convert the units of the P-T profile posteriors and the true value
        local_retrieved_pressures_extrapolated    = self.units.truth_unit_conversion('Pressure',retrieval_units['y_unit'],local_units['y_unit'],self.retrieved_pressures_extrapolated,printing=False)
        local_retrieved_temperatures_extrapolated = self.units.truth_unit_conversion('Temperature',retrieval_units['x_unit'],local_units['x_unit'],self.retrieved_temperatures_extrapolated,printing=False)
        local_retrieved_pressures    = self.units.truth_unit_conversion('Pressure',retrieval_units['y_unit'],local_units['y_unit'],self.retrieved_pressures,printing=False)
        local_retrieved_temperatures = self.units.truth_unit_conversion('Temperature',retrieval_units['x_unit'],local_units['x_unit'],self.retrieved_temperatures,printing=False)
        if self.settings['include_scattering']['clouds'] == True: #self.settings['clouds'] == 'opaque':
            local_retrieved_pressures_cloud_top    = self.units.truth_unit_conversion('Pressure',retrieval_units['y_unit'],local_units['y_unit'],self.retrieved_pressures_cloud_top,printing=False)
            local_retrieved_temperatures_cloud_top = self.units.truth_unit_conversion('Temperature',retrieval_units['x_unit'],local_units['x_unit'],self.retrieved_temperatures_cloud_top,printing=False)
        if hasattr(self, 'true_pressures'):
            local_true_pressures    = self.units.truth_unit_conversion('Pressure',retrieval_units['y_unit'],local_units['y_unit'],self.true_pressures,printing=False)[0]
            local_true_temperatures = self.units.truth_unit_conversion('Temperature',retrieval_units['x_unit'],local_units['x_unit'],self.true_temperatures,printing=False)[0]
            if self.settings['include_scattering']['clouds'] == True: #self.settings['clouds'] == 'opaque':
                local_true_pressures_cloud_top    = self.units.truth_unit_conversion('Pressure',retrieval_units['y_unit'],local_units['y_unit'],self.true_pressures_cloud_top,printing=False)[0]
                local_true_temperatures_cloud_top = self.units.truth_unit_conversion('Temperature',retrieval_units['x_unit'],local_units['x_unit'],self.true_temperatures_cloud_top,printing=False)[0]
        else:
            local_true_temperatures_cloud_top,local_true_pressures_cloud_top = true_cloud_top[1],true_cloud_top[0]

        # find the quantiles for the different pressures and temperatures
        p_layers_quantiles = [np.nanquantile(local_retrieved_pressures_extrapolated,q,axis=0) for q in quantiles]
        if plot_residual:
            T_layers_quantiles = [np.nanquantile(local_retrieved_temperatures_extrapolated,q,axis=0)-np.nanquantile(local_retrieved_temperatures_extrapolated,0.5,axis=0) for q in quantiles]
        else:
            T_layers_quantiles = [np.nanquantile(local_retrieved_temperatures_extrapolated,q,axis=0) for q in quantiles]

        # Merge the P-T profile quantiles with the surface pressure if retrieved
        p_max = 1e6
        p_layers_bottom = len(quantiles)//2*[[]]
        T_layers_bottom = len(quantiles)//2*[[]]
        if not self.settings['include_scattering']['clouds'] == True: #self.settings['clouds'] == 'opaque':

            if plot_residual:
                mean_S_T = np.median(local_retrieved_temperatures[:,-1])
            else:
                mean_S_T = 0

            # Define limits and make a 2d histogram of the surface pressures and temperatures
            t_lim = [np.min(local_retrieved_temperatures[:,-1])-mean_S_T,np.max(local_retrieved_temperatures[:,-1])-mean_S_T]
            t_range = t_lim[1]-t_lim[0]
            p_lim = [np.min(np.log10(local_retrieved_pressures[:,-1])),np.max(np.log10(local_retrieved_pressures[:,-1]))]
            p_range = p_lim[1]-p_lim[0]

            # Calculate Contours for the surface pressure
            Z,X,Y=np.histogram2d(local_retrieved_temperatures[:,-1]-mean_S_T,np.log10(local_retrieved_pressures[:,-1]),bins=100,
                            range = [[t_lim[0]-0.1*t_range,t_lim[1]+0.1*t_range],[p_lim[0]-0.1*p_range,p_lim[1]+0.1*p_range]])
            Z = sp.ndimage.filters.gaussian_filter(Z, [7,7], mode='reflect')
            color_levels, level_thresholds, N_levels = generate_quantile_color_levels(color,quantiles)
            map, norm, levels = generate_color_map_from_levels(Z,color_levels,level_thresholds)
            contour = plt.contour((X[:-1]+X[1:])/2,10**((Y[:-1]+Y[1:])/2),Z.T,levels=np.array(levels),alpha=1,zorder=2).allsegs[:-1]
            p_max = np.max(contour[0][0][:,1])

            # iterate over all contours
            for i in range(len(contour)):
                # Calculate the distance between the contour and the P-T profile quantiles
                dist  = sp.spatial.distance.cdist(np.array([contour[i][0][:,0]/1000,(np.log10(contour[i][0][:,1])+6)/10]).T,
                                                    np.array([T_layers_quantiles[-(i+1)]/1000,(np.log10(p_layers_quantiles[-(i+1)])+6)/10]).T)
                dist2 = sp.spatial.distance.cdist(np.array([contour[i][0][:,0]/1000,(np.log10(contour[i][0][:,1])+6)/10]).T,
                                                    np.array([T_layers_quantiles[i]/1000,(np.log10(p_layers_quantiles[i])+6)/10]).T)

                # Find the points of minimal distance on the contour (use 6 points to get bot minimas)
                num = 6
                s  = np.shape(dist)
                s2 = np.shape(dist2)
                ind  = np.array([[i//s[1] ,i%s[1] ] for i in np.argsort(dist , axis=None)[:num]])
                ind2 = np.array([[i//s2[1],i%s2[1]] for i in np.argsort(dist2, axis=None)[:num]])
                ind  = [ind[np.argmax(p_layers_quantiles[-(i+1)][ind[:,1]])],ind[np.argmin(p_layers_quantiles[-(i+1)][ind[:,1]])]]
                ind2 = [ind2[np.argmax(p_layers_quantiles[i][ind2[:,1]])],   ind2[np.argmin(p_layers_quantiles[i][ind2[:,1]])]]

                # Save the segments of the contours for later plotting
                p_layers_bottom[i] = contour[i][0][ind[0][0]:ind2[0][0],1]
                T_layers_bottom[i] = contour[i][0][ind[0][0]:ind2[0][0],0]

                # Reject P-T quantiles with pressures higher than the surface pressure
                p_layers_quantiles[-(i+1)]= p_layers_quantiles[-(i+1)][:ind[0][1]]
                T_layers_quantiles[-(i+1)]= T_layers_quantiles[-(i+1)][:ind[0][1]]
                p_layers_quantiles[i]     = p_layers_quantiles[i][:ind2[0][1]]
                T_layers_quantiles[i]     = T_layers_quantiles[i][:ind2[0][1]]
            
        # If wanted find the quantiles for cloud top and bottom pressures
        if plot_clouds:
            cloud_top_quantiles = [np.quantile(local_retrieved_pressures_cloud_top,q) for q in quantiles]

        # Generate colorlevels for the different quantiles
        color_levels, level_thresholds, N_levels = generate_quantile_color_levels(color,quantiles)
        color_levels_c, level_thresholds_c, N_levels_c = generate_quantile_color_levels('#898989',quantiles)

        # Start of the plotting
        ax_arg = ax
        if ax is None:
            figure = plt.figure(figsize=figsize)
            ax = figure.add_axes([0.1, 0.1, 0.8, 0.8])
        else:
            pass

        # If wanted: plotting the retrieved cloud top
        if plot_clouds:
            for i in range(N_levels_c):
                ax.fill([-10000,10000,10000,-10000],[cloud_top_quantiles[i],cloud_top_quantiles[i],cloud_top_quantiles[-i-1],cloud_top_quantiles[-i-1]],color = tuple(color_levels_c[i, :]),clip_box=True,zorder=-1)
            for i in range(N_levels_c):
                ax.hlines([cloud_top_quantiles[i],cloud_top_quantiles[-i-1]],xmin = -10000, xmax = 10000,color = tuple(color_levels_c[i, :]),ls='-',zorder=0)

        # Plotting the retrieved PT profile
        for i in range(N_levels):
            if self.settings['include_scattering']['clouds'] == True: #self.settings['clouds'] == 'opaque':
                ax.fill(np.append(np.append(sp.ndimage.filters.gaussian_filter1d(T_layers_quantiles[i], 5, mode='nearest'),T_layers_bottom[i][::-1]),np.flip(sp.ndimage.filters.gaussian_filter1d(T_layers_quantiles[-i-1], 5, mode='nearest'))),
                        np.append(np.append(p_layers_quantiles[i],p_layers_bottom[i][::-1]),np.flip(p_layers_quantiles[-i-1])),color = tuple(color_levels[i, :]),lw = 0,clip_box=True,zorder=1)
            else:
                ax.fill(sp.ndimage.filters.gaussian_filter1d(np.append(np.append(T_layers_quantiles[i],T_layers_bottom[i][::-1]),np.flip(T_layers_quantiles[-i-1])), 10, mode='nearest'),
                        np.append(np.append(p_layers_quantiles[i],p_layers_bottom[i][::-1]),np.flip(p_layers_quantiles[-i-1])),color = tuple(color_levels[i, :]),lw = 0,clip_box=True,zorder=1)
        if plot_residual:
            ax.semilogy([0,0], y_lim,color ='black', linestyle=':')
            ax.annotate('Retrieved\nP-T Median',[0+0.035*x_lim[1],10**(0.975*(np.log10(y_lim[1])-np.log10(y_lim[0]))+np.log10(y_lim[0]))],color = 'black',rotation=0,ha='left')

        # If wanted: plotting the retrieved cloud top
        if plot_clouds:
            for i in range(N_levels_c):
                ax.hlines([cloud_top_quantiles[i],cloud_top_quantiles[-i-1]],xmin = -10000, xmax = 10000,color = tuple(color_levels_c[i, :]),ls=':',zorder=2)

        # Plotting the true/input profile (interpolation for smoothing)
        if plot_residual:
            y = np.nanquantile(local_retrieved_temperatures_extrapolated,0.5,axis=0)
            x = np.nanquantile(local_retrieved_pressures_extrapolated,0.5,axis=0)
            yinterp = np.interp(local_true_pressures, x, y)
            smooth_T_true = sp.ndimage.filters.gaussian_filter1d(local_true_temperatures-yinterp,sigma = 5)
            smooth_T_true[np.where(local_true_pressures>p_max)]=np.nan

            # Check if the retrieved PT profile reaches al the way to the true surface and plot accordingly.
            if np.isnan(smooth_T_true[-10]):
                num_nan = np.count_nonzero(np.isnan(smooth_T_true))
                ax.semilogy(smooth_T_true[:-num_nan-30],local_true_pressures[:-num_nan-30],color ='black', label = 'P-T Profile')
                ax.semilogy(smooth_T_true[-num_nan-30:],local_true_pressures[-num_nan-30:],color ='black', ls = ':')
            else:
                ax.semilogy(smooth_T_true,local_true_pressures,color ='black', label = 'P-T Profile')

                # Plotting the true/input surface temperature/pressure
                ax.plot(local_true_temperatures[-1]-yinterp[-1],local_true_pressures[-1],marker='s',color='C3',ms=7, markeredgecolor='black',lw=0,label = 'Surface')

            # If wanted: plotting the true/input cloud top temperature/pressure
            try:
                ind_ct = (np.argmin(np.abs(np.log10(local_true_pressures_cloud_top)-np.log10(local_true_pressures))))
                ax.plot(smooth_T_true[ind_ct],local_true_pressures_cloud_top,marker='o',color='C1',lw=0,ms=7, markeredgecolor='black',label = 'Cloud-Top')
            except:
                pass
        else:
            ax.semilogy(local_true_temperatures,local_true_pressures,color ='black', label = 'P-T Profile')

            # Plotting the true/input surface temperature/pressure
            ax.plot(local_true_temperatures[-1],local_true_pressures[-1],marker='s',color='C3',ms=7, markeredgecolor='black',lw=0,label = 'Surface')

            # If wanted: plotting the true/input cloud top temperature/pressure
            try:
                ax.plot(local_true_temperatures_cloud_top,local_true_pressures_cloud_top,marker='o',color='C1',lw=0,ms=7, markeredgecolor='black',label = 'Cloud-Top')
            except:
                pass

        # If it is a single plot show the axes titles
        if ax_arg is None:
            if plot_residual:
                ax.set_xlabel('Difference to Retrieved Median '+unit_titles['x_unit'])
            else:
                ax.set_xlabel('Temperature '+unit_titles['x_unit'])
            ax.set_ylabel('Pressure '+unit_titles['y_unit'])
        
        # Set the limits for the plot axes
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.invert_yaxis()

        # Inlay plot
        # generate and position the inlay plot
        ax2 = add_inlay_plot(inlay_loc,figure,ax_arg,ax,h_cover=h_cover)

        # Plotting the cloud top temperature/pressure
        if self.settings['include_scattering']['clouds'] == True: #self.settings['clouds'] == 'opaque':
            # Define the plot titles
            ax2_xlabel = '$T^\mathrm{cloud}_\mathrm{top}$ '+unit_titles['x_unit']
            ax2_ylabel = '$P^\mathrm{cloud}_\mathrm{top}$ '+unit_titles['y_unit']

            # Define limits and make a 2d histogram of the cloud top pressures and temperatures
            t_lim = [np.min(local_retrieved_temperatures_cloud_top),np.max(local_retrieved_temperatures_cloud_top)]
            t_range = t_lim[1]-t_lim[0]
            p_lim = [np.min(np.log10(local_retrieved_pressures_cloud_top)),np.max(np.log10(local_retrieved_pressures_cloud_top))]
            p_range = p_lim[1]-p_lim[0]
            Z,X,Y=np.histogram2d(local_retrieved_temperatures_cloud_top[:,0],np.log10(local_retrieved_pressures_cloud_top)[:,0],bins=bins_inlay,
                range = [[t_lim[0]-0.1*t_range,t_lim[1]+0.1*t_range],[p_lim[0]-0.1*p_range,p_lim[1]+0.1*p_range]])

        else:
            # Define the plot titles
            ax2_xlabel = '$\mathrm{T_0}$ '+unit_titles['x_unit']
            ax2_ylabel = '$\mathrm{P_0}$ '+unit_titles['y_unit']

            # Define limits and make a 2d histogram of the surface pressures and temperatures
            t_lim = [np.min(local_retrieved_temperatures[:,-1]),np.max(local_retrieved_temperatures[:,-1])]
            t_range = t_lim[1]-t_lim[0]
            p_lim = [np.min(np.log10(local_retrieved_pressures[:,-1])),np.max(np.log10(local_retrieved_pressures[:,-1]))]
            p_range = p_lim[1]-p_lim[0]

            # Use previously defined limits to calculate a 2d histogram of the surface pressures and temperatures
            Z,X,Y=np.histogram2d(local_retrieved_temperatures[:,-1],np.log10(local_retrieved_pressures[:,-1]),bins=bins_inlay,
                range = [[t_lim[0]-0.1*t_range,t_lim[1]+0.1*t_range],[p_lim[0]-0.1*p_range,p_lim[1]+0.1*p_range]])
        
        Z = sp.ndimage.filters.gaussian_filter(Z, [0.75,0.75], mode='reflect')

        # Generate the colormap and plot the contours of the 2d histogram
        map, norm, levels = generate_color_map_from_levels(Z,color_levels,level_thresholds)
        contour = ax2.contourf((X[:-1]+X[1:])/2,10**((Y[:-1]+Y[1:])/2),Z.T,cmap=map,norm=norm,levels=np.array(levels))

        # plot the true values that were used to generate the input spectrum
        ax2.plot(local_true_temperatures[-1],local_true_pressures[-1],marker='s',color='C3',lw=0,ms=7, markeredgecolor='black')
        try:
            ax2.plot(local_true_temperatures_cloud_top,(local_true_pressures_cloud_top),marker='o',color='C1',lw=0,ms=7, markeredgecolor='black')
        except:
            pass
        
        # Arange the ticks for the inlay
        add_inlay_plot_labels(ax2,ax2_xlabel,ax2_ylabel,inlay_loc)

        # Find the minima and maxima of the outermost contour
        t_lim = [np.min([np.min(contour.allsegs[0][i][:,0]) for i in range(len(contour.allsegs[0]))]), np.max([np.max(contour.allsegs[0][i][:,0]) for i in range(len(contour.allsegs[0]))])]
        p_lim = [np.min([np.min(contour.allsegs[0][i][:,1]) for i in range(len(contour.allsegs[0]))]), np.max([np.max(contour.allsegs[0][i][:,1]) for i in range(len(contour.allsegs[0]))])]

        if x_lim_inlay is None:
            # Find the limits for the inlay plot from the contours (+- 10%)
            # if the span in pressure exceeds 2 orders of magnitude use log axes 
            ax2_xlim = [t_lim[0]-0.1*(t_lim[1]-t_lim[0]),t_lim[1]+0.1*(t_lim[1]-t_lim[0])]
        else:
            ax2_xlim=x_lim_inlay

        if y_lim_inlay is None:
            if np.log10(p_lim[1])-np.log10(p_lim[0]) >= 1.2:
                log_p = True
                ax2_ylim = [10**(np.log10(p_lim[0])-0.1*(np.log10(p_lim[1])-np.log10(p_lim[0]))),10**(np.log10(p_lim[1])+0.1*(np.log10(p_lim[1])-np.log10(p_lim[0])))]
                ax2.set_yscale('log')
            else:
                log_p = False
                ax2_ylim = [max([p_lim[0]-0.1*(p_lim[1]-p_lim[0]),0]),p_lim[1]+0.1*(p_lim[1]-p_lim[0])]
        else:
            ax2_ylim = y_lim_inlay
            ax2.set_yscale('log')

            log_p = True

        # Set the limits and ticks for the axes
        # x axis
        xticks = np.array([(1-pos)*ax2_xlim[0]+pos*ax2_xlim[1] for pos in [0.2,0.4,0.6,0.8]])
        roundx = np.log10(np.abs(xticks[1]-xticks[0]))
        ax2.set_xticks(xticks)
        if roundx>=0.5:
            ax2.set_xticklabels(xticks.astype(int),rotation=90)
        else:
            ax2.set_xticklabels(np.round(xticks,int(-np.floor(roundx-0.5))),rotation=90)
        ax2.set_xlim(ax2_xlim)

        # y axis
        if log_p:
            log_range = np.floor(np.log10(ax2_ylim)).astype(int)
            yticks = [10**i for i in range(log_range[0]+1,log_range[1]+1)]
            ax2.set_yticks(yticks)
        else:
            yticks = np.array([(1-pos)*ax2_ylim[0]+pos*ax2_ylim[1] for pos in [0.2,0.4,0.6,0.8]])
            roundy = np.log10(np.abs(yticks[1]-yticks[0]))
            ax2.set_yticks(yticks)
            if roundy>=0.5:
                ax2.set_yticklabels(yticks.astype(int))
            else:
                ax2.set_yticklabels(np.round(yticks,int(-np.floor(roundy-0.5))))
        ax2.set_ylim(ax2_ylim[::-1])

        # Legend cosmetics
        handles, labels = ax.get_legend_handles_labels()

        # Add the patches to the legend
        if plot_clouds:
            patch_handles = [MulticolorPatch([tuple(color_levels[i, :]),tuple(3*[0.9-i*0.15])],[1,1]) for i in range(N_levels)]
        else:
            patch_handles = [MulticolorPatch([tuple(color_levels[i, :])],[1]) for i in range(N_levels)]

        # Define the titles for the patches
        if quantiles_title is None:
            patch_labels = [str(quantiles[i])+'-'+str(quantiles[-i-1]) for i in range(N_levels)]
        else:
            patch_labels = quantiles_title
            
        # Add the legend
        if case_identifier=='':
            lgd = ax.legend(['Retrieval:']+patch_handles+[' ','Truth:']+handles,[' ']+patch_labels+[' ',' ']+labels,\
                            handler_map={str:  Handles(), MulticolorPatch:  MulticolorPatchHandler()}, ncol=legend_n_col,loc=legend_loc,frameon=False)
        else:
            lgd = ax.legend([case_identifier,'Retrieval:']+patch_handles+[' ','Truth:']+handles,[' ',' ']+patch_labels+[' ',' ']+labels,\
                            handler_map={str:  Handles(), MulticolorPatch:  MulticolorPatchHandler()}, ncol=legend_n_col,loc=legend_loc,frameon=False)

        # Save or pass back the figure
        if ax_arg is not None:
            pass
        elif save:
            if plot_residual:
                plt.savefig(self.results_directory+'Plots_New/plot_pt_structure_residual.pdf', bbox_inches='tight',bbox_extra_artists=(lgd,), transparent=True)
            else:
                plt.savefig(self.results_directory+'Plots_New/plot_pt_structure.pdf', bbox_inches='tight',bbox_extra_artists=(lgd,), transparent=True)
            return figure, ax
        else:
            return figure, ax