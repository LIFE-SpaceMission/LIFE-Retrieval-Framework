from argparse import ArgumentParser, Namespace
import sys
import os
from pyretlife.retrieval_plotting.run_plotting import retrieval_plotting_object
import astropy.units as u
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import shlex,subprocess
import yaml

# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------


def get_cli_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--nproc",
        required=True,
        help="number of processes.",
    )
    parser.add_argument(
        "--sampler",
        required=False,
        help="Nested sampling algorithm ([MultiNest] or Nautilus)."
    )
    args = parser.parse_args()
    return args


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == "__main__":


    # Read the command line arguments (config file path)
    args = get_cli_arguments()

    with open(str(args.config), 'r') as file:
        config_file = yaml.safe_load(file)
    path = str(config_file['RUN SETTINGS']['output_folder'])
    
    if not os.path.exists(os.getcwd() + '/main.py'):
        raise FileNotFoundError('Could not locate file LIFE-Retrieval-Framework/scripts/main.py.' + \
                                'Make sure to launch this file from the scripts dir or create a new version and point to main.py.')
    
    if str(args.sampler).casefold() == 'nautilus':
        inputs  = shlex.split('python ' + os.getcwd() + '/main.py --config '\
                              +str(args.config) + ' --sampler ' +str(args.sampler) +' --nproc ' +str(args.nproc))
    else:
        inputs  = shlex.split('mpiexec -n ' + str(int(args.nproc)) + \
                              ' python ' + os.getcwd() + '/main.py --config '\
                              +str(args.config) + ' --sampler ' +str(args.sampler))
        

    process = subprocess.Popen(inputs,env=os.environ)
    process.wait()
    process.terminate()
 
    # Initializes a RetrievalObject (the pyret_ship)
    results = retrieval_plotting_object(results_directory = path)

    # Calculates and saves the PT profiles and spectra for plotting and 
    results.calculate_posterior_pt_profile(n_processes=min([200,int(args.nproc)]),reevaluate_PT=False)

    results.calculate_posterior_spectrum(n_processes=min([200,int(args.nproc)]),reevaluate_spectra=False)

    results.deduce_bond_albedo(stellar_luminosity=1.0,
                                        error_stellar_luminosity=0.01,
                                        planet_star_separation=1.0,
                                        error_planet_star_separation=0.01,
                                        true_equilibrium_temperature = 255,
                                        true_bond_albedo = 0.29,
                                        reevaluate_bond_albedo=False)
    results.deduce_abundance_profiles(reevaluate_abundance_profiles=False)
    
    results.deduce_gravity(true_gravity = 981)
    results.deduce_surface_temperature(true_surface_temperature = 273)

    unit_titles = {'R_pl':'$\mathrm{R_{Earth}}$','M_pl':'$\mathrm{M_{Earth}}$'}
    
    #Corner plot
    results.Posteriors(save=True,
                        plot_corner=True,
                        add_table=True,
                                                            
                        plot_pt=True,
                        log_pressures=True,
                        log_mass=True,
                        log_abundances=True,
                        log_particle_radii=False,
                                                                
                        bins=40,
                        quantiles1d=[0.16, 0.5, 0.84],
                        color='#009e73',
                        color_truth='k',
                        parameter_units='input',
                        #custom_parameter_titles=custom_parameter_titles,
                        custom_unit_titles=unit_titles,
                        ULU_lim=[-1.8,3])
    
    #Retrieved spectrum
    results.plot_retrieved_flux(quantiles = [0.05,0.15,0.25,0.35,0.65,0.75,0.85,0.95],
                        quantiles_title = [r'$5-95\%$',r'$15-85\%$',r'$25-75\%$',r'$35-65\%$'],
                        ax = None,
                        color = '#009e73',
                        case_identifier = None,
                        figsize=(12,2),
                        legend_loc = 'upper left',
                        x_lim = None,
                        y_lim = None,
                        noise_title = 'Observation Noise',  
                        plot_instruments_separately=False,     
                        plot_residual = False,
                        plot_log_wavelength=False,
                        plot_log_flux=False,
                        plot_unit_wavelength=None,
                        plot_unit_flux=None,#u.photon/u.m**2/u.s/ u.micron,
                        plot_retrieved_median=False,
                                                                
                        plot_noise = True,
                        plot_true_spectrum = True,
                        plot_datapoints = False)
    
    #Spectrum residuals
    results.plot_retrieved_flux(quantiles = [0.05,0.15,0.25,0.35,0.65,0.75,0.85,0.95],
                        quantiles_title = [r'$5-95\%$',r'$15-85\%$',r'$25-75\%$',r'$35-65\%$'],
                        ax = None,
                        color = '#009e73',
                        case_identifier = None,
                        figsize=(12,2),
                        legend_loc = 'upper center',
                        x_lim = None,
                        y_lim = None,
                        noise_title = 'Observation Noise',  
                        plot_instruments_separately=False,     
                        plot_residual = True,
                        plot_log_wavelength=False,
                        plot_log_flux=False,
                        plot_unit_wavelength=None,
                        plot_unit_flux=None,#u.photon/u.m**2/u.s/ u.micron,
                        plot_retrieved_median=False,
                                                                
                        plot_noise = True,
                        plot_true_spectrum = True,
                        plot_datapoints = False)
    
    #Retrieved PT profile
    results.plot_retrieved_pt_profile(save = True,
                                      quantiles = [0.05,0.15,0.25,0.35,0.65,0.75,0.85,0.95],
                                      quantiles_title = [r'$5-95\%$',r'$15-85\%$',r'$25-75\%$',r'$35-65\%$'],
                                      x_lim = [0, 700],
                                      y_lim = [1e-4, 1e2],
                                      ax = None,
                                      color = '#009e73',
                                      case_identifier = None,
                                      legend_loc = 'lower right',
                                      plot_truth = False,)
    
    # #Gas abundances
    # results.plot_abundance_profiles()
