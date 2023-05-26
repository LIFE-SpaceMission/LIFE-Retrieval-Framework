"""
The main program of the retrieval suite.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from argparse import ArgumentParser, Namespace

from pymultinest.solve import solve

from pyretlife.retrieval_plotting.run_plotting import retrieval_plotting_object

import astropy.units as u

import sys


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------


def get_cli_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--res_dir",
        required=True,
        help="Path to the folder containing the results.",
    )
    args = parser.parse_args()
    return args


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Set the number of threads to 1
    # os.environ["OMP_NUM_THREADS"] = "1"
    # warnings.simplefilter("ignore")
    #

    # Read the command line arguments (config file path)
    args = get_cli_arguments()

    # Initializes a RetrievalObject (the pyret_ship)
    plotting = retrieval_plotting_object(results_directory = args.res_dir)
    #addition for old venus runs
    plotting.posteriors['M_pl']=plotting.posteriors['M_pl']*5.972167867791379e+27
    plotting.posteriors['R_pl']=plotting.posteriors['R_pl']*637810000.0

    # Calculates and saves the PT profiles and spectra for plotting and 
    plotting.calculate_posterior_pt_profile(n_processes=10,reevaluate_PT=False)
    plotting.calculate_true_pt_profile()

    plotting.calculate_true_spectrum()


    plotting.calculate_posterior_spectrum(n_processes=50,reevaluate_spectra=False)

    #plotting.calculate_bond_albedo(stellar_luminosity=1,
    #                               error_stellar_luminosity=0.05,
    #                               planet_star_separation=1,
    #                               error_planet_star_separation=0.05,
    #                               true_equilibrium_temperature = 255,
    #                               true_bond_albedo = 0.29)

    plotting.calculate_bond_albedo(stellar_luminosity=1,
                                   error_stellar_luminosity=0.05,
                                   planet_star_separation=0.723,
                                   error_planet_star_separation=0.723*0.05,
                                   true_equilibrium_temperature = 226,
                                   true_bond_albedo = 0.77)

    unit_titles = {'R_pl':'$\mathrm{R_{Earth}}$','M_pl':'$\mathrm{M_{Earth}}$'}
    custom_parameter_titles = {'H2SO484(c)_am':r'$\mathrm{Species^{cloud}})$',
                               'H2SO484(c)_am_top_pressure':r'$P^\mathrm{cloud}_\mathrm{top}$',
                               'H2SO484(c)_am_thickness':r'$P^\mathrm{cloud}_\mathrm{span}$',
                               'H2SO484(c)_am_particle_radius':r'$\bar{R}^\mathrm{cloud}$',
                               'H2SO484(c)_am_sigma_lnorm':r'$\sigma^\mathrm{cloud}$'
                               }


    plotting.Posteriors(save=True,
                        plot_corner=True,
                        add_table=True,
                        
                        plot_pt=True,

                        log_pressures=True,
                        log_mass=True,
                        log_abundances=True,
                        log_particle_radii=True,
                        
                        bins=40,
                        quantiles1d=[0.16, 0.5, 0.84],
                        color='#009e73',
                        color_truth='k',
                        parameter_units='input',
                        custom_parameter_titles=custom_parameter_titles,
                        custom_unit_titles=unit_titles,
                        ULU_lim=[-1.8,3])#,plot_bond=[1,0.05,1,0.05,255,0.31]) # plot_bond=[1,0.05,0.723,0.723*0.05,226,0.77]



    plotting.plot_retrieved_flux(
                    quantiles = [0.05,0.15,0.25,0.35,0.65,0.75,0.85,0.95],
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
    


    plotting.plot_retrieved_pt_profile(save=True,    inlay_loc='upper right', x_lim =[110,790], y_lim = [1e-6,10**(3.9)],legend_loc = 'lower left',
            quantiles=[0.05,0.15,0.25,0.35,0.65,0.75,0.85,0.95],quantiles_title=[r'$5-95\%$',r'$15-85\%$',r'$25-75\%$',r'$35-65\%$'],bins_inlay = 20,figsize=(5, 4.4),
            
            color='#009e73',

            true_cloud_top=[None,None],

            plot_residual = False,
            plot_clouds =True,
            plot_unit_temperature=None,
            plot_unit_pressure=None,)
    

"""    
    plotting.plot_retrieved_pt_profile(save=True,  x_lim =[0,1000], y_lim = [1e-6,1e4], quantiles=[0.05,0.15,0.25,0.35,0.65,0.75,0.85,0.95],
                    quantiles_title = None, inlay_loc='upper right', bins_inlay = 20,x_lim_inlay =None, y_lim_inlay = None, figure = None, ax = None, color='C2', case_identifier = '',
                    legend_n_col = 2, legend_loc = 'best',n_processes=50,figsize=(6.4, 4.8),h_cover=0.45,reevaluate_PT = False,
                    
                    true_cloud_top=[None,None],

                    plot_residual = False,
                    plot_clouds = False,
                    plot_unit_temperature=None,
                    plot_unit_pressure=None,)
    

    pyret_ship.load_configuration(config_file=args.config)
    pyret_ship.unit_conversion()
    pyret_ship.assign_knowns()
    pyret_ship.assign_prior_functions()
    pyret_ship.vae_initialization()
    pyret_ship.petitRADTRANS_initialization()

    # TODO Paste the full config file (including the default arguments) to the output directory (and also other things
    #  e.g. retrieval version, github commit string, environment variables for future backtracing)
    pyret_ship.saving_inputs_to_folder()

    # # Run MultiNest
    result = solve(
        LogLikelihood=pyret_ship.calculate_log_likelihood,
        Prior=pyret_ship.unity_cube_to_prior_space,
        n_dims=len(pyret_ship.parameters),
        outputfiles_basename=str(pyret_ship.settings["output_folder"]) + "/",
        n_live_points=pyret_ship.settings["live_points"],
        verbose=True,
    )
    #
    # # Print final results
    # print("\n evidence: %(logZ).1f +- %(logZerr).1f\n" % result)
    # print("parameter values:")
    # for name, col in zip(g.params, result["samples"].transpose()):
    #     print("%15s : %.10f +- %.10f" % (name, col.mean(), col.std()))
"""