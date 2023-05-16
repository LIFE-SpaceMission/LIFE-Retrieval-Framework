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

    # Calculates and saves the PT profiles and spectra for plotting and 
    plotting.calculate_posterior_pt_profile(n_processes=10,reevaluate_PT=True)
    plotting.calculate_true_pt_profile()
    plotting.calculate_posterior_spectrum(n_processes=40,reevaluate_spectra=True)
    plotting.calculate_true_spectrum()
    #print(plotting.__dict__.keys())

    plotting.Flux_Error(skip =1, plot_residual = True,save =True,quantiles=[0.05,0.15,0.25,0.35,0.65,0.75,0.85,0.95],quantiles_title=[r'$5-95\%$',r'$15-85\%$',r'$25-75\%$',r'$35-65\%$'],n_processes=30,
                    plot_noise = True, plot_true_spectrum = True, legend_loc = 'upper center',color = '#009e73',noise_title = 'Photon Noise',figsize=(15.275, 3.02),median_only=False,reevaluate_spectra = False,split_instruments=True)

    #plotting.plot_retrieved_flux(quantiles = [0.05,0.15,0.25,0.35,0.65,0.75,0.85,0.95],
    #                quantiles_title = [r'$5-95\%$',r'$15-85\%$',r'$25-75\%$',r'$35-65\%$'], ax = None, color = '#009e73', case_identifier = None,
    #                
    #                figsize=(12,2),
    #                legend_loc = 'upper center',
    #                x_lim = None,
    #                y_lim = None,
    #                
    #                noise_title = 'Observation Noise',  
    #
    #                
    #                plot_instruments_separately=False,     
    #                plot_residual = True,
    #                plot_log_wavelength=False,
    #                plot_log_flux=False,
    #                plot_unit_wavelength=None,
    #                plot_unit_flux=u.photon/u.m**2/u.s/ u.micron,
    #                plot_retrieved_median=False,
    #                
    #                plot_noise = True,
    #                plot_true_spectrum = True,
    #                plot_datapoints = False)

    plotting.plot_retrieved_pt_profile(save=True,    inlay_loc='upper right', x_lim =[110,790], y_lim = [1e-6,10**(3.9)],legend_loc = 'lower left',
            quantiles=[0.05,0.15,0.25,0.35,0.65,0.75,0.85,0.95],quantiles_title=[r'$5-95\%$',r'$15-85\%$',r'$25-75\%$',r'$35-65\%$'],bins_inlay = 20,figsize=(5, 4.4),
            
            color='#009e73',

            true_cloud_top=[None,None],

            plot_residual = False,
            plot_clouds =False,
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