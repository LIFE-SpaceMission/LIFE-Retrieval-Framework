"""
The main program of the retrieval suite.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from argparse import ArgumentParser, Namespace

from pprint import pprint

import os,warnings

from pyretlife.retrieval import global_class as rp_globals
# from pymultinest.solve import solve

from pyretlife.config import read_config_file, check_if_configs_match,get_check_opacity_path,get_check_pRT_path


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
    args = parser.parse_args()
    return args


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # Set the number of threads to 1
    os.environ["OMP_NUM_THREADS"] = "1"
    warnings.simplefilter("ignore")

    # Get and check the goodness of the environmental variables
    input_opacity_path=get_check_opacity_path()
    input_prt_path = get_check_pRT_path()


    # Read the command line arguments (config file path)
    args = get_cli_arguments()

    # Read in the configuration and check if there is already one in the file
    config = read_config_file(file_path=args.config)
    if not check_if_configs_match(config=config):
        raise RuntimeError("Config exists and does not match!")
    pprint(config)

    # Validate the config file: does it have all we need?


    # Read the configuration file
    # g = rp_globals.globals()
    # g.read_var()
    # g.read_data()
    # g.check_temp_params()
    # g.init_rt()
    # g.print_params()

    # # Run MultiNest
    # result = solve(
    #     LogLikelihood=g.log_likelihood,
    #     Prior=g.priors,
    #     n_dims=len(g.params),
    #     outputfiles_basename=g.prefix,
    #     n_live_points=600,  # TODO: Make this a parameter configurable
    #     verbose=True,
    # )
    #
    # # Print final results
    # print("\n evidence: %(logZ).1f +- %(logZerr).1f\n" % result)
    # print("parameter values:")
    # for name, col in zip(g.params, result["samples"].transpose()):
    #     print("%15s : %.10f +- %.10f" % (name, col.mean(), col.std()))
