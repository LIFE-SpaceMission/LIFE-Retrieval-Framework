"""
The main program of the retrieval suite.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from argparse import ArgumentParser, Namespace
import os,warnings, sys

# from pymultinest.solve import solve
from pyretlife import RetrievalObject
from pyretlife.config import read_config_file, check_if_configs_match,get_check_opacity_path,get_check_prt_path,set_prt_opacity, validate_config


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
    input_prt_path = get_check_prt_path()
    sys.path.append(str(input_prt_path))
    # WORKS BUT COMMENTED TO NOT OVERWRITE THINGS WHILE BUILDING THE CODE
    # set_prt_opacity(input_prt_path,input_opacity_path)

    # Read the command line arguments (config file path)
    args = get_cli_arguments()

    # Load standard configurations (hard-coded)

    # Read in the configuration and check if there is already one in the file
    config = read_config_file(file_path=args.config)
    print(config)
    # Check if configuration file exists and if it matches
    if not check_if_configs_match(config=config):
        raise RuntimeError("Config exists and does not match!")

    # Populate the config file
    Knowns, Parameters, Settings = populate_dictionaries(config)
    import ipdb;ipdb.set_trace()
    # Validate the config file: does it have all we need?
    # TODO: It already includes the old check_temp_pars. More checks are necessary
    validate_config(config)

    # Paste the full config file (including the default arguments) to the output directory (and also other things e.g. retrieval version, github commit string, environment variables for future backtracing)


    # Initializes a RetrievalObject (the pyret_ship)
    pyret_ship = RetrievalObject.RetrievalObject(config_file= config, run_retrieval=True)

    # Read the configuration file
    # TODO: old g.read_var()





    # g.read_data()
    # g.init_rt()

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
