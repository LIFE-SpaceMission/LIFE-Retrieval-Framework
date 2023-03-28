"""
The main program of the retrieval suite.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from argparse import ArgumentParser, Namespace
import os, warnings, sys

# from pymultinest.solve import solve

from pyretlife.retrieval import RetrievalObject


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
    # os.environ["OMP_NUM_THREADS"] = "1"
    # warnings.simplefilter("ignore")
    #


    # Read the command line arguments (config file path)
    args = get_cli_arguments()

    # Initializes a RetrievalObject (the pyret_ship)
    pyret_ship = RetrievalObject.RetrievalObject(run_retrieval=True)
    pyret_ship.load_configuration(config_file=args.config)
    pyret_ship.unit_conversion()
    pyret_ship.assign_prior_functions()
    pyret_ship.petitRADTRANS_initialization()
    import ipdb

    ipdb.set_trace()

    # Paste the full config file (including the default arguments) to the output directory (and also other things e.g. retrieval version, github commit string, environment variables for future backtracing)
    #pyret_ship.saving_inputs_to_folder()

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
