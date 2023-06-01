"""
The main program of the retrieval suite.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from argparse import ArgumentParser, Namespace

from pymultinest.solve import solve

from mpi4py import MPI

from pyretlife.retrieval.run import RetrievalObject


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

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Read the command line arguments (config file path)
    args = get_cli_arguments()

    # Initializes a RetrievalObject (the pyret_ship)
    pyret_ship = RetrievalObject(run_retrieval=True)
    pyret_ship.load_configuration(config_file=args.config)
    pyret_ship.unit_conversion()
    pyret_ship.assign_knowns()
    pyret_ship.assign_prior_functions()
    pyret_ship.vae_initialization()
    pyret_ship.petitRADTRANS_initialization()
    comm.Barrier()

    if rank == 0:
        pyret_ship.saving_inputs_to_folder(config_file=args.config)
    comm.Barrier()

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
