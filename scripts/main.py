"""
The main program of the retrieval suite.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from argparse import ArgumentParser, Namespace

from pyretlife.retrieval.run import RetrievalObject

import sys

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
        "--sampler",
        required=False,
        help="Choice of nested sampling algorithm ([Naultilus] or MultiNest).",
    )
    parser.add_argument(
        "--nproc",
        required=False,
        help="Number of processes (used with Nautilus sampler)."
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
    
    comm = None
    rank = 0

    # Read the command line arguments (config file path)
    args = get_cli_arguments()
    
    if str(args.sampler).casefold() == "multinest":
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

    # Initializes a RetrievalObject (the pyret_ship)
    pyret_ship = RetrievalObject(run_retrieval=True)
    pyret_ship.load_configuration(config_file=args.config)
    pyret_ship.unit_conversion()
    pyret_ship.assign_knowns()
    pyret_ship.assign_prior_functions()
    pyret_ship.vae_initialization()
    pyret_ship.petitRADTRANS_initialization()
    if args.sampler.casefold() == "multinest":
        comm.Barrier()

    if rank == 0:
        pyret_ship.saving_inputs_to_folder(config_file=args.config)
    
    
    # # Run Nautilus
    if str(args.sampler).casefold() == "nautilus" or str(args.sampler).casefold() == "none":
        print("Begining retrieval with Nautilus")
        import numpy as np
        # from mpi4py.futures import MPIPoolExecutor
        from nautilus import Sampler
        sampler = Sampler(
            pyret_ship.unity_cube_to_prior_space,
            pyret_ship.calculate_log_likelihood,
            n_dim = len(pyret_ship.parameters),
            # pool = MPIPoolExecutor(),
            pool = int(args.nproc),
            filepath = str(pyret_ship.settings["output_folder"]) + "/" + "checkpoint.hdf5",
            resume = True,
            n_live = 2000,
            )
        complete = sampler.run(discard_exploration=True, verbose=False)
        with open(str(pyret_ship.settings["output_folder"]) + "/" + "summary.txt", 'a') as f:
            if complete:
                f.write(f"Retrieval completed successfully.\nlog_Z = {sampler.log_z}\n" \
                        + f"N_eff = {sampler.n_eff}\neta = {sampler.eta}")
            else:
                f.write("Retrieval encountered an error. Check log.")
        print("Retrieval completed, writing out results")
        points, log_w, log_l = sampler.posterior()
        np.savetxt(str(pyret_ship.settings["output_folder"]) + "/" + "posteriors_unequal.txt",
                   np.hstack((points, log_w.reshape((len(log_w),1)), log_l.reshape((len(log_l),1)))))
        points, log_w, log_l = sampler.posterior(equal_weight=True)
        np.savetxt(str(pyret_ship.settings["output_folder"]) + "/" + "posteriors_equal.txt",
                   np.hstack((points, log_w.reshape((len(log_w),1)), log_l.reshape((len(log_l),1)))))
    
    # # Run MultiNest
    elif str(args.sampler).casefold() == "multinest":
        comm.Barrier()
        from pymultinest.solve import solve
        result = solve(
            LogLikelihood=pyret_ship.calculate_log_likelihood,
            Prior=pyret_ship.unity_cube_to_prior_space,
            n_dims=len(pyret_ship.parameters),
            outputfiles_basename=str(pyret_ship.settings["output_folder"]) + "/",
            n_live_points=pyret_ship.settings["live_points"],
            verbose=True,
        )
        
    # # Chosen sampler not found
    else:
        raise ValueError("Chosen sampler not found. Supported samplers are Nautilus and MultiNest.")
    
    #
    # # Print final results
    # print("\n evidence: %(logZ).1f +- %(logZerr).1f\n" % result)
    # print("parameter values:")
    # for name, col in zip(g.params, result["samples"].transpose()):
    #     print("%15s : %.10f +- %.10f" % (name, col.mean(), col.std()))
