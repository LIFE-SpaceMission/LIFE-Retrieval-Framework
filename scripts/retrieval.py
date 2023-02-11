"""
The main program of the retrieval suite.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import os

from pyretlife.retrieval import global_class as rp_globals
from pymultinest.solve import solve


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Set the number of threads to 1
    os.environ["OMP_NUM_THREADS"] = "1"

    # Read the configuration file
    g = rp_globals.globals()
    g.read_var()
    g.read_data()
    g.check_temp_params()
    g.init_rt()
    g.print_params()

    # Run MultiNest
    result = solve(
        LogLikelihood=g.log_likelihood,
        Prior=g.priors,
        n_dims=len(g.params),
        outputfiles_basename=g.prefix,
        n_live_points=600,  # TODO: Make this a parameter configurable
        verbose=True,
    )

    # Print final results
    print("\n evidence: %(logZ).1f +- %(logZerr).1f\n" % result)
    print("parameter values:")
    for name, col in zip(g.params, result["samples"].transpose()):
        print("%15s : %.10f +- %.10f" % (name, col.mean(), col.std()))
