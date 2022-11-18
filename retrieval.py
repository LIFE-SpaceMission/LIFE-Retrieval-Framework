#from __future__ import absolute_import, unicode_literals, print_function

#!/usr/bin/env python

'''
REQUIREMENTS:
 -  python (including numpy)
 -  gfortran

BEFORE RUNNING:
Inside src, execute:
f2py -c --opt='-O3 -funroll-loops -ftree-vectorize -ftree-loop-optimize -msse -msse2 -m3dnow' -m rebin_give_width rebin_give_width.f90

BASIC STRUCTURE OF THE RETRIEVAL FOLDER:

- src/
    |
    |-- retrieval.py        # It calls the various functions
    |-- globals.py          # It stores the global variables
    |-- configure.py        # It reads config.ini and configures the run
    |-- forward_model.py    # It generates an emission spectrum
    |-- priors.py           # It fills the priors for the parameters
    |-- likelihood.py       # It calculates the likelihood

- config.ini    # It stores all useful info for the run

- chains/       # It stores all output files for each run
                # (created by the code if not present)

(- input/       # OPTIONAL: It stores the emission spectra to be
                # processed; config.ini determines the path for the
                # input files)

Any given run should be launched via command line:

mpiexec -n N_CORES python src/retrieval.py

'''

"""
The main program of the retrieval suite.


Calls
----------
Solve(LogLikelihood, Prior, n_dims, **args)
     PyMultiNest routine to solve the retrieval.
LogLike(cube)
    Calculates the log(likelihood) of the forward model generated
    with parameters and known variables.
Priors(cube)
    Converts the unity cube to prior cube by recognizing the names
    of the Parameters and giving them identificators.

"""
from retrieval_support import retrieval_global_class as rp_globals
from pymultinest.solve import solve
import numpy as np
import os

os.environ["OMP_NUM_THREADS"] = "1"

g=rp_globals.globals()
g.read_var()
g.read_data()
g.check_temp_params()
g.init_rt()
g.print_params()



# Run MultiNest
result = solve(LogLikelihood=g.LogLike, Prior=g.Priors,
                n_dims=len(g.params), outputfiles_basename=g.prefix,
                n_live_points=600, verbose=True) #600

# # Print final results
print('\n evidence: %(logZ).1f +- %(logZerr).1f\n' % result)
print('parameter values:')
for name, col in zip(g.params, result['samples'].transpose()):
     print('%15s : %.10f +- %.10f' % (name, col.mean(), col.std()))