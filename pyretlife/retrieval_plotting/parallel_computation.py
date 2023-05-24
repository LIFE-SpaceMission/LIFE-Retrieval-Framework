"""
This module contains the `RetrievalPlottingObject` class, which is the main
class used to generate plots of the pyretlife retrievals.
"""

__author__ = "Konrad, Alei, Molliere, Quanz"
__copyright__ = "Copyright 2022, Konrad, Alei, Molliere, Quanz"
__maintainer__ = "Björn S. Konrad, Eleonora Alei"
__email__ = "konradb@ethz.ch, elalei@phys.ethz.ch"
__status__ = "Development"

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
import multiprocessing as mp
import contextlib
import pyretlife.retrieval_plotting.calculate_secondary_quantities as secondary_quantites


# Class that enables the parallelization of a function
class parallel():
    def __init__(self,num_proc):
        self.num_proc = num_proc

        # Define a manager process that collects the
        # data from the other processes
        self.manager = mp.Manager()
        self.result = self.manager.dict()
        self.jobs = []

    def calculate(self,results_directory,function,function_args):

        # Initialize the processes and start the calculation
        for process in range(self.num_proc):
            p = mp.Process(target=self.__worker, args=(process,results_directory,function,function_args))
            self.jobs.append(p)
            p.start()

        # Wait untill all the processes are done
        for proc in self.jobs:
            proc.join()

        # Return the data to the user
        return self.result

    def __worker(self,process,results_direectory,function,function_args):
        # Initialization of a new radtrans object
        with contextlib.redirect_stdout(None):
            from pyretlife.retrieval_plotting.run_plotting import retrieval_plotting_object
            results_temp = retrieval_plotting_object(results_directory = results_direectory)

        # Function calculation
        function_args['process'] = process
        function_args['rp_object'] = results_temp
        self.result[process] = getattr(secondary_quantites,function)(**function_args)