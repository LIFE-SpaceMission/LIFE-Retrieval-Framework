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
    """
    A class used to represent parallel function execution across multiple processes.

    This class enables parallelization of a function across a specified number of processes using 
    Python's `multiprocessing` module. It manages the creation of processes, distributes tasks, and 
    collects the results in a shared memory dictionary.

    Attributes
    ----------
    num_proc : int
        The number of processes to be used for parallel execution.
    manager : mp.Manager
        A manager object to handle shared data between processes.
    result : mp.Manager.dict
        A dictionary to store the results from each process.
    jobs : list
        A list of process objects representing the parallel tasks.

    Methods
    -------
    __init__(num_proc)
        Initializes the parallel execution setup with the specified number of processes.
    
    calculate(results_directory, function, function_args)
        Executes the given function in parallel across the specified number of processes.
    
    __worker(process, results_directory, function, function_args)
        The worker method executed by each process to run the given function and store the results.
    """

    def __init__(self,num_proc):
        """
        Initializes the parallel execution setup.

        :param num_proc: The number of processes to be used for parallel execution.
        :type num_proc: int
        """

        self.num_proc = num_proc

        # Define a manager process that collects the
        # data from the other processes
        self.manager = mp.Manager()
        self.result = self.manager.dict()
        self.jobs = []

    def calculate(self,results_directory,function,function_args):
        """
        Executes the provided function in parallel using multiple processes.

        This method initializes the processes, starts the calculations, and waits until all processes are done.
        The results are stored in a shared dictionary and returned once all processes finish.

        :param results_directory: Directory where the results will be saved or accessed.
        :type results_directory: str
        :param function: The function to be executed in parallel.
        :type function: callable
        :param function_args: Arguments to be passed to the function.
        :type function_args: dict

        :return: A dictionary containing the results from each process.
        :rtype: dict
        """

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
        """
        Worker function executed by each process.

        This function initializes a new object (e.g., a `retrieval_plotting_object`), executes the provided 
        function, and stores the result in a shared dictionary.

        :param process: The index of the current process.
        :type process: int
        :param results_directory: Directory where the results will be saved or accessed.
        :type results_directory: str
        :param function: The function to be executed by this process.
        :type function: callable
        :param function_args: Arguments to be passed to the function.
        :type function_args: dict
        """

        # Initialization of a new radtrans object
        with contextlib.redirect_stdout(None):
            from pyretlife.retrieval_plotting.run_plotting import retrieval_plotting_object
            results_temp = retrieval_plotting_object(results_directory = results_direectory)

        # Function calculation
        function_args['process'] = process
        function_args['rp_object'] = results_temp
        self.result[process] = getattr(secondary_quantites,function)(**function_args)