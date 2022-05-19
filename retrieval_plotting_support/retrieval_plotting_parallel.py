__author__ = "Konrad"
__copyright__ = "Copyright 2022, Konrad"
__maintainer__ = "Björn S. Konrad"
__email__ = "konradb@phys.ethz.ch"
__status__ = "Development"

# Standard Libraries
import retrieval_plotting as rp
import multiprocessing as mp





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
        results_temp =  rp.retrieval_plotting(results_direectory)

        # Function calculation
        function_args['process'] = process
        self.result[process] = getattr(results_temp,function)(**function_args)
        