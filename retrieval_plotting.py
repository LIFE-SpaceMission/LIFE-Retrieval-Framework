__author__ = "Konrad"
__copyright__ = "Copyright 2022, Konrad"
__maintainer__ = "Björn S. Konrad"
__email__ = "konradb@phys.ethz.ch"
__status__ = "Development"

# Standard Libraries
import sys, os, re
import math as m
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.ndimage.filters import gaussian_filter1d
import scipy.interpolate as scp
import scipy.optimize as sco
import time as t

# Import additional external files
from retrieval_support import retrieval_global_class as r_globals
from retrieval_support import retrieval_posteriors as r_post
from retrieval_plotting_support import retrieval_plotting_colors as rp_col
from retrieval_plotting_support import retrieval_plotting_posteriors as rp_posteriors
from retrieval_plotting_support import retrieval_plotting_handlerbase as rp_hndl
from retrieval_plotting_support import retrieval_plotting_inlay as rp_inlay
from retrieval_plotting_support import retrieval_plotting_parallel as rp_parallel
from retrieval_plotting_support import retrieval_plotting_mathfunc as rp_mathfunc

# Additional Libraries
import pymultinest as nest
import spectres as spectres





# Create a new class that inherits functions from globals
class retrieval_plotting(r_globals.globals):

    def __init__(self, results_directory):
        '''
        This function reads the input.ini file as well as the retrieval
        results files of imterest to us and stores the read in data
        in order to generate the retrieval plots of interest to us.
        '''
        self.results_directory = results_directory
        self.code_directory = os.path.dirname(os.path.realpath(__file__))
        super().__init__(input_file=str(self.results_directory+'input.ini'),retrieval = False)
        
        if not os.path.exists(self.results_directory + 'Plots/'):
            os.makedirs(self.results_directory + 'Plots/')

        # Define the lists containing the titles for the data
        self.titles = []
        self.truths = []
        self.priors = []
        self.priors_range = []

        # Read the input data
        self.read_var()
        
        # if the vae_pt is selected initialize the pt profile model
        if self.settings['parametrization'] == 'vae_pt':
            from retrieval_support import retrieval_pt_vae as vae
            self.vae_pt = vae.VAE_PT_Model(file_path=self.code_directory+'/retrieval_support/vae_pt_models/'+self.settings['vae_net'])

        # Read the retrieval results from the chain files
        # self.n_params is the number of retrieved parameters
        self.n_params = len(self.params.keys())
        self.data = nest.Analyzer(self.n_params,outputfiles_basename = self.results_directory)
        self.equal_weighted_post = self.data.get_equal_weighted_posterior()
        self.best_fit = self.data.get_best_fit()['parameters']
        self.evidence = [self.data.get_stats()['global evidence'],self.data.get_stats()['global evidence error']]





    """
    #################################################################################
    #                                                                               #
    #   General routines for getting data from the runfiles.                        #
    #                                                                               #
    #################################################################################
    """



    def read_var(self):
        '''
        This function reads the input.ini file and fills up the three
        dictionaries: settings, params, knowns.
        '''
        for section in self.config_file.sections():

            for (key, val) in self.config_file.items(section):
                if 'settings' in key:
                    self.settings[key[9:]] = val
                # check if the first element is a letter (the priors)           
                elif val[:1].upper().isupper():
                    param = {'prior': val,
                             'type': section}
                    self.params[key] = param
                    if section == 'CHEMICAL COMPOSITION PARAMETERS':
                        # Define the titles such that they work well for the chemical abundances
                        self.titles.append('$\\mathrm{'+'_'.join(re.sub( r"([0-9])", r" \1", key.split('_')[0]).split())+'}$')
                    elif section == 'PHYSICAL PARAMETERS':
                        # Define the titles such that they work well for the chemical abundances
                        s = key.split('_')
                        try:
                            self.titles.append('$\\mathrm{'+s[0]+'_{'+s[1]+'}}$')
                        except:
                            self.titles.append('$\\mathrm{'+'_'.join(re.sub( r"([0-9])", r" \1", key).split())+'}$')
                    elif section == 'TEMPERATURE PARAMETERS':
                        # Define the titles such that they work well for the pt parameters
                        self.titles.append('$\\mathrm{'+str(key)+'}$')
                    elif section == 'CLOUD PARAMETERS':
                        # Define the titles such that they work well for the chemical abundances
                        temp = key.split('_')
                        if 'H2SO4' in temp[0]:
                            temp[0] = 'H2SO4(c)'
                        temp[0] = '$\\mathrm{'+'_'.join(re.sub( r"([0-9])", r" \1", temp[0][:-3]).split())+'}$'
                        temp.pop(1)
                        self.titles.append('\n'.join(temp))
                    else:
                        self.titles.append(key)

                    # Storing the truths if provided
                    if val.split(' ')[-2]== 'T':
                        self.truths.append(float(val.split(' ')[-1]))
                    else:
                        self.truths.append(None)

                    # Storing the prior Type
                    self.priors.append(val.split(' ')[0])
                    self.priors_range.append([float(val.split(' ')[i]) for i in range(1,3)])
                    
                else:
                    known = {'value': float(val),
                                 'type': section}
                    self.knowns[key] = known
        
        self.input_wavelength, self.input_flux, self.input_error = np.loadtxt(self.results_directory + 'input_spectrum.txt').T
        try:
            self.input_temperature, self.input_pressure = np.loadtxt(self.settings['input_profile']).T
        except:
            pass



    def __get_knowns(self):
        '''
        This function creates libraries for the known
        parameters in a retrieval.
        '''

        # Dictionaries to stor the parameters for spectrum calculation
        temp_vars_known = {}
        phys_vars_known = {}
        chem_vars_known = {}
        cloud_vars_known = {}

        # Read in all known spectral parameters
        for par in self.knowns.keys():
            #key = list(self.knowns.keys()).index(par)
            if self.knowns[par]['type'] == 'TEMPERATURE PARAMETERS':
                temp_vars_known[par] = self.knowns[par]['value']
            elif self.knowns[par]['type'] == 'PHYSICAL PARAMETERS':
                phys_vars_known[par] = self.knowns[par]['value']
            elif self.knowns[par]['type'] == 'CHEMICAL COMPOSITION PARAMETERS':
                chem_vars_known[par.split('_',1)[0]] = self.knowns[par]['value']
            elif self.knowns[par]['type'] == 'CLOUD PARAMETERS':
                if not '_'.join(par.split('_',2)[:2]) in cloud_vars_known.keys():
                    cloud_vars_known['_'.join(par.split('_',2)[:2])] = {}
                try:
                    cloud_vars_known['_'.join(par.split('_',2)[:2])][par.split('_',2)[2]] = self.knowns[par]['value']
                except:
                    cloud_vars_known['_'.join(par.split('_',2)[:2])]['abundance'] = self.knowns[par]['value']
                    chem_vars_known[par.split('_',1)[0]] = self.knowns[par]['value']
        
        return temp_vars_known, phys_vars_known, chem_vars_known, cloud_vars_known



    def __get_retrieved(self,temp_vars_known,chem_vars_known,phys_vars_known,cloud_vars_known,temp_equal_weighted_post,ind,ind_bypass=False):
        '''
        This function creates libraries for the retrieved
        parameters in a retrieval for a given index of the
        equal weighted posterior.
        '''

        # Copy the known parameters into temporary dictionaries
        self.temp_vars = temp_vars_known.copy()
        self.chem_vars = chem_vars_known.copy()
        self.phys_vars = phys_vars_known.copy()
        self.cloud_vars = cloud_vars_known.copy()
            
        # Read in the values from the equal weighted posteriors
        retrieved_params = list(self.params.keys())
        for par in range(len(retrieved_params)):
            if self.params[retrieved_params[par]]['type'] == 'TEMPERATURE PARAMETERS':
                self.temp_vars[retrieved_params[par]] = temp_equal_weighted_post[ind,par]
            elif self.params[retrieved_params[par]]['type'] == 'CHEMICAL COMPOSITION PARAMETERS':
                self.chem_vars[retrieved_params[par]] = temp_equal_weighted_post[ind,par]
            elif self.params[retrieved_params[par]]['type'] == 'PHYSICAL PARAMETERS':
                self.phys_vars[retrieved_params[par]] = temp_equal_weighted_post[ind,par]
            elif self.params[retrieved_params[par]]['type'] == 'CLOUD PARAMETERS':
                if not '_'.join(retrieved_params[par].split('_',2)[:2]) in self.cloud_vars.keys():
                    self.cloud_vars['_'.join(retrieved_params[par].split('_',2)[:2])] = {}
                try:
                    self.cloud_vars['_'.join(retrieved_params[par].split('_',2)[:2])][retrieved_params[par].split('_',2)[2]] = temp_equal_weighted_post[ind,par]
                    if ind == 0 or ind_bypass:
                        if self.settings['clouds'] == 'opaque':
                            if retrieved_params[par].split('_',2)[2] == 'thickness':
                                min_cloud_center = temp_equal_weighted_post[2:,par-1]+temp_equal_weighted_post[2:,par]/2
                                hist = np.histogram(np.log10(min_cloud_center),bins=50)
                                self.cloud_opacity_cut = sco.curve_fit(rp_mathfunc.Logistic_Function,(hist[1][:-1]+hist[1][1:])/2,hist[0],p0=[80,10,0])[0]
                                self.opacity = rp_mathfunc.Inverse_Logistic_Function(self.cloud_opacity_cut[0]*np.array([0.16,0.5,0.84]),*self.cloud_opacity_cut)
                except:
                    self.cloud_vars['_'.join(retrieved_params[par].split('_',2)[:2])]['abundance'] = temp_equal_weighted_post[ind,par]
                    self.chem_vars[retrieved_params[par].split('_',1)[0]] = temp_equal_weighted_post[ind,par]



    def __g_test(self):
        '''
        Function to check if the surface gravity is provided or can
        be calculated from the provided parameters and brings it to
        the correct format for petit radtrans
        '''
        from petitRADTRANS import nat_cst as nc

        # Calculate the surface gravity g given M_Pl and R_pl or log_g. If in knowns already, skip
        if 'g' not in self.phys_vars.keys():
            if 'log_g' in self.phys_vars.keys():
                self.phys_vars['g'] = 10**self.phys_vars['log_g']
            else:
                try:
                    self.phys_vars['M_pl'] = self.phys_vars['M_pl'] * nc.m_earth
                except:
                    print("ERROR! Planetary mass is missing!")
                    sys.exit()
                self.phys_vars['g'] = nc.G*self.phys_vars['M_pl'] / (self.phys_vars['R_pl'])**2



    def __P0_test(self,ind=None):
        '''
        Function to check if the surface pressure is provided or can
        be calculated from the provided parameters and brings it to
        the correct format for petit radtrans
        '''

        if self.settings['clouds'] == 'opaque':
            # Choose a surface pressure below the lower cloud deck
            if not (('log_P0' in self.phys_vars.keys()) or ('P0' in self.phys_vars.keys())):
                self.phys_vars['log_P0'] = 4
            else:
                if (('log_P0' in self.knowns.keys()) or ('P0' in self.knowns.keys())):
                    if ind is not None:
                        if ind == 0:
                            if 'P0' in self.knowns.keys():
                                self.phys_vars['log_P0'] = np.log10(self.knowns['P0']['value'])
                            else:
                                self.phys_vars['log_P0'] = self.knowns['log_P0']['value']
                        else:
                            self.phys_vars['log_P0'] = 4
                    else:
                        self.phys_vars['log_P0'] = 4
                else:
                    print("ERROR! For opaque cloud models the surface pressure P0 is not retrievable!")
                    sys.exit()
        else:
            if 'log_P0' not in self.phys_vars.keys():
                if 'P0' in self.phys_vars.keys():
                    self.phys_vars['log_P0'] = np.log10(self.phys_vars['P0'])
                else:
                    print("ERROR! Either log_P0 or P0 is needed!")
                    sys.exit()





    """
    #################################################################################
    #                                                                               #
    #   Routines for calculating spectra and PT profiles form posteriors.           #
    #                                                                               #
    #################################################################################
    """



    def Calc_PT_Profiles(self,skip = 1,layers=500,p_surf=4,n_processes=None,process=None):
        '''
        Function to calculate the PT profiles corresponding to the retrieved posterior distributions
        for subsequent plotting in the flux PT plotting functions.
        '''

        # Add the thest fit spectrum and the input parameters to the equal weighted posterior
        temp_equal_weighted_post = np.append(np.array([self.best_fit]),self.equal_weighted_post[:,:-1],axis=0)
        temp_equal_weighted_post = np.append(np.array([self.truths]),temp_equal_weighted_post,axis=0)

        # Fetch the known parameters
        temp_vars_known, phys_vars_known, chem_vars_known, cloud_vars_known = self.__get_knowns()

        # Iterate over the equal weighted posterior distribution using the user-defined skip value
        dimension = np.shape(temp_equal_weighted_post)[0]//skip

        #If run in multiprocessing split up the jobs
        if (n_processes is not None) and (process is not None):

            ind_start = process*dimension//n_processes
            ind_end = min(dimension,(process+1)*dimension//n_processes)

            # Printing infor ffor the multiprocessing case
            if process == 0:
                print('\n-----------------------------------------------------')
                print('\n    PT-Profile calculation on multiple CPUs:')
                print('')
                print('    PT-Profiles to calculate: ',str(dimension))
                print('    Number of processes: '+str(n_processes))
                print('')
                print('    Distribution of tasks:')
                for proc_ind in range(n_processes):
                    print('\tProcess '+str(proc_ind)+':\tProfiles:\t'+str(proc_ind*dimension//n_processes+1)+'-'+str(min(dimension,(proc_ind+1)*dimension//n_processes)))
                print('\n-----------------------------------------------------\n')
        else:
            ind_start = 0
            ind_end = dimension
            process = 0

        # Import petitRADTRANS
        sys.path.append(self.path_prt)
        from petitRADTRANS import Radtrans
        from petitRADTRANS import nat_cst as nc

        self.read_data(retrieval = False)
        
        # Print status of calculation
        if process == 0:
            print('Starting PT-profile calculation.')
            print('\t0.00 % of PT-profiles calculated.', end = "\r")

        results = {}
        t_start = t.time()
        for i in range(ind_start,ind_end):
            ind = min(2,i)+skip*max(0,i-2)

            # Fetch the retrieved parameters for a given ind
            self.__get_retrieved(temp_vars_known,chem_vars_known,phys_vars_known,cloud_vars_known,temp_equal_weighted_post,ind,ind_bypass=(i==ind_start))
        
            # Test the values of P0 and g and change to required values if necessary
            self.__g_test()
            self.__P0_test(ind=i)
            
            # Calculate the cloud bottom pressure from the cloud thickness parameter
            cloud_tops = []
            cloud_bottoms = []
            for key in self.cloud_vars.keys():
                cloud_bottoms += [self.cloud_vars[key]['top_pressure']+self.cloud_vars[key]['thickness']]
                cloud_tops += [self.cloud_vars[key]['top_pressure']]
                self.make_press_temp_terr(log_top_pressure=np.log10(np.min(cloud_tops)),layers=layers)
                pressure_cloud_top = self.press[0]
                temperature_cloud_top = self.temp[0]

            self.make_press_temp_terr(log_ground_pressure=p_surf,layers=layers)
            pressure_full = self.press
            temperature_full = self.temp
            ind = np.where(self.press > 10**self.phys_vars['log_P0'])
            pressure_full[ind] = np.nan
            temperature_full[ind] = np.nan

            # Calculate the pressure temperature profile corresponding to the set of parameters
            if ((self.settings['clouds'] == 'opaque') and (i>0)):
                self.phys_vars['log_P0'] = self.cloud_opacity_cut[-1]
                self.make_press_temp_terr(layers=layers)
            else:
                self.make_press_temp_terr(layers=layers)

            # store the calculated values
            if i == 0:
                if not hasattr(self, 'input_temperature'):
                    results['input_pressure'] = self.press
                    results['input_temperature'] = self.temp
                if len(self.cloud_vars) != 0:
                    results['true_pressure_cloud_top'] = [pressure_cloud_top]
                    results['true_temperature_cloud_top'] = [temperature_cloud_top]
            elif i == 1:
                results['best_pressure'] = self.press
                results['best_temperature'] = self.temp
                if len(self.cloud_vars) != 0:
                    results['best_pressure_cloud_top'] = [pressure_cloud_top]
                    results['best_temperature_cloud_top'] = [temperature_cloud_top]

            else:
                if (i == 2) or (i==ind_start):
                    if i==2:
                        size = ind_end-ind_start-2
                    else:
                        size = ind_end-ind_start

                    # Initialize the arrays for storage
                    results['pressure'] = np.zeros((size,len(self.press)))
                    results['temperature'] = np.zeros((size,len(self.temp)))
                    results['pressure_full'] = np.zeros((size,len(pressure_full)))
                    results['temperature_full'] = np.zeros((size,len(temperature_full)))
                    if len(self.cloud_vars) != 0:
                        results['pressure_cloud_top'] = np.zeros((size,len([pressure_cloud_top])))
                        results['temperature_cloud_top'] = np.zeros((size,len([temperature_cloud_top])))

                if process == 0:
                    save = i - 2
                else:
                    save = i-ind_start

                # Save the results
                results['pressure'][save,:] = self.press
                results['temperature'][save,:] = self.temp
                results['pressure_full'][save,:] = pressure_full
                results['temperature_full'][save,:] = temperature_full
                if len(self.cloud_vars) != 0:
                    results['pressure_cloud_top'][save,:] = pressure_cloud_top
                    results['temperature_cloud_top'][save,:] = temperature_cloud_top

            # Print status of calculation
            if process == 0:
                t_end = t.time()
                remain_time = (t_end-t_start)/((i+1)/(ind_end-ind_start))-(t_end-t_start)
                print('\t'+str(np.round((i+1)/(ind_end-ind_start)*100,2))+' % of PT-profiles calculated. Estimated time remaining: '+str(remain_time//3600)+
                        ' h, '+str((remain_time%3600)//60)+' min.        ', end = "\r")

        # Print status of calculation
        if process == 0:
            print('\nPT-profile calculation completed.')
        
        # Return the results
        return results



    def Calc_Spectra(self,skip,n_processes=None,process=None):
        '''
        Function to calculate the fluxes corresponding to the retrieved posterior distributions
        for subsequent plotting in the flux plotting functions.
        '''

        # Add the thest fit spectrum and the input parameters to the equal weighted posterior
        temp_equal_weighted_post = np.append(np.array([self.best_fit]),self.equal_weighted_post[:,:-1],axis=0)
        temp_equal_weighted_post = np.append(np.array([self.truths]),temp_equal_weighted_post,axis=0)

        # Fetch the known parameters
        temp_vars_known, phys_vars_known, chem_vars_known, cloud_vars_known = self.__get_knowns()

        # Iterate over the equal weighted posterior distribution using the user-defined skip value
        dimension = np.shape(temp_equal_weighted_post)[0]//skip

        #If run in multiprocessing split up the jobs
        if (n_processes is not None) and (process is not None):

            ind_start = process*dimension//n_processes
            ind_end = min(dimension,(process+1)*dimension//n_processes)

            # Printing infor ffor the multiprocessing case
            if process == 0:
                print('\n-----------------------------------------------------')
                print('\n    Spectrum calculation on multiple CPUs:')
                print('')
                print('    Spectra to calculate: ',str(dimension))
                print('    Number of processes: '+str(n_processes))
                print('')
                print('    Distribution of tasks:')
                for proc_ind in range(n_processes):
                    print('\tProcess '+str(proc_ind)+':\tSpectra:\t'+str(proc_ind*dimension//n_processes+1)+'-'+str(min(dimension,(proc_ind+1)*dimension//n_processes)))
                print('\n-----------------------------------------------------\n')
        else:
            ind_start = 0
            ind_end = dimension
            process = 0

        # Import petitRADTRANS
        sys.path.append(self.path_prt)
        from petitRADTRANS import Radtrans
        from petitRADTRANS import nat_cst as nc
        self.init_rt()
        self.read_data(retrieval = False)

        # Print status of calculation
        if process == 0:
            print('Starting spectrum calculation.')
            print('\t0.00 % of spectra calculated.', end = "\r")

        results = {}
        t_start = t.time()
        for i in range(ind_start,ind_end):
            ind = min(2,i)+skip*max(0,i-2)

            # Fetch the retrieved parameters for a given ind
            self.__get_retrieved(temp_vars_known,chem_vars_known,phys_vars_known,cloud_vars_known,temp_equal_weighted_post,ind)

            # Scaling physical variables of the system to correct units
            try:
                self.phys_vars['d_syst'] = self.phys_vars['d_syst'] * nc.pc / 100
            except:
                print("ERROR! Distance from the star is missing!")
                sys.exit()
            try:
                self.phys_vars['R_pl'] = self.phys_vars['R_pl'] * nc.r_earth
            except:
                print("ERROR! Planetary radius is missing!")
                sys.exit()

            # Test the values of P0 and g and change to required values if necessary
            self.__g_test()
            self.__P0_test()

            # Calculate the pressure temperature profile corresponding to the set of parameters
            self.make_press_temp_terr()
            self.rt_object.setup_opa_structure(self.press)

            # Calculate the cloud bottom pressure from the cloud thickness parameter
            for key in self.cloud_vars.keys():
                self.cloud_vars[key]['bottom_pressure'] = self.cloud_vars[key]['top_pressure']+self.cloud_vars[key]['thickness']

            # Ensure that the total atmospheric weight is equal to 1
            metal_sum = sum(self.chem_vars.values())
            self.inert = (1-metal_sum) *np.ones_like(self.press)

            # Calculate the forward model; this returns the wavelengths in cm
            # and the flux F_nu in erg/cm^2/s/Hz.
            self.retrieval_model_plain()

            # Convert the calculated flux to the flux recieved at earth per m^2
            if self.phys_vars['d_syst'] is not None:
                self.rt_object.flux *= self.phys_vars['R_pl']**2/self.phys_vars['d_syst']**2

            for instrument in self.dwlen.keys():  # CURRENTLY USELESS
                # Rebin the spectrum according to the input spectrum
                if not np.size(self.rt_object.freq) == np.size(self.dwlen[instrument]):
                    self.rt_object.flux = spectres.spectres(self.dwlen[instrument],
                                                        nc.c / self.rt_object.freq,
                                                        self.rt_object.flux)

                #Store the calculated flux according to the considered case
                if i == 0:
                    results['wavelength'] = nc.c/self.rt_object.freq/1e-4 #self.dwlen[instrument]
                    results['true_flux'] = np.array(self.rt_object.flux)

                elif i == 1:
                    results['best_flux'] = np.array(self.rt_object.flux)

                else:
                    if (i == 2) or (i==ind_start):
                        if i==2:
                            size = ind_end-ind_start-2
                        else:
                            size = ind_end-ind_start

                        # Initialize the arrays for storage
                        results['retrieved_fluxes'] = np.zeros((size,len(self.rt_object.flux)))

                    if process == 0:
                        save = i - 2
                    else:
                        save = i-ind_start
                
                    # Save the results
                    results['retrieved_fluxes'][save,:] = self.rt_object.flux
            
            # Print status of calculation
            if process == 0:
                t_end = t.time()
                remain_time = (t_end-t_start)/((i+1)/(ind_end-ind_start))-(t_end-t_start)
                print('\t'+str(np.round((i+1)/(ind_end-ind_start)*100,2))+' % of spectra calculated. Estimated time remaining: '+str(remain_time//3600)+
                        ' h, '+str((remain_time%3600)//60)+' min.            ', end = "\r")

        # Print status of calculation
        if process == 0:
            print('\nSpectrum calculation completed.')

        #return the calculated results
        return results



    def get_spectra(self,skip=1,n_processes=50):
        '''
        gets the PT profiles corresponding to the parameter values 
        of the equal weighted posteriors.
        '''

        # If not yet done calculate the data corresponding
        # to the retrieved posterior distributions of partameters
        if not hasattr(self, 'retrieved_fluxes'):
            function_args = {'skip':skip,'n_processes':n_processes}
            self.__get_data(data_type='Spec',data_name='spectra',function_name='Calc_Spectra',function_args=function_args)



    def get_pt(self,skip=1,layers=500,p_surf=4,n_processes=50):
        '''
        gets the PT profiles corresponding to the parameter values 
        of the equal weighted posteriors.
        '''

        # If not yet done calculate the data corresponding
        # to the retrieved posterior distributions of partameters
        if not hasattr(self, 'pressure'):
            function_args = {'skip':skip,'layers':layers,'p_surf':p_surf,'n_processes':n_processes}
            self.__get_data(data_type='PT',data_name='PT profiles',function_name='Calc_PT_Profiles',function_args=function_args)



    def __get_data(self,data_type,data_name,function_name,function_args):
        '''
        gets the data corresponding to the parameter values 
        of the equal weighted posteriors.
        '''

        # check if the data for the specified skip
        # values are already calculated
        try:
            load_file = open(self.results_directory+'Plots/Ret_'+data_type+'_Skip_'+str(function_args['skip'])+'.pkl', "rb")
            loaded_data = pickle.load(load_file)
            load_file.close()

            # Initialize class attributes for the data from the different processes
            for key in loaded_data.keys():
                setattr(self,key,loaded_data[key])

            print('Loaded previously calculated '+data_name+' from '+self.results_directory+'Plots/Ret_'+data_type+'_Skip_'+str(function_args['skip'])+'.pkl.')

        # check if the data for the specified skip
        # values are already calculated    
        except:
            print('Calculating retrieved '+data_name+' from scratch.')

            # Check that we do not use too many CPUs
            if (np.shape(self.equal_weighted_post)[0]//function_args['skip'])//function_args['n_processes'] < 3:
                print('Not enough jobs for the specified number of processes!')
                while (np.shape(self.equal_weighted_post)[0]//function_args['skip'])//function_args['n_processes'] < 3:
                    function_args['n_processes'] -= 1
                print('I lowered n_processes to '+str(function_args['n_processes'])+'.')

            # Start the paralel calculation
            parallel_calculation = rp_parallel.parallel(function_args['n_processes'])
            result_process = parallel_calculation.calculate(self.results_directory,function_name,function_args)

            # Combine the data from the different processes
            print('Combining all data.')
            result_combined = {}
            for key in result_process[0].keys():
                combined = result_process[0][key].copy()
                for process in range(1,function_args['n_processes']):
                    try:
                        combined = np.append(combined,result_process[process][key],axis = 0)
                    except:
                        pass
                result_combined[key] = combined

                # Initialize class attributes for the data from the different processes
                setattr(self,key,combined)
            print('Done combining all data.')

            # Save the calculated data in a pickle file for later reloading to save time
            save_file = open(self.results_directory+'Plots/Ret_'+data_type+'_Skip_'+str(function_args['skip'])+'.pkl', "wb")
            pickle.dump(result_combined, save_file)
            save_file.close()
            print('Saved calculated '+data_name+' in '+self.results_directory+'Plots/Ret_'+data_type+'_Skip_'+str(function_args['skip'])+'.pkl.')





    """
    #################################################################################
    #                                                                               #
    #   Routines for generating cornerplots.                                        #
    #                                                                               #
    #################################################################################
    """



    def Posteriors(self, save=False, plot_corner=True, log_pressures=True, log_mass=True, log_abundances=True, log_particle_radii=True, plot_pt=True, plot_physparam=True,
                    plot_clouds=True,plot_chemcomp=True,plot_bond=None, titles = None, units=None, bins=20, quantiles1d=[0.16, 0.5, 0.84],color='k'):
        '''
        This function generates a corner plot for the retrieved parameters.
        '''

        # Define the dimension of the corner plot, which is equal to the number of retrieved parameters 
        # Copy the equal weigted posterior and truths files to ensure that no changes are made to the file
        dimension = self.n_params
        
        local_equal_weighted_post = np.copy(self.equal_weighted_post)
        local_truths = self.truths.copy()
        local_titles = self.titles.copy()

        inds_pt = []
        inds_physparam = []
        inds_chemcomp = []
        inds_clouds = []

        param_names = list(self.params.keys())
        for i in range(len(param_names)):
            if self.params[param_names[i]]['type'] == 'CHEMICAL COMPOSITION PARAMETERS':
                inds_chemcomp += [i]
            if self.params[param_names[i]]['type'] == 'CLOUD PARAMETERS':
                inds_clouds += [i]
            if self.params[param_names[i]]['type'] == 'PHYSICAL PARAMETERS':
                inds_physparam += [i]
            # Adjust retrieved abundances for the line absorbers
            if self.params[param_names[i]]['type'] == 'TEMPERATURE PARAMETERS':
                inds_pt += [i]
                #Plotting for the special upper limit prior case
                if self.priors[i] == 'THU':
                    local_equal_weighted_post[:,i] = local_equal_weighted_post[:,i]**3
                    local_titles[i] = r'sqrt$_3(' + local_titles[i] + '$)$'
                    if not local_truths[i] is None:
                        local_truths[i] = local_truths[i]**3
                else:
                    pass

        # If we want to use log abundnces update data to log abundances
        if log_abundances:
            param_names = list(self.params.keys())
            for i in range(len(param_names)):
                # Adjust retrieved abundances for the line absorbers
                if self.params[param_names[i]]['type'] == 'CHEMICAL COMPOSITION PARAMETERS':
                    #Plotting for the special upper limit prior case
                    if self.priors[i] == 'ULU':
                        local_equal_weighted_post[:,i] = np.log10(1-local_equal_weighted_post[:,i])
                        local_titles[i] = 'log$_{10}(1-$ ' + local_titles[i] + '$)$'
                        if not local_truths[i] is None:
                            local_truths[i] = np.log10(1-local_truths[i])
                    else:
                        local_equal_weighted_post[:,i] = np.log10(local_equal_weighted_post[:,i])
                        local_titles[i] = 'log$_{10}$ ' + local_titles[i]
                        if not local_truths[i] is None:
                            local_truths[i] = np.log10(local_truths[i])
                # Adjust retrieved abundances for the clod absorbers
                if self.params[param_names[i]]['type'] == 'CLOUD PARAMETERS':
                    if len(param_names[i].split('_')) == 2:
                        local_equal_weighted_post[:,i] = np.log10(local_equal_weighted_post[:,i])
                        local_titles[i] = 'log$_{10}$ ' + local_titles[i]
                        if not local_truths[i] is None: 
                            local_truths[i] = np.log10(local_truths[i])
        
        # If we want to use log particle radii
        if log_particle_radii:
            param_names = list(self.params.keys())
            for i in range(len(param_names)):
                if self.params[param_names[i]]['type'] == 'CLOUD PARAMETERS':
                    if param_names[i] == 'H2SO484(c)_am_particle_radius':
                        local_equal_weighted_post[:,i] = np.log10(local_equal_weighted_post[:,i])
                        local_titles[i] = 'log$_{10}$ ' + local_titles[i]
                        if not local_truths[i] is None: 
                            local_truths[i] = np.log10(local_truths[i])
        
        # If we want to use log mass in the corner plot
        if log_mass:
            param_names = list(self.params.keys())
            for i in range(len(param_names)):
                if self.params[param_names[i]]['type'] == 'PHYSICAL PARAMETERS':
                    if param_names[i] == 'M_pl':
                        local_equal_weighted_post[:,i] = np.log10(local_equal_weighted_post[:,i])
                        local_titles[i] = 'log$_{10}$ ' + local_titles[i]
                        if not local_truths[i] is None: 
                            local_truths[i] = np.log10(local_truths[i])

        # If we want to use log pressures update data to log pressures
        if log_pressures:
            param_names = list(self.params.keys())
            for i in range(len(param_names)):
                # Adjust retrieved abundances for the clod absorbers
                if self.params[param_names[i]]['type'] == 'CLOUD PARAMETERS':
                    var_type = param_names[i].split('_',2)[-1]
                    if (var_type == 'top_pressure') or (var_type == 'thickness'):
                        local_equal_weighted_post[:,i] = np.log10(local_equal_weighted_post[:,i])
                        local_titles[i] = 'log$_{10}$ ' + local_titles[i]
                        if not local_truths[i] is None: 
                            local_truths[i] = np.log10(local_truths[i])
                if self.params[param_names[i]]['type'] == 'PHYSICAL PARAMETERS':
                    if 'P0' in param_names[i]:
                        if not 'log' in param_names[i]:
                            local_equal_weighted_post[:,i] = np.log10(local_equal_weighted_post[:,i])
                            if not local_truths[i] is None: 
                                local_truths[i] = np.log10(local_truths[i])
                            local_titles[i] = 'log$_{10}$ P$_0$'
                        else:
                            local_titles[i] = 'log$_{10}$ P$_0$'
                    

        # add all wanted parameters to the corner plot
        inds = []
        if plot_pt:
            inds += [i for i in range(len(param_names)) if i in inds_pt]
        if plot_physparam:
            inds += [i for i in range(len(param_names)) if i in inds_physparam]
        if plot_chemcomp:
            inds += [i for i in range(len(param_names)) if i in inds_chemcomp]
        if plot_clouds:
            inds += [i for i in range(len(param_names)) if i in inds_clouds]

        def none_test(input,inds):
            try:
                return [input[i] for i in inds]
            except:
                return None

        # If wanted add the bond albedo and the equilibrium temperature to the plot
        if plot_bond is not None:
            if not hasattr(self, 'A_Bond_ret'):
                self.Plot_Ret_Bond_Albedo(*plot_bond[:-2],A_Bond_true = plot_bond[-1], T_equ_true=plot_bond[-2],save = True,bins=20)
            local_equal_weighted_post = np.append(local_equal_weighted_post, self.ret_opaque_T,axis=1)
            local_equal_weighted_post = np.append(local_equal_weighted_post, self.A_Bond_ret,axis=1)
            local_truths += plot_bond[-2:]
            inds += [-2,-1]
            local_titles += [r'$\mathrm{T_{eq,\,Planet}}$',r'$\mathrm{A_{B,\,Planet}}$']

        
        if not titles is None:
            local_titles=titles

        if plot_corner:        
            fig, axs = rp_posteriors.Corner(local_equal_weighted_post[:,inds],local_titles[inds],dimension = len(inds),truths=none_test(local_truths,inds),
                                quantiles1d=quantiles1d,units=none_test(units,inds),bins=bins,color=color)
            # Save the figure or retrun the figure object
            if save:
                plt.savefig(self.results_directory+'Plots/plot_corner.pdf', bbox_inches='tight')
            return fig, axs
        else:



            
            if not os.path.exists(self.results_directory + 'Plots/Posteriors/'):
                os.makedirs(self.results_directory + 'Plots/Posteriors/')
            for i in inds:
                fig, axs = rp_posteriors.Posterior(local_equal_weighted_post[:,i],local_titles[i],truth=none_test(local_truths,[i]),
                                    quantiles1d=quantiles1d,units=none_test(units,[i]),bins=bins,color=color)
                if save:
                    plt.savefig(self.results_directory+'Plots/Posteriors/'+list(self.params.keys())[i]+'.pdf', bbox_inches='tight')
                plt.close(fig)





    """
    #################################################################################
    #                                                                               #
    #   Routines for generating PT profile plots.                                   #
    #                                                                               #
    #################################################################################
    """



    def PT_Envelope(self, save=False, plot_residual = False, skip=1, plot_clouds = False, x_lim =[0,1000], y_lim = [1e-6,1e4], quantiles=[0.05,0.15,0.25,0.35,0.65,0.75,0.85,0.95],
                    quantiles_title = None, inlay_loc='upper right', bins_inlay = 20, ax = None, color='C2', case_identifier = '', legend_loc = 'best',n_processes=50,true_cloud_top=None):
        '''
        This Function creates a plot that visualizes the absolute uncertainty on the
        retrieval results in comparison with the input PT profile for the retrieval.
        '''

        self.get_pt(skip=skip,n_processes=n_processes)

        # find the quantiles for the different pressures and temperatures
        p_layers_quantiles = [np.nanquantile(self.pressure_full,q,axis=0) for q in quantiles]
        if plot_residual:
            T_layers_quantiles = [np.nanquantile(self.temperature_full,q,axis=0)-np.nanquantile(self.temperature_full,0.5,axis=0) for q in quantiles]
        else:
            T_layers_quantiles = [np.nanquantile(self.temperature_full,q,axis=0) for q in quantiles]
        not_nan = np.count_nonzero(~np.isnan(self.pressure_full),axis = 0)/np.shape(self.pressure_full)[0]

        for q in range(len(quantiles)):
            T_layers_quantiles[q][np.where(not_nan<min(2*(1-quantiles[q]),2*(quantiles[q])))] = np.nan
        
            notnan = ~np.isnan(T_layers_quantiles[q])
            T_layers_quantiles[q] = T_layers_quantiles[q][notnan]
            p_layers_quantiles[q] = p_layers_quantiles[q][notnan]

            X_Y_Spline = scp.make_interp_spline(np.array(p_layers_quantiles[q]),np.array(T_layers_quantiles[q]))
            p_layers_quantiles[q] = np.logspace(np.log10(p_layers_quantiles[q].min()),np.log10(p_layers_quantiles[q].max()),80)
            T_layers_quantiles[q] = X_Y_Spline(p_layers_quantiles[q])
            
        # If wanted find the quantiles for cloud top and bottom pressures
        if plot_clouds:
            median_temperature_cloud_top, median_pressure_cloud_top = np.median(self.temperature_cloud_top), np.median(self.pressure_cloud_top)
            cloud_top_quantiles = [np.quantile(self.pressure_cloud_top,q) for q in quantiles]

        # Generate colorlevels for the different quantiles
        color_levels, level_thresholds, N_levels = rp_col.color_levels(color,quantiles)

        # Start of the plotting
        ax_arg = ax
        if ax is None:
            figure = plt.figure()
            ax = figure.add_axes([0.1, 0.1, 0.8, 0.8])
        else:
            pass

        # Main plot
        # Plotting the retrieved PT profile
        for i in range(N_levels):
            ax.fill(np.append(T_layers_quantiles[i],np.flip(T_layers_quantiles[-i-1])),
                    np.append(p_layers_quantiles[i],np.flip(p_layers_quantiles[-i-1])),color = tuple(color_levels[i, :]),lw = 0,clip_box=True)
        if plot_residual:
            ax.semilogy([0,0], [0,1000],color ='black', linestyle=':')
            ax.annotate('Retrieved PT Median',[0-0.025*x_lim[1],10**(0.975*(np.log10(y_lim[1])-np.log10(y_lim[0]))+np.log10(y_lim[0]))],color = 'black',rotation=90,ha='right')

        # If wanted: plotting the retrieved cloud top
        if plot_clouds:
            for i in range(int(len(quantiles)/2)):
                alpha = 0.1+i*0.15
                if i == len(quantiles)-i-2:
                    ax.fill([-10000,10000,10000,-10000],[cloud_top_quantiles[i],cloud_top_quantiles[i],cloud_top_quantiles[i+1],cloud_top_quantiles[i+1]],color = 'black',alpha=alpha,lw = 0)
                else:
                    ax.fill([-10000,10000,10000,-10000],[cloud_top_quantiles[i],cloud_top_quantiles[i],cloud_top_quantiles[i+1],cloud_top_quantiles[i+1]],color = 'black',alpha=alpha,lw = 0)
                    ax.fill([-10000,10000,10000,-10000],[cloud_top_quantiles[-i-2],cloud_top_quantiles[-i-2],cloud_top_quantiles[-i-1],cloud_top_quantiles[-i-1]],color = 'black',alpha=alpha,lw = 0)
            ax.plot([-10000,10000],[median_pressure_cloud_top,median_pressure_cloud_top],'k--')
            ax.annotate('Retrieved Median Cloud Top',[x_lim[0]+0.0125*(x_lim[1]-x_lim[0]),10**(-0.025*(np.log10(y_lim[1])-np.log10(y_lim[0]))+np.log10(median_pressure_cloud_top))],color = 'black',ha='left')

        # Plotting the true/input profile (interpolation for smoothing)
        if plot_residual:
            y = np.nanquantile(self.temperature_full,0.5,axis=0)
            x = np.nanquantile(self.pressure_full,0.5,axis=0)
            yinterp = np.interp(self.input_pressure, x, y)
            smooth_T_true = gaussian_filter1d(self.input_temperature-yinterp,sigma = 10)

            # Check if the retrieved PT profile reaches al the way to the true surface and plot accordingly.
            if np.isnan(smooth_T_true[-10]):
                num_nan = np.count_nonzero(np.isnan(smooth_T_true))
                ax.semilogy(smooth_T_true[:-num_nan-10],self.input_pressure[:-num_nan-10],color ='black', label = 'Input Profile')
                ax.semilogy(smooth_T_true[-num_nan-10:],self.input_pressure[-num_nan-10:],color ='black', ls = ':')
            else:
                ax.semilogy(smooth_T_true,self.input_pressure,color ='black', label = 'Input Profile')

                # Plotting the true/input surface temperature/pressure
                ax.plot(self.input_temperature[-1]-yinterp[-1],self.input_pressure[-1],marker='s',color='C3',ms=7, markeredgecolor='black',lw=0,label = 'Input Surface')

            # If wanted: plotting the true/input cloud top temperature/pressure
            try:
                if true_cloud_top is not None:
                    self.true_temperature_cloud_top,self.true_pressure_cloud_top = true_cloud_top[1],true_cloud_top[0]
                ind_ct = (np.argmin(np.abs(np.log10(self.true_pressure_cloud_top)-np.log10(self.input_pressure))))
                ax.plot(smooth_T_true[ind_ct],self.true_pressure_cloud_top,marker='o',color='C1',lw=0,ms=7, markeredgecolor='black',label = 'Input Cloud Top')
            except:
                pass
        else:
            ax.semilogy(self.input_temperature,self.input_pressure,color ='black', label = 'Input Profile')

            # Plotting the true/input surface temperature/pressure
            ax.plot(self.input_temperature[-1],self.input_pressure[-1],marker='s',color='C3',ms=7, markeredgecolor='black',lw=0,label = 'Input Surface')

            # If wanted: plotting the true/input cloud top temperature/pressure
            try:
                if true_cloud_top is not None:
                    self.true_temperature_cloud_top,self.true_pressure_cloud_top = true_cloud_top[1],true_cloud_top[0]
                ax.plot(self.true_temperature_cloud_top,self.true_pressure_cloud_top,marker='o',color='C1',lw=0,ms=7, markeredgecolor='black',label = 'Input Cloud Top')
            except:
                pass

        # Print the case identifier
        ax.annotate(case_identifier,[x_lim[1]-0.05*(x_lim[1]-x_lim[0]),10**(np.log10(y_lim[1])-0.05*(np.log10(y_lim[1])-np.log10(y_lim[0])))],ha='right',va='bottom')

        # If it is a single plot show the axes titles
        if ax_arg is None:
            if plot_residual:
                ax.set_xlabel('Temperature Relative to Retrieved Median [K]')
            else:
                ax.set_xlabel('Temperature [K]')
            ax.set_ylabel('Pressure [bar]')
        
        # Set the limits for the plot axes
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.invert_yaxis()

        # Inlay plot
        # generate and position the inlay plot
        ax2 = rp_inlay.position_inlay(inlay_loc,figure,ax_arg,ax)

        # Plotting the cloud top temperature/pressure
        if self.settings['clouds'] == 'opaque':
            # Define the plot titles
            ax2_xlabel = r'$T^\mathrm{cloud}_\mathrm{top}\,\,\left[\mathrm{K}\right]$'
            ax2_ylabel = r'$P^\mathrm{cloud}_\mathrm{top}\,\,\left[\mathrm{bar}\right]$'

            # Define limits and make a 2d histogram of the cloud top pressures and temperatures
            t_lim = [np.min(self.temperature_cloud_top),np.max(self.temperature_cloud_top)]
            t_range = t_lim[1]-t_lim[0]
            p_lim = [np.min(np.log10(self.pressure_cloud_top)),np.max(np.log10(self.pressure_cloud_top))]
            p_range = p_lim[1]-p_lim[0]
            Z,X,Y=np.histogram2d(self.temperature_cloud_top[:,0],np.log10(self.pressure_cloud_top)[:,0],bins=bins_inlay,
                range = [[t_lim[0]-0.1*t_range,t_lim[1]+0.1*t_range],[p_lim[0]-0.1*p_range,p_lim[1]+0.1*p_range]])

        else:
            # Define the plot titles
            ax2_xlabel = r'$\mathrm{T_0}\,\,\left[\mathrm{K}\right]$'
            ax2_ylabel = r'$\mathrm{P_0}\,\,\left[\mathrm{bar}\right]$'

            # Define limits and make a 2d histogram of the surface pressures and temperatures
            t_lim = [np.min(self.temperature[:,-1]),np.max(self.temperature[:,-1])]
            t_range = t_lim[1]-t_lim[0]
            p_lim = [np.min(np.log10(self.pressure[:,-1])),np.max(np.log10(self.pressure[:,-1]))]
            p_range = p_lim[1]-p_lim[0]
            Z,X,Y=np.histogram2d(self.temperature[:,-1],np.log10(self.pressure[:,-1]),bins=bins_inlay,
                range = [[t_lim[0]-0.1*t_range,t_lim[1]+0.1*t_range],[p_lim[0]-0.1*p_range,p_lim[1]+0.1*p_range]])

        # Generate the colormap and plot the contours of the 2d histogram
        map, norm, levels = rp_col.color_map(Z,color_levels,level_thresholds)
        contour = ax2.contourf((X[:-1]+X[1:])/2,10**((Y[:-1]+Y[1:])/2),Z.T,cmap=map,norm=norm,levels=np.array(levels))

        # plot the true values that were used to generate the input spectrum
        ax2.plot(self.input_temperature,self.input_pressure,color ='black')
        ax2.plot(self.input_temperature[-1],self.input_pressure[-1],marker='s',color='C3',lw=0,ms=7, markeredgecolor='black')
        try:
            ax2.plot(self.true_temperature_cloud_top,(self.true_pressure_cloud_top),marker='o',color='C1',lw=0,ms=7, markeredgecolor='black')
        except:
            pass
        
        # Arange the ticks for the inlay
        rp_inlay.axesticks_inlay(ax2,ax2_xlabel,ax2_ylabel,inlay_loc)

        # Find the minima and maxima of the outermost contour
        t_lim = [np.min([np.min(contour.allsegs[0][i][:,0]) for i in range(len(contour.allsegs[0]))]), np.max([np.max(contour.allsegs[0][i][:,0]) for i in range(len(contour.allsegs[0]))])]
        p_lim = [np.min([np.min(contour.allsegs[0][i][:,1]) for i in range(len(contour.allsegs[0]))]), np.max([np.max(contour.allsegs[0][i][:,1]) for i in range(len(contour.allsegs[0]))])]

        # Find the limits for the inlay plot from the contours (+- 10%)
        # if the span in pressure exceeds 2 orders of magnitude use log axes 
        ax2_xlim = [t_lim[0]-0.1*(t_lim[1]-t_lim[0]),t_lim[1]+0.1*(t_lim[1]-t_lim[0])]
        if np.log10(p_lim[1])-np.log10(p_lim[0]) >= 1.2:
            log_p = True
            ax2_ylim = [10**(np.log10(p_lim[0])-0.1*(np.log10(p_lim[1])-np.log10(p_lim[0]))),10**(np.log10(p_lim[1])+0.1*(np.log10(p_lim[1])-np.log10(p_lim[0])))]
            ax2.set_yscale('log')
        else:
            log_p = False
            ax2_ylim = [max([p_lim[0]-0.1*(p_lim[1]-p_lim[0]),0]),p_lim[1]+0.1*(p_lim[1]-p_lim[0])]

        # Set the limits and ticks for the axes
        # x axis
        xticks = np.array([(1-pos)*ax2_xlim[0]+pos*ax2_xlim[1] for pos in [0.2,0.4,0.6,0.8]])
        roundx = np.log10(np.abs(xticks[1]-xticks[0]))
        ax2.set_xticks(xticks)
        if roundx>=0.5:
            ax2.set_xticklabels(((xticks*10**(-roundx)).astype(int)*10**(roundx)).astype(int),rotation=90)
        else:
            ax2.set_xticklabels(np.round(xticks,int(-np.floor(roundx-0.5))),rotation=90)
        ax2.set_xlim(ax2_xlim)

        # y axis
        if log_p:
            log_range = np.floor(np.log10(ax2_ylim)).astype(int)
            yticks = [10**i for i in range(log_range[0]+1,log_range[1]+1)]
            ax2.set_yticks(yticks)
        else:
            yticks = np.array([(1-pos)*ax2_ylim[0]+pos*ax2_ylim[1] for pos in [0.2,0.4,0.6,0.8]])
            roundy = np.log10(np.abs(yticks[1]-yticks[0]))
            ax2.set_yticks(yticks)
            if roundy>=0.5:
                ax2.set_yticklabels(((yticks*10**(-roundy)).astype(int)*10**(roundy)).astype(int))
            else:
                ax2.set_yticklabels(np.round(yticks,int(-np.floor(roundy-0.5))))
        ax2.set_ylim(ax2_ylim[::-1])

        # Legend cosmetics
        handles, labels = ax.get_legend_handles_labels()
        if ax_arg is not None:
            pass
        else:
            # Add the patches to the legend
            if plot_clouds:
                patch_handles = [rp_hndl.MulticolorPatch([tuple(color_levels[i, :]),tuple(3*[0.9-i*0.15])],[1,1]) for i in range(N_levels)]
            else:
                patch_handles = [rp_hndl.MulticolorPatch([tuple(color_levels[i, :])],[1]) for i in range(N_levels)]

            # Define the titles for the patches
            if quantiles_title is None:
                patch_labels = [str(quantiles[i])+'-'+str(quantiles[-i-1]) for i in range(N_levels)]
            else:
                patch_labels = quantiles_title
            
            # Add the legend
            lgd = ax.legend(handles+patch_handles,labels+patch_labels,\
                            handler_map={str:  rp_hndl.Handles(), rp_hndl.MulticolorPatch:  rp_hndl.MulticolorPatchHandler()}, ncol=1,loc=legend_loc,frameon=False)

        # Save or pass back the figure
        if ax_arg is not None:
            return handles+['CloudTopPressure','SurfacePressure'], labels+['True Cloud Top','True Surface']
        elif save:
            if plot_residual:
                plt.savefig(self.results_directory+'Plots/plot_pt_structure_residual.pdf', bbox_inches='tight',bbox_extra_artists=(lgd,))
            else:
                plt.savefig(self.results_directory+'Plots/plot_pt_structure.pdf', bbox_inches='tight',bbox_extra_artists=(lgd,))
        return figure, ax, ax2



    def PT_Histogram(self, ax=None, save=False, plot_clouds=False, x_lim=[0,1000], y_lim=[1e-6,1e4], color_map=None, skip=1, bins=200, legend_color='white', truth_color='white',
                    legend_loc = 'best',n_processes=50):
        '''
        This Function creates a plot that visualizes the 2d histogram of the retrieved
        2d PT profile in comparison with the PT profile spectrum for the retrieval.
        '''

        self.get_pt(skip=skip,n_processes=n_processes)

        # Create colormap if requaested
        if color_map is not None:
            color_map = rp_col.uniform_color_map(color_map)
        else:
            color_map = 'afmhot'

        # Start of the plotting
        ax_arg = ax
        if ax is None:
            figure = plt.figure()
            ax = figure.add_axes([0.1, 0.1, 0.8, 0.8])
        else:
            pass
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        # Plotting the histogram of the PT profiles in the equal weighted posterior
        bins = np.array([np.linspace(x_lim[0],x_lim[1],bins),np.logspace(np.log10(y_lim[0]),np.log10(y_lim[1]),bins)])
        ax.hist2d(self.temperature_full.flatten()[~np.isnan(self.temperature_full.flatten())],self.pressure_full.flatten()[~np.isnan(self.pressure_full.flatten())],bins=bins,cmap=color_map)
        
        # Plotting the input profile
        ax.semilogy(self.input_temperature,self.input_pressure,color = truth_color, label = 'Input Profile')

        # Plotting the true/input surface temperature/pressure
        ax.plot(self.input_temperature[-1],self.input_pressure[-1],marker='s',color='C3',ms=7, markeredgecolor= truth_color,lw=0,label = 'Input Surface')

        # If wanted: plotting the true/input cloud top temperature/pressure
        if plot_clouds:
            ax.plot(self.true_temperature_cloud_top,self.true_pressure_cloud_top,marker='o',color='C1',lw=0,ms=7, markeredgecolor = truth_color,label = 'Input Cloud Top')

        # If it is a single plot show the axes titles
        if ax_arg is None:
            ax.set_xlabel('Temperature [K]')
            ax.set_ylabel('Pressure [bar]')
        ax.invert_yaxis()

        # Legend cosmetics
        handles, labels = ax.get_legend_handles_labels()
        if ax_arg is not None:
            pass
        else:
            lgd = ax.legend(handles,labels, ncol=1,loc=legend_loc,frameon=False,labelcolor = legend_color)

        # Save or pass back the figure
        if ax_arg is not None:
            return handles, labels
        elif save:
            plt.savefig(self.results_directory+'Plots/plot_pt_histogram.pdf', bbox_inches='tight',bbox_extra_artists=(lgd,))
        return figure, ax





    """
    #################################################################################
    #                                                                               #
    #   Routine for generating Spectrum plots.                                      #
    #                                                                               #
    #################################################################################
    """



    def Flux_Error(self, save=False, plot_residual = False, skip=1, x_lim = None, y_lim = None, quantiles = [0.05,0.15,0.25,0.35,0.65,0.75,0.85,0.95],
                    quantiles_title = None, ax = None, color='C2', case_identifier = None, plot_noise = False, plot_true_spectrum = False, plot_datapoints = False,
                    noise_title = 'Observation Noise', plot_best_fit = False, legend_loc = 'best', n_processes=50):
        '''
        This Function creates a plot that visualizes the absolute uncertainty on the
        retrieval results in comparison with the input spectrum for the retrieval.
        '''

        self.get_spectra(skip=skip,n_processes=n_processes)

        # Find the quantiles for the different spectra
        if plot_residual:
            flux_quantiles = [(np.quantile(self.retrieved_fluxes,q,axis=0)-self.input_flux)/self.input_flux*100 for q in quantiles]
        else:
            flux_quantiles = [np.quantile(self.retrieved_fluxes,q,axis=0) for q in quantiles]

        # Generate colorlevels for the different quantiles
        color_levels, level_thresholds, N_levels = rp_col.color_levels(color,quantiles)

        # Start of the plotting
        ax_arg = ax
        if ax is None:
            if plot_residual:
                figure = plt.figure(figsize=(12,2))
            else:
                figure = plt.figure()
            ax = figure.add_axes([0.1, 0.1, 0.8, 0.8])
        else:
            pass


        if plot_noise:
            if plot_residual:
                ax.fill(np.append(self.wavelength,np.flip(self.wavelength)),np.append((self.input_error)/self.input_flux*100,
                            np.flip((-self.input_error)/self.input_flux*100)),color = (0.8,0.8,0.8,1),lw = 0,clip_box=True)
            else:
                ax.fill(np.append(self.wavelength,np.flip(self.wavelength)),np.append(self.input_flux+self.input_error,
                            np.flip(self.input_flux-self.input_error)),color = (0.8,0.8,0.8,1),lw = 0,clip_box=True)

        # Plotting the retrieved Spectra
        for i in range(N_levels):
            ax.fill(np.append(self.wavelength,np.flip(self.wavelength)),
                    np.append(flux_quantiles[i],np.flip(flux_quantiles[-i-1])),color = tuple(color_levels[i, :]),lw = 0,clip_box=True)
        
        # Plotting the input spectra
        if plot_true_spectrum:
            if plot_residual:
                ax.plot(self.input_wavelength,self.input_flux*0,color = 'black',ls =':')
            else:
                ax.plot(self.input_wavelength,self.input_flux,color = 'black',ls='-',label = 'Input Spectrum')
                #ax.plot(self.input_wavelength,self.true_flux,color = 'black',ls='-',label = 'True Spectrum')
        if plot_datapoints:
            if not plot_residual:
                ax.errorbar(self.input_wavelength,self.input_flux,yerr=self.input_error,color = 'k',ms = 3,marker='o',ls='',label = 'Input Spectrum')
        
        # Plotting results for the best fit.
        if plot_best_fit:
            if plot_residual:
                ax.plot(self.wavelength,(self.best_flux-self.input_flux)/self.input_flux,color = 'C3',ls ='--',label = 'Best fit Spectrum')
            else:
                ax.plot(self.wavelength,self.best_flux,color = 'C3',ls ='--',label = 'Best fit Spectrum')

        # If it is a single plot show the axes titles
        if ax_arg is None:
            if plot_residual:
                ax.set_ylabel(r'Retrieval Residual $\left[\%\right]$')
            else:
                ax.set_ylabel(r'Flux at 10 pc $\left[\mathrm{\frac{erg}{s\,Hz\,m^2}}\right]$')
            ax.set_xlabel(r'Wavelength [$\mu$m]')

        # Set the limits for the plot axes
        if x_lim is not None:
            ax.set_xlim(x_lim)
        else:
            x_lim = [np.min(self.wavelength),np.max(self.wavelength)]
            ax.set_xlim([np.min(self.wavelength),np.max(self.wavelength)])

        if y_lim is not None:
            ax.set_ylim(y_lim)
        else:
            if plot_residual:
                y_lim=[-55,55]
                ax.set_ylim(y_lim)
            else:
                y_lim=[0,list(ax.get_ylim())[1]]
                ax.set_ylim(y_lim)
        
        # Print the case identifier
        if case_identifier is not None:
            if plot_residual:
                ax.annotate(case_identifier,[x_lim[1]-0.025*(x_lim[1]-x_lim[0]),y_lim[0]+0.1*(y_lim[1]-y_lim[0])],ha='right',va='bottom')
            else:
                ax.annotate(case_identifier,[x_lim[1]-0.05*(x_lim[1]-x_lim[0]),y_lim[0]+0.05*(y_lim[1]-y_lim[0])],ha='right',va='bottom')

        # Legend cosmetics
        handles, labels = ax.get_legend_handles_labels()
        if ax_arg is not None:
            pass
        else:
            # Add the patches to the legend
            patch_handles = [rp_hndl.MulticolorPatch([tuple(color_levels[i, :])],[1]) for i in range(N_levels)]
            if quantiles_title is None:
                patch_labels = [str(quantiles[i])+'-'+str(quantiles[-i-1]) for i in range(N_levels)]
            else:
                patch_labels = quantiles_title

            if plot_noise:
                patch_handles = [rp_hndl.MulticolorPatch([(0.8,0.8,0.8)],[1])]+patch_handles
                patch_labels = [noise_title]+patch_labels

            # Add the legend
            if plot_residual:
                lgd = ax.legend(handles+patch_handles,labels+patch_labels,
                    handler_map={str:  rp_hndl.Handles(), rp_hndl.MulticolorPatch:  rp_hndl.MulticolorPatchHandler()}, ncol=len(labels+patch_labels),loc=legend_loc,frameon=False)
            else:
                lgd = ax.legend(handles+patch_handles,labels+patch_labels,
                    handler_map={str:  rp_hndl.Handles(), rp_hndl.MulticolorPatch:  rp_hndl.MulticolorPatchHandler()}, ncol=1,loc=legend_loc,frameon=False)


        # Save or pass back the figure
        if ax_arg is not None:
            return handles+patch_handles,labels+patch_labels
        elif save:
            if plot_residual:
                plt.savefig(self.results_directory+'Plots/plot_spectrum_residual.pdf', bbox_inches='tight',bbox_extra_artists=(lgd,))
            else:
                plt.savefig(self.results_directory+'Plots/plot_spectrum.pdf', bbox_inches='tight',bbox_extra_artists=(lgd,))
        return figure, ax





    """
    #################################################################################
    #                                                                               #
    #   Plotting routine for the Bond Albedo                                        #
    #                                                                               #
    #################################################################################
    """



    def Plot_Ret_Bond_Albedo(self, L_star, sigma_L_star, sep_planet, sigma_sep_planet, A_Bond_true = None, T_equ_true= None,
                            skip=1, quantiles1d=[0.16, 0.5, 0.84], bins=50, save=False, plot=True, n_processes=50,
                            titles = [r'$\mathrm{L_{Star}}$',r'$\mathrm{a_{Planet}}$',r'$\mathrm{T_{eq,\,Planet}}$',r'$\mathrm{A_{B,\,Planet}}$'],
                            units = [r'$\left[\mathrm{L}_\odot\right]$',r'$\left[\mathrm{AU}\right]$',r'$\left[\mathrm{K}\right]$','']):
        
        self.get_pt(skip=skip,n_processes=n_processes)

        # Find the temperature at which the atmosphere becomes opaque
        if self.settings['clouds'] == 'opaque':
            self.ret_opaque_T = self.temperature_cloud_top
        else:
            self.ret_opaque_T = np.array([self.temperature[:,-1]]).T

        # Generating random data for the stellar luminosity and the panet separation
        L_star_data = L_star + sigma_L_star*np.random.randn(*self.ret_opaque_T.shape)
        sep_planet_data = sep_planet + sigma_sep_planet*np.random.randn(*self.ret_opaque_T.shape)

        # Defining constants needed for the calculations
        L_sun = 3.826*1e26
        AU = 1.495978707*1e11
        sigma_SBoltzmann = 5.670374419*1e-8

        # Converting stellar luminosity and planet separation to SI
        L_star_data_SI = L_star_data * L_sun
        sep_planet_data_SI = sep_planet_data * AU

        # Calculate the bond albedo
        self.A_Bond_ret = 1 - 16*np.pi*sep_planet_data_SI**2*sigma_SBoltzmann*self.ret_opaque_T**4/L_star_data_SI

        if plot:
            # Combine the different parameters into one array
            data = np.hstack([L_star_data,sep_planet_data,self.ret_opaque_T,self.A_Bond_ret])
            

            # Generate the corner plot
            fig, axs = rp_posteriors.Corner(data,titles,truths=[L_star,sep_planet,T_equ_true,A_Bond_true],units=units,bins=bins,
                                    quantiles1d=quantiles1d)

            # Save the figure or retrun the figure object
            if save:
                plt.savefig(self.results_directory+'Plots/plot_bond_albedo.pdf', bbox_inches='tight')
            return fig, axs
































    """
    #################################################################################
    #                                                                               #
    #   Plotting routine for the Bond Albedo                                        #
    #                                                                               #
    #################################################################################
    """





    def Plot_Ice_Surface_Test(self,MMW_atm=44,MMW_H2O=18,ax=None,skip=1,bins=50,save=False,n_processes=50):
        
        self.get_pt(skip=skip,n_processes=n_processes)

        if self.settings['clouds'] == 'opaque':
            ret_surface_T = self.temperature_cloud_top
            ret_surface_p = self.pressure_cloud_top
        else:
            ret_surface_T = self.temperature[:,-1]
            ret_surface_p = self.pressure[:,-1]


        # Calculating the Ice Vapour pressure at different temperatures
        T,p_part = np.loadtxt(self.code_directory+'/retrieval_plotting_support/vapour_pressure_data/ice.dat').T
        p_part = np.log10(p_part/1000)
        p_part_fit = np.zeros_like(ret_surface_T)

        for i in range(len(ret_surface_T)):
            ind = np.argsort(np.abs(T-ret_surface_T[i]))[:2]
            p_part_fit[i] = p_part[ind[0]]-(T[ind[0]]-ret_surface_T[i])*(p_part[ind[0]]-p_part[ind[1]])/(T[ind[0]]-T[ind[1]])

        ab_H20 = MMW_H2O/MMW_atm*10**p_part_fit/ret_surface_p


        ax_arg = ax
        if ax is None:
            figure = plt.figure(figsize = (10,5))
            ax = figure.add_axes([0.1, 0.1, 0.8, 0.8])
        else:
            pass

        keys = list(self.params.keys())
        ind = np.where(np.array(keys)==str('H2O'))[0][0]
        ret_H2O = self.equal_weighted_post[:,ind]


        ax.hist(np.log10(ab_H20),label='Inferred H2O Abundance',bins=bins)
        ax.hist(np.log10(ret_H2O),label='Retrieved H2O Abundance',bins=bins)

        # Either save the plot or return the figure object to allow for further formatting by the user
        handles, labels = ax.get_legend_handles_labels()
        if ax_arg is not None:
            return handles, labels
        elif save:
            ax.legend(loc='best')
            plt.savefig(self.results_directory+'Plots/Ice_Surface_Exclusion.png',dpi = 450)
        else:
            ax.legend(loc='best')
            return figure






    def Posterior_Classification(self,parameters=['H2O','CO2','CO','H2SO484(c)','R_pl','M_pl'],relative=None,limits = None,plot_classification=True,p0_SSG=[8,6,0.007,0.5,0.5],p0_SS=None,s_max=2,s_ssg_max=5):
        self.best_post_model = {}
        self.best_post_limit = {} 

        count_param = 0

        # Iterate over all parameters of interest
        for param in parameters:
            try:
                keys = list(self.params.keys())
                ind = np.where(np.array(keys)==str(param))[0][0]
                post = self.equal_weighted_post[:,ind]
                if relative is not None:
                    ind_rel = np.where(np.array(keys)==str(relative))[0][0]
                    post_rel = self.equal_weighted_post[:,ind_rel]

                if limits is None:
                    prior = self.priors[ind]
                                    
                    if prior in ['ULU', 'LU']:
                        if prior == 'LU':
                            if relative is None:
                                post = np.log10(post)
                            else:
                                post = np.log10(post) - np.log10(post_rel) 
                            self.best_post_limit[param] = sorted(list(self.priors_range[ind]))
                        else:
                            if relative is None:
                                post = np.log10(1-post)
                            else:
                                post = np.log10(post) - np.log10(post_rel) 
                            self.best_post_limit[param] = [-7,0]
                    elif prior in ['G','LG']:
                        mean = self.priors_range[ind][0]
                        sigma  = self.priors_range[ind][1]
                        self.best_post_limit[param] = [mean-5*sigma,mean+5*sigma]
                        if prior == 'LG':
                            post = np.log10(post)
                    else:
                        self.best_post_limit[param] = limits[count_param]

                span = abs(self.best_post_limit[param][1]-self.best_post_limit[param][0])
                center = span/2 + self.best_post_limit[param][0]
                p0_Gauss=[span/5,center,1.0]
                p0_SS=[-span/5,center,1.0]
                p0_u_SS=[span/5,center,1.0]


                x_bins = np.linspace(self.best_post_limit[param][0],self.best_post_limit[param][1],1000)
                binned_data = np.histogram(post,bins=100,range=self.best_post_limit[param],density=True)
                                
                model_likelihood = []
                                
                # Try to Fit each model to the retrieved data
                try:
                    params_F,cov_F = sco.curve_fit(r_post.Model_Flat,(binned_data[1][:-1]+binned_data[1][1:])/2,binned_data[0])
                    model_likelihood.append(r_post.log_likelihood(params_F,(binned_data[1][:-1]+binned_data[1][1:])/2,binned_data[0],r_post.Model_Flat))
                except:
                    model_likelihood.append(-np.inf)
                                    
                try:
                    params_SS,cov_SS = sco.curve_fit(r_post.Model_SoftStep,(binned_data[1][:-1]+binned_data[1][1:])/2,binned_data[0],p0=p0_SS)
                    model_likelihood.append(r_post.log_likelihood(params_SS,(binned_data[1][:-1]+binned_data[1][1:])/2,binned_data[0],r_post.Model_SoftStep))
                except:
                    model_likelihood.append(-np.inf)
                                
                try:
                    params_SSG,cov_SSG = sco.curve_fit(r_post.Model_SoftStepG,(binned_data[1][:-1]+binned_data[1][1:])/2,binned_data[0],p0=p0_SSG)
                    model_likelihood.append(r_post.log_likelihood(params_SSG,(binned_data[1][:-1]+binned_data[1][1:])/2,binned_data[0],r_post.Model_SoftStepG))
                    line_SSG = r_post.Model_SoftStepG(x_bins,*params_SSG)
                except:
                    model_likelihood.append(-np.inf)
                                
                try:
                    params_G,cov_G = sco.curve_fit(r_post.Model_Gauss,(binned_data[1][:-1]+binned_data[1][1:])/2,binned_data[0],p0=p0_Gauss)
                    model_likelihood.append(r_post.log_likelihood(params_G,(binned_data[1][:-1]+binned_data[1][1:])/2,binned_data[0],r_post.Model_Gauss))
                except:
                    model_likelihood.append(-np.inf)

                try:
                    params_u_SS,cov_u_SS = sco.curve_fit(r_post.Model_upper_SoftStep,(binned_data[1][:-1]+binned_data[1][1:])/2,binned_data[0],p0=p0_u_SS)
                    model_likelihood.append(r_post.log_likelihood(params_u_SS,(binned_data[1][:-1]+binned_data[1][1:])/2,binned_data[0],r_post.Model_upper_SoftStep))
                except:
                    model_likelihood.append(-np.inf)

                # Select the optimal model for the considered data case
                if model_likelihood[3]!=-np.inf:
                    if params_G[2]>span/6.0:
                        model_likelihood[3]=-np.inf
                if model_likelihood[1]!=-np.inf:
                    if r_post.Model_SoftStep(self.best_post_limit[param][1],*params_SS)>=params_SS[-1]/10:
                        model_likelihood[1]=-np.inf
                if model_likelihood[2]!=-np.inf:
                    if np.max(line_SSG)<=1.4*params_SSG[2] or np.max(line_SSG)>=15*params_SSG[2]:
                        model_likelihood[2]=-np.inf
                    if line_SSG[0]>=1.05*params_SSG[2]:
                        model_likelihood[2]=-np.inf
                    if line_SSG[-1]>=params_SSG[2]/20:
                        model_likelihood[2]=-np.inf
                    if params_SSG[-2]>=s_ssg_max:
                        model_likelihood[2]=-np.inf
                if model_likelihood[4]!=-np.inf:
                    if params_u_SS[0]<0:
                        model_likelihood[4]=-np.inf

                print(best_fit)
                
                # Storing the best fit model for the parameters of interest
                best_fit = np.argmax(model_likelihood)
                if best_fit == 0:
                    self.best_post_model[param] = ['F',params_F]
                elif best_fit == 1:
                    self.best_post_model[param] = ['SS',params_SS]
                elif best_fit == 2:
                    self.best_post_model[param] = ['SSG',params_SSG]
                elif best_fit == 3:
                    self.best_post_model[param] = ['G',params_G]
                elif best_fit == 4:
                    self.best_post_model[param] = ['USS',params_u_SS]
                else:
                    print(str(best_fit) + ' is not a valid model!')
        
                if plot_classification:
                    plt.figure(figsize=(10,10))
                    h = plt.hist(post,bins=100,range=self.best_post_limit[param],alpha=0.2,density=True)


                    if best_fit == 0:
                        plt.plot(x_bins,r_post.Model_Flat(x_bins,*params_F),'g-',lw=5)
                    if best_fit == 1:
                        plt.plot(x_bins,r_post.Model_SoftStep(x_bins,*params_SS),'r-',lw=5)
                    if best_fit == 2:
                        plt.plot(x_bins,r_post.Model_SoftStepG(x_bins,*params_SSG),'b-',lw=5)
                    if best_fit == 3:
                        plt.plot(x_bins,r_post.Model_Gauss(x_bins,*params_G),'m-',lw=5)               
                    if best_fit == 4:
                        plt.plot(x_bins,r_post.Model_upper_SoftStep(x_bins,*params_u_SS),'y-',lw=5)   
                    plt.plot([-15,0],[0,0],'k-',alpha=1)
                    plt.ylim([-max(h[0])/4,1.1*max(h[0])])
                    plt.yticks([])
                    plt.xticks([])
                    plt.xlim(self.best_post_limit[param])
                    plt.show()
            except:
                print(str(param) + ' was not a retrieved parameter in this retrieval run')

            count_param += 1
