__author__ = "Konrad"
__copyright__ = "Copyright 2022, Konrad"
__maintainer__ = "Björn S. Konrad"
__email__ = "konradb@phys.ethz.ch"
__status__ = "Development"

# Standard Libraries
#from ctypes.wintypes import RGB
import sys, os, re
import math as m
import numpy as np
import scipy.interpolate as scp
import scipy.optimize as sco
import matplotlib.pyplot as plt
import matplotlib.colors as col
from scipy.ndimage.filters import gaussian_filter1d

# Library for parallelization
import multiprocessing as mp
import time as t

# Import additional external files
from retrieval_support import retrieval_global_class as r_globals
from retrieval_support import retrieval_posteriors as r_post
from retrieval_plotting_support import retrieval_plotting_colors as rp_col
from retrieval_plotting_support import retrieval_plotting_handlerbase as rp_hndl
from retrieval_plotting_support import retrieval_plotting_inlay as rp_inlay

# Additional Libraries
import pymultinest as nest
import spectres as spectres





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

        return self.result

    def __worker(self,process,results_direectory,function,function_args):
        # Initialization of a new radtrans object
        results_temp =  retrieval_plotting(results_direectory)

        # Spectrum calculation
        function_args['process']=process
        getattr(results_temp,function)(**function_args)

        # Sve the results to the manager
        self.result[process] = results_temp.retrieved_fluxes



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
            self.true_wavelength, self.true_flux, self.true_error = np.loadtxt(self.results_directory + 'true_spectrum.txt').T
        except:
            pass
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



    def __get_retrieved(self,temp_vars_known,chem_vars_known,phys_vars_known,cloud_vars_known,temp_equal_weighted_post,ind):
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
                    if ind == 0:
                        if self.settings['clouds'] == 'opaque':
                            if retrieved_params[par].split('_',2)[2] == 'thickness':
                                min_cloud_center = temp_equal_weighted_post[2:,par-1]+temp_equal_weighted_post[2:,par]/2
                                hist = np.histogram(np.log10(min_cloud_center),bins=50)
                                self.cloud_opacity_cut = sco.curve_fit(self.Logistic_Function,(hist[1][:-1]+hist[1][1:])/2,hist[0],p0=[80,10,0])[0]
                                self.opacity = self.Inverse_Logistic_Function(self.cloud_opacity_cut[0]*np.array([0.16,0.5,0.84]),*self.cloud_opacity_cut)
                except:
                    self.cloud_vars['_'.join(retrieved_params[par].split('_',2)[:2])]['abundance'] = temp_equal_weighted_post[ind,par]
                    self.chem_vars[retrieved_params[par].split('_',1)[0]] = temp_equal_weighted_post[ind,par]



    def get_fluxes(self,skip=1000,n_processes=10):
        '''
        gets the fluxes corresponding to the parameter values 
        of the equal weighted posteriors.
        '''

        # If not yet done calculate the spectra corresponding
        # to the retrieved posterior distributions of partameters
        if not hasattr(self, 'retrieved_fluxes'):
                  
            # check if the spectra for the specified skip
            # values are already calculated
            try:
                ret_spec = np.loadtxt(self.results_directory+'Ret_Fluxes_Skip_'+str(skip)+'.txt')
                print('Loaded previously calculated retrieved spectra.')

                self.retrieved_fluxes=ret_spec
            except:
                print('Calculating retrieved spectra from scratch.')
                
                # Check that we do not use too many CPUs
                if (np.shape(self.equal_weighted_post)[0]//skip)//n_processes < 3:
                    print('Not enough jobs for the specified number of processes!')
                    while (np.shape(self.equal_weighted_post)[0]//skip)//n_processes < 3:
                        n_processes -= 1
                    print('I lowered n_processes to '+str(n_processes)+'.')

                # Start the paralel calculation
                parallel_calculation = parallel(n_processes)
                result = parallel_calculation.calculate(self.results_directory,'Calc_Spectra',{'skip':skip,'n_processes':n_processes})

                # Merge the calculated data
                retrieved_fluxes = list(result[0])
                for i in range(1,n_processes):
                    retrieved_fluxes += list(result[i])

                # Save the newly calculated data
                self.retrieved_fluxes = np.array(retrieved_fluxes)
                np.savetxt(self.results_directory+'Ret_Fluxes_Skip_'+str(skip)+'.txt',self.retrieved_fluxes)
            
            self.Calc_Spectra(skip=np.shape(self.equal_weighted_post)[0]-2)

            self.flux_mean = np.mean(self.retrieved_fluxes,axis=0)
            self.flux_median = np.median(self.retrieved_fluxes,axis=0)


    def get_pt(self,skip=1000):
        '''
        gets the fluxes corresponding to the parameter values 
        of the equal weighted posteriors.
        '''

        # If not yet done calculate the spectra corresponding
        # to the retrieved posterior distributions of partameters
        if not hasattr(self, 'pressures'):
                  
            # check if the spectra for the specified skip
            # values are already calculated
            try:
                ret_pt = np.loadtxt(self.results_directory+'Ret_PT_Skip_'+str(skip)+'.txt')
                print('Loaded previously calculated retrieved pt profiles.')

                self.Calc_PT_Profiles_uniform_bins(skip=np.shape(self.equal_weighted_post)[0]-2)
                self.retrieved_fluxes=ret_pt
            except:
                print('Calculating retrieved spectra from scratch.')
                
                self.Calc_PT_Profiles_uniform_bins(skip=skip)

                np.savetxt(self.results_directory+'Ret_PT_Skip_'+str(skip)+'.txt',self.temperature)


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

        """
        # for the vae_pt we currently cannot retrieve the surface pressure
        if self.settings['parametrization'] == 'vae_pt':
            if (('log_P0' in self.phys_vars.keys()) or ('P0' in self.phys_vars.keys()) or ('log_P0' in self.knowns.keys()) or ('P0' in self.knowns.keys())):
                print("ERROR! For the VAE-PT profile model the surface pressure cannot be retrieved. It is fixed at a value of")
                sys.exit()
            else:
                self.phys_vars['log_P0'] = np.log10(self.vae_pt.max_t) 
                self.phys_vars['log_Ptop'] = np.log10(self.vae_pt.min_t) 
        """
        #else:
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



    def Corner(self,data,titles,units=None,truths=None,dimension=None,precision = 2,quantiles = [0.16, 0.5, 0.84],bins=50):
        if dimension is None:
            dimension=np.shape(data)[1]
    
        fig, axs = plt.subplots(dimension, dimension,figsize=(dimension*2.5,dimension*2.5))
        fig.subplots_adjust(hspace=0.0)
        fig.subplots_adjust(wspace=0.0)
        fs = 18
        
        
        # Iterate over the equal weighted posteriors of all retrieved parameters.
        for i in range(dimension):
            q = []

            # Plot the 1d histogram for each retrieved parameter on the diagonal.
            h = axs[i,i].hist(data[:,i],histtype='step',color='black',density=True,bins=bins)

            # Define the limits of the plot and remove the yticks
            axs[i,i].set_ylim([0,1.1*np.max(h[0])])
            axs[i,i].set_yticks([])
            axs[i,i].set_xlim([h[1][0],h[1][-1]])

            # Plotting the secified quantiles
            if quantiles is not None:
                for ind in quantiles:
                    q.append(np.quantile(data[:,i],ind))
                    axs[i,i].plot([q[-1],q[-1]],axs[i,i].get_ylim(),'k--',linewidth = 1)
            
            # Plot the ground truth values if known
            if not truths[i] is None:
                axs[i,i].plot([truths[i],truths[i]],axs[i,i].get_ylim(),color='xkcd:red',linestyle = ':',linewidth = 2)
        
            # Print the defined quantiles for the retrieved value above the histogram plot
            axs[i,i].set_title(str(np.round(q[1],int(precision)))+r' $_{\,'+\
                            str(np.round(q[0]-q[1],int(precision)))+r'}^{\,+'+\
                            str(np.round(q[2]-q[1],int(precision)))+r'}$',fontsize=fs)

        for i in range(dimension):
            for j in range(dimension):
                # Find the axis boundaries and set the x limits
                ylim = axs[i,i].get_xlim()
                xlim = axs[j,j].get_xlim()

                # Setting 4 even ticks over the range defined by the limits of the subplot
                yticks = [(1-pos)*ylim[0]+pos*ylim[1] for pos in [0.2,0.4,0.6,0.8]]
                xticks = [(1-pos)*xlim[0]+pos*xlim[1] for pos in [0.2,0.4,0.6,0.8]]

                # Setting the limits of the x-Axis
                axs[i,j].set_xticks(xticks)

                # For all subplots below the Diagonal
                if i > j:
                    # Plot the local truth values if provided
                    
                    if not truths[j] is None:
                        axs[i,j].plot([truths[j],truths[j]],ylim,color='xkcd:red',linestyle = ':',linewidth = 2)
                    if not truths[i] is None:
                        axs[i,j].plot(xlim,[truths[i],truths[i]],color='xkcd:red',linestyle = ':',linewidth = 2)
                    if not ((truths[j] is None) or (truths[i] is None)):
                        axs[i,j].plot(truths[j],truths[i],color='xkcd:red',marker='o',markersize=8)
                    
                    # Plot the 2d histograms between different parameters to show correlations between the parameters
                    Z,X,Y=np.histogram2d(data[:,j],data[:,i],bins=15)
                    axs[i,j].contourf((X[:-1]+X[1:])/2,(Y[:-1]+Y[1:])/2,Z.T,cmap='Greys',levels=np.array([0.05,0.15,0.3,0.45,0.6,0.75,0.95,1])*np.max(Z))
                    
                    # Setting the limit of the y-axis
                    axs[i,j].set_yticks(yticks)

                    # Setting the boundaries for the axis
                    axs[i,j].set_ylim(ylim)
                    axs[i,j].set_xlim(xlim)

                    # Removing unnecessary ticks and labels                   
                    if j != 0:
                        axs[i,j].set_yticklabels([])
                        axs[i,j].tick_params(axis='y', length=0)
                        
                    if i != dimension-1:
                        axs[i,j].tick_params(axis='x', length=0)

                # No Subplots show above the diagonal
                elif i<j:
                    axs[i,j].axis('off')
        
                # Add the ticks and the axis labels where necessary on the y axis               
                if j == 0 and i!=0:
                    r = m.floor(np.log10(abs(yticks[0]-yticks[1])))-1
                    if r >= 0:
                        axs[i,j].set_yticklabels([int(y) for y in yticks],fontsize=fs,rotation=45)
                    else:
                        axs[i,j].set_yticklabels(np.round(yticks,-r),fontsize=fs,rotation=45)
                        
                    
                    if units is None:
                        axs[i,j].set_ylabel(titles[i],fontsize=fs)
                    else:
                        if units[i] == '':
                            axs[i,j].set_ylabel(titles[i],fontsize=fs)
                        else:
                            axs[i,j].set_ylabel(titles[i]+' '+units[i],fontsize=fs)

                # Add the ticks and the axis labels where necessary on the y axis on the x axis 
                if i == dimension-1:
                    r = m.floor(np.log10(abs(xticks[0]-xticks[1])))-1
                    if r>=0:
                        axs[i,j].set_xticklabels([int(x) for x in xticks],fontsize=fs,rotation=45)
                    else:
                        axs[i,j].set_xticklabels(np.round(xticks,-r),fontsize=fs,rotation=45,ha='right')
                    
                    
                    if units is None:
                        axs[i,j].set_xlabel(titles[j],fontsize=fs)
                    else:
                        if units[j] == '':
                            axs[i,j].set_xlabel(titles[j],fontsize=fs)
                        else:
                            axs[i,j].set_xlabel(titles[j]+' '+units[j],fontsize=fs)

        # Set all titles at a uniform distance from the subplots
        fig.align_ylabels(axs[:, 0])
        fig.align_xlabels(axs[-1,:])
        return fig










    def Corner_Plot(self, save=False, log_pressures=True, log_mass=True, log_abundances=True, log_particle_radii=True, plot_pt = True, plot_physparam = True, plot_clouds = True,plot_chemcomp=True,plot_bond=None, titles = None, units=None, bins=20,
                    quantiles=[0.16, 0.5, 0.84], precision=2):
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
                        else:
                            local_titles[i] = local_titles[i].split('_',1)[-1]
                        local_titles[i] = 'log$_{10}$ ' + local_titles[i]
        
        # exlude all unwanted p[arameters from the corner plot
        inds =  [i for i in range(len(param_names))]
        if not plot_pt:
            inds += [i for i in range(len(param_names)) if i not in inds_pt]
        if not plot_clouds:
            inds += [i for i in range(len(param_names)) if i not in inds_clouds]
        if not plot_chemcomp:
            inds += [i for i in range(len(param_names)) if i not in inds_chemcomp]
        if not plot_physparam:
            inds += [i for i in range(len(param_names)) if i not in inds_physparam]
        def none_test(input,inds):
            try:
                return [input[i] for i in inds]
            except:
                return None

        if not titles is None:
            local_titles=titles

        if plot_bond is not None:
            if not hasattr(self, 'A_Bond_ret'):
                results.Plot_Ret_Bond_Albedo(*plot_bond[:-2],A_Bond_true = plot_bond[-1], T_equ_true=plot_bond[-2],save = True,bins=20)
            local_equal_weighted_post = np.append(local_equal_weighted_post, np.array([self.ret_surface_T]).T,axis=1)
            local_equal_weighted_post = np.append(local_equal_weighted_post, np.array([self.A_Bond_ret]).T,axis=1)
            local_truths += plot_bond[-2:]
            inds += [-2,-1]
            
        fig = self.Corner(local_equal_weighted_post[:,inds],none_test(local_titles,inds),dimension = len(inds),truths=none_test(local_truths,inds),precision=precision,quantiles=quantiles,units=none_test(units,inds),bins=bins)
        # Save the figure or retrun the figure object
        if save:
            plt.savefig(self.results_directory+'Plots/plot_corner.pdf', bbox_inches='tight')
            pass
        else:
            return fig
        







    def Calc_PT_Profiles(self,skip = 10):
        '''
        Function to calculate the PT profiles corresponding to the retrieved posterior distributions
        for subsequent plotting in the flux PT plotting functions.
        '''

        # Import petitRADTRANS
        sys.path.append(self.path_prt)
        from petitRADTRANS import Radtrans
        from petitRADTRANS import nat_cst as nc

        # Add the thest fit spectrum and the input parameters to the equal weighted posterior
        temp_equal_weighted_post = np.append(np.array([self.best_fit]),self.equal_weighted_post[:,:-1],axis=0)
        temp_equal_weighted_post = np.append(np.array([self.truths]),temp_equal_weighted_post,axis=0)

        # Fetch the known parameters
        temp_vars_known, phys_vars_known, chem_vars_known, cloud_vars_known = self.__get_knowns()
        
        # Print status of calculation
        print('Starting PT-profile calculation.')
        print('\t0.00 % of PT-profiles calculated.', end = "\r")

        # Iterate over the equal weighted posterior distribution using the user-defined skip value
        dimension = np.shape(temp_equal_weighted_post)[0]//skip   




        for i in range(dimension):
            ind = min(2,i)+skip*max(0,i-2)

            # Fetch the retrieved parameters for a given ind
            self.__get_retrieved(self,temp_vars_known,chem_vars_known,phys_vars_known,cloud_vars_known,temp_equal_weighted_post,ind)



            # Test the values of P0 and g and change to required values if necessary
            self.__g_test()            
            self.__P0_test()

            # Calculate the cloud bottom pressure from the cloud thickness parameter
            cloud_tops = []
            cloud_bottoms = []
            for key in self.cloud_vars.keys():
                cloud_bottoms += [self.cloud_vars[key]['top_pressure']+self.cloud_vars[key]['thickness']]
                cloud_tops += [self.cloud_vars[key]['top_pressure']]
                self.make_press_temp_terr(log_top_pressure=np.log10(np.min(cloud_tops)),layers=1000)
                pressure_cloud_top = self.press
                temperature_cloud_top = self.temp

            # Calculate the pressure temperature profile corresponding to the set of parameters
            if ((self.settings['clouds'] == 'opaque') and (i>0)):
                self.make_press_temp_terr(log_top_pressure=self.cloud_opacity_cut[-1],layers=1000)
                pressure_extrapol = self.press
                temperature_extrapol = self.temp
                self.phys_vars['log_P0'] = self.cloud_opacity_cut[-1]
                self.make_press_temp_terr(layers=1000)
            else:
                self.make_press_temp_terr(layers=1000)

            # store the calculated values
            if i == 0:
                if not hasattr(self, 'input_temperature'):
                    self.input_pressure = self.press
                    self.input_temperature = self.temp
                if len(self.cloud_vars) != 0:
                    self.true_cloud_top = cloud_tops
                    self.true_cloud_bottom = cloud_bottoms
                    self.true_pressure_cloud_top = pressure_cloud_top
                    self.true_temperature_cloud_top = temperature_cloud_top
            elif i == 1:
                self.best_pressure = self.press
                self.best_temperature = self.temp
                if len(self.cloud_vars) != 0:
                    self.best_cloud_top = cloud_tops
                    self.best_cloud_bottom = cloud_bottoms
                    self.best_pressure_cloud_top = pressure_cloud_top
                    self.best_temperature_cloud_top = temperature_cloud_top
                if self.settings['clouds'] == 'opaque':
                    self.best_pressure_extrapol = pressure_extrapol
                    self.best_temperature_extrapol = temperature_extrapol
            elif i == 2:
                self.pressure = np.array([self.press])
                self.temperature = np.array([self.temp])
                if len(self.cloud_vars) != 0:
                    self.cloud_top = np.array([cloud_tops])
                    self.cloud_bottom = np.array([cloud_bottoms])
                    self.pressure_cloud_top = np.array([pressure_cloud_top])
                    self.temperature_cloud_top = np.array([temperature_cloud_top])
                if self.settings['clouds'] == 'opaque':
                    self.pressure_extrapol = np.array([pressure_extrapol])
                    self.temperature_extrapol = np.array([temperature_extrapol])
            else:
                self.pressure = np.append(self.pressure,np.array([self.press]),axis=0)
                self.temperature = np.append(self.temperature,np.array([self.temp]),axis=0)
                if len(self.cloud_vars) != 0:
                    self.cloud_top = np.append(self.cloud_top,np.array([cloud_tops]),axis=0)
                    self.cloud_bottom = np.append(self.cloud_bottom,np.array([cloud_bottoms]),axis=0)
                    self.pressure_cloud_top = np.append(self.pressure_cloud_top,np.array([pressure_cloud_top]),axis=0)
                    self.temperature_cloud_top = np.append(self.temperature_cloud_top,np.array([temperature_cloud_top]),axis=0)
                if self.settings['clouds'] == 'opaque':
                    self.pressure_extrapol = np.append(self.pressure_extrapol,np.array([pressure_extrapol]),axis=0)
                    self.temperature_extrapol = np.append(self.temperature_extrapol,np.array([temperature_extrapol]),axis=0)
            # Print status of calculation
            print('\t'+str(np.round((i+1)/dimension*100,2))+' % of PT-profiles calculated.', end = "\r")

        # Print status of calculation
        print('\nPT-profile calculation completed.')





    def Calc_PT_Profiles_uniform_bins(self,skip = 10,layers=5000,p_surf=4):
        '''
        Function to calculate the PT profiles corresponding to the retrieved posterior distributions
        for subsequent plotting in the flux PT plotting functions.
        '''

        # Import petitRADTRANS
        sys.path.append(self.path_prt)
        from petitRADTRANS import Radtrans
        from petitRADTRANS import nat_cst as nc

        # Add the thest fit spectrum and the input parameters to the equal weighted posterior
        temp_equal_weighted_post = np.append(np.array([self.best_fit]),self.equal_weighted_post[:,:-1],axis=0)
        temp_equal_weighted_post = np.append(np.array([self.truths]),temp_equal_weighted_post,axis=0)

        # Fetch the known parameters
        temp_vars_known, phys_vars_known, chem_vars_known, cloud_vars_known = self.__get_knowns()
        
        # Print status of calculation
        print('Starting PT-profile calculation.')
        print('\t0.00 % of PT-profiles calculated.', end = "\r")

        # Iterate over the equal weighted posterior distribution using the user-defined skip value
        dimension = np.shape(temp_equal_weighted_post)[0]//skip
        for i in range(dimension):
            ind = min(2,i)+skip*max(0,i-2)

            # Fetch the retrieved parameters for a given ind
            self.__get_retrieved(temp_vars_known,chem_vars_known,phys_vars_known,cloud_vars_known,temp_equal_weighted_post,ind)
        
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
                pressure_cloud_top = self.press
                temperature_cloud_top = self.temp

            self.make_press_temp_terr(log_ground_pressure=p_surf,layers=layers)
            pressure_full = self.press
            temperature_full = self.temp
            ind = np.where(self.press > 10**self.phys_vars['log_P0'])
            pressure_full[ind] = np.nan
            temperature_full[ind] = np.nan

            if ((self.settings['clouds'] == 'opaque') and (i>0)):
                self.make_press_temp_terr(log_ground_pressure=p_surf,layers=layers)
                pressure_full_ct = self.press
                temperature_full_ct = self.temp
                ind = np.where(self.press > np.min(cloud_tops))
                pressure_full_ct[ind] = np.nan
                temperature_full_ct[ind] = np.nan

            # Calculate the pressure temperature profile corresponding to the set of parameters
            if ((self.settings['clouds'] == 'opaque') and (i>0)):
                self.make_press_temp_terr(log_top_pressure=self.cloud_opacity_cut[-1],layers=layers)
                pressure_extrapol = self.press
                temperature_extrapol = self.temp
                self.phys_vars['log_P0'] = self.cloud_opacity_cut[-1]
                self.make_press_temp_terr(layers=layers)
            else:
                self.make_press_temp_terr(layers=layers)

            # store the calculated values
            if i == 0:
                if not hasattr(self, 'input_temperature'):
                    self.input_pressure = self.press
                    self.input_temperature = self.temp
                if len(self.cloud_vars) != 0:
                    self.true_cloud_top = cloud_tops
                    self.true_cloud_bottom = cloud_bottoms
                    self.true_pressure_cloud_top = pressure_cloud_top
                    self.true_temperature_cloud_top = temperature_cloud_top
            elif i == 1:
                self.best_pressure = self.press
                self.best_temperature = self.temp
                if len(self.cloud_vars) != 0:
                    self.best_cloud_top = cloud_tops
                    self.best_cloud_bottom = cloud_bottoms
                    self.best_pressure_cloud_top = pressure_cloud_top
                    self.best_temperature_cloud_top = temperature_cloud_top
                if self.settings['clouds'] == 'opaque':
                    self.best_pressure_extrapol = pressure_extrapol
                    self.best_temperature_extrapol = temperature_extrapol
            else:
                if i==2:
                    # initialize the arrays for storage
                    self.pressure = np.zeros((dimension-2,len(self.press)))
                    self.temperature = np.zeros((dimension-2,len(self.temp)))
                    self.pressure_full = np.zeros((dimension-2,len(pressure_full)))
                    self.temperature_full = np.zeros((dimension-2,len(temperature_full)))
                    if len(self.cloud_vars) != 0:
                        self.cloud_top = np.zeros((dimension-2,len(cloud_tops)))
                        self.cloud_bottom = np.zeros((dimension-2,len(cloud_bottoms)))
                        self.pressure_cloud_top = np.zeros((dimension-2,len(pressure_cloud_top)))
                        self.temperature_cloud_top = np.zeros((dimension-2,len(temperature_cloud_top)))
                    if self.settings['clouds'] == 'opaque':
                        self.pressure_extrapol = np.zeros((dimension-2,len(pressure_extrapol)))
                        self.temperature_extrapol = np.zeros((dimension-2,len(temperature_extrapol)))
                        self.pressure_full_ct = np.zeros((dimension-2,len(pressure_full_ct)))
                        self.temperature_full_ct = np.zeros((dimension-2,len(temperature_full_ct)))

                self.pressure[i-2,:] = self.press
                self.temperature[i-2,:] = self.temp
                self.pressure_full[i-2,:] = pressure_full
                self.temperature_full[i-2,:] = temperature_full
                if len(self.cloud_vars) != 0:
                    self.cloud_top[i-2,:] = cloud_tops
                    self.cloud_bottom[i-2,:] = cloud_bottoms
                    self.pressure_cloud_top[i-2,:] = pressure_cloud_top
                    self.temperature_cloud_top[i-2,:] = temperature_cloud_top
                if self.settings['clouds'] == 'opaque':
                    self.pressure_extrapol[i-2,:] = pressure_extrapol
                    self.temperature_extrapol[i-2,:] = temperature_extrapol
                    self.pressure_full_ct[i-2,:] = pressure_full_ct
                    self.temperature_full_ct[i-2,:] = temperature_full_ct

            # Print status of calculation
            print('\t'+str(np.round((i+1)/dimension*100,2))+' % of PT-profiles calculated.', end = "\r")

        # Print status of calculation
        print('\nPT-profile calculation completed.')






    def Logistic_Function(self,x,L,k,hm):
        return L/(1+np.exp(-k*(x-hm)))

    def Inverse_Logistic_Function(self,y,L,k,hm):
        return hm - np.log(L/y-1)/k








    def PT_Envelope(self,ax = None, save=False, plot_clouds = False, color='C2', loc_surface='top right', x_lim =[0,1000], y_lim = [1e-6,1e4],
                    quantiles=[0.01,0.10,0.25,0.75,0.90,0.99],skip=1):
    
        if not hasattr(self, 'pressures'):
            self.Calc_PT_Profiles_uniform_bins(skip = skip)

        # find the quantiles for the different pressures and temperatures
        p_layers_quantiles = [np.nanquantile(self.pressure_full,q,axis=0) for q in quantiles]
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



        # find the quantiles for the pressures above the clouds
        if self.settings['clouds'] == 'opaque':
            p_layers_quantiles_ct = [np.nanquantile(self.pressure_full_ct,q,axis=0) for q in quantiles]
            T_layers_quantiles_ct = [np.nanquantile(self.temperature_full_ct,q,axis=0) for q in quantiles]
            not_nan_ct = np.count_nonzero(~np.isnan(self.pressure_full_ct),axis = 0)/np.shape(self.pressure_full_ct)[0]
            for q in range(len(quantiles)):
                T_layers_quantiles_ct[q][np.where(not_nan_ct<min(2*(1-quantiles[q]),2*(quantiles[q])))] = np.nan
            for q in range(len(quantiles)):
                notnan_ct = ~np.isnan(T_layers_quantiles_ct[q])
                T_layers_quantiles_ct[q] = T_layers_quantiles_ct[q][notnan_ct]
                p_layers_quantiles_ct[q] = p_layers_quantiles_ct[q][notnan_ct]
                
                X_Y_Spline = scp.make_interp_spline(p_layers_quantiles_ct[q],T_layers_quantiles_ct[q])
                p_layers_quantiles_ct[q] = np.logspace(np.log10(p_layers_quantiles_ct[q].min()),np.log10(p_layers_quantiles_ct[q].max()),80)
                T_layers_quantiles_ct[q] = X_Y_Spline(p_layers_quantiles_ct[q])

        #find the median and the mean pressure values
        median_temperature, median_pressure = np.median(self.temperature,axis=0), np.median(self.pressure,axis=0)
        mean_temperature, mean_pressure = np.mean(self.temperature,axis=0),np.mean(self.pressure,axis=0)
        if self.settings['clouds'] == 'opaque':
            median_temperature_extrapol, median_pressure_extrapol = np.median(self.temperature_extrapol,axis=0), np.median(self.pressure_extrapol,axis=0)
            mean_temperature_extrapol, mean_pressure_extrapol = np.mean(self.temperature_extrapol,axis=0),np.mean(self.pressure_extrapol,axis=0)
            median_temperature_cloud_top, median_pressure_cloud_top = np.median(self.temperature_cloud_top,axis=0), np.median(self.pressure_cloud_top,axis=0)
            mean_temperature_cloud_top, mean_pressure_cloud_top = np.mean(self.temperature_cloud_top,axis=0),np.mean(self.pressure_cloud_top,axis=0)

        # If wanted find the quantiles for cloud top and bottom pressures
        if plot_clouds == True:
            cloud_top_quantiles = [np.quantile(self.cloud_top,q,axis=0) for q in quantiles]
            cloud_bottom_quantiles = [np.quantile(self.cloud_bottom,q,axis=0) for q in quantiles]

        # Start of the plotting
        ax_arg = ax
        if ax is None:
            figure = plt.figure() #figsize = (10,5))
            ax = figure.add_axes([0.1, 0.1, 0.8, 0.8])
        else:
            pass
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        #If wanted plot the clouds
        if plot_clouds:
            if self.settings['clouds'] == 'opaque':
                for i in range(len(quantiles)-1):
                    ax.fill([0,10000,10000,0],[cloud_top_quantiles[i],cloud_top_quantiles[i],cloud_top_quantiles[i+1],cloud_top_quantiles[i+1]],
                        color='black',alpha=0.1+i*0.07, lw = 0)
                ax.fill([0,10000,10000,0],[cloud_top_quantiles[-1],cloud_top_quantiles[-1],1e4,1e4],
                    color='black',alpha=0.1+(len(quantiles)-1)*0.07, lw = 0)
            else:
                for i in range(len(quantiles)-1):
                    ax.fill([0,10000,10000,0],[cloud_top_quantiles[i],cloud_top_quantiles[i],cloud_top_quantiles[i+1],cloud_top_quantiles[i+1]],
                        color='black',alpha=0.1+i*0.07, lw = 0)
                    ax.fill([0,10000,10000,0],[cloud_bottom_quantiles[-i-1],cloud_bottom_quantiles[-i-1],cloud_bottom_quantiles[-i-2],cloud_bottom_quantiles[-i-2]],
                        color='black',alpha=0.1+i*0.07, lw = 0)
                ax.fill([0,10000,10000,0],[cloud_top_quantiles[-1],cloud_top_quantiles[-1],cloud_bottom_quantiles[0],cloud_bottom_quantiles[0]],
                    color='black',alpha=0.1+(len(quantiles)-1)*0.07, lw = 0)

        #plotting the uncertainty on the retrieved spectrum
        for i in range(int(len(quantiles)/2)):
            if self.settings['clouds'] == 'opaque':
                alpha = 0.1
            else:
                alpha = 0.3
            if i == 0:
                ax.fill(np.append(T_layers_quantiles[i],np.flip(T_layers_quantiles[-i-1])),
                        np.append(p_layers_quantiles[i],np.flip(p_layers_quantiles[-i-1])),color = 'white',lw = 0)
            ax.fill(np.append(T_layers_quantiles[i],np.flip(T_layers_quantiles[-i-1])),
                    np.append(p_layers_quantiles[i],np.flip(p_layers_quantiles[-i-1])),color = color,alpha=alpha,lw = 0)
        
        title = [r'$1\%-99\%$',r'$10\%-90\%$',r'$25\%-75\%$']
        if self.settings['clouds'] == 'opaque':
            for i in range(int(len(quantiles)/2)):
                alpha = 0.3+i*0.3
                
                ax.fill(np.append(T_layers_quantiles_ct[i],np.flip(T_layers_quantiles_ct[-i-1])),
                        np.append(p_layers_quantiles_ct[i],np.flip(p_layers_quantiles_ct[-i-1])),color = 'white',lw = 0)
                ax.fill(np.append(T_layers_quantiles_ct[i],np.flip(T_layers_quantiles_ct[-i-1])),
                        np.append(p_layers_quantiles_ct[i],np.flip(p_layers_quantiles_ct[-i-1])),color = color,alpha=alpha,lw = 0,label = title[i])

        #if self.settings['clouds'] == 'opaque':
        #    T = (T_layers_quantiles_extrapol[i//2][0]+T_layers_quantiles_extrapol[i//2+1][0])/2
        #    ax.plot([0.5*T,1.5*T],[10**self.opacity[1],10**self.opacity[1]],'r-')
        #    ax.fill([0.6*T,1.4*T,1.4*T,0.6*T],[10**self.opacity[2],10**self.opacity[2],10**self.opacity[0],10**self.opacity[0]],color='red',alpha=0.3,lw = 0)


        # Plot best, mean and median fit profiles
        #ax.semilogy(self.best_temperature,self.best_pressure,color ='green', label = 'Best Fit PT', zorder=4)
        #ax.semilogy(median_temperature, median_pressure,color ='blue', linestyle='-.', label = 'Median fit', zorder=5)
        #ax.semilogy(mean_temperature, mean_pressure,color ='darkorchid', linestyle=':',label = 'Mean fit', zorder=5)

        
        #if self.settings['clouds'] == 'opaque':
        #    #ax.semilogy(self.best_temperature_extrapol,self.best_pressure_extrapol,color ='green',alpha=0.4, zorder=4)
        #    ax.plot(median_temperature_extrapol, median_pressure_extrapol,color ='blue',alpha=0.4, linestyle='-.', zorder=5)
        #    ax.plot(mean_temperature_extrapol, mean_pressure_extrapol,color ='darkorchid',alpha=0.4, linestyle=':', zorder=5)

        # True input profile
        ax.semilogy(self.input_temperature,self.input_pressure,color ='black', label = 'Input Profile', zorder=3)

        # True surface temperature and surface pressure
        y_bound = (np.log10(y_lim[1])-np.log10(y_lim[0]))/20
        ax.plot([self.input_temperature[-1]-(x_lim[1]-x_lim[0])/40,self.input_temperature[-1]+(x_lim[1]-x_lim[0])/40],[self.input_pressure[-1],self.input_pressure[-1]],color='C3',linestyle = '-',linewidth = 2, zorder=6)
        ax.plot([self.input_temperature[-1],self.input_temperature[-1]],[10**(np.log10(self.input_pressure[-1])+y_bound),10**(np.log10(self.input_pressure[-1])-y_bound)],color='C3',linestyle = '-',linewidth = 2, zorder=6)
        ax.plot(self.input_temperature[-1],self.input_pressure[-1],marker='o',color='C3',ms=5, zorder=6)
        #ax1.set_xlim(0.8*min(self.input_temperature),max(self.input_temperature)*1.2)

        # True cloud top pressure and temperature
        if self.settings['clouds'] == 'opaque':
            ax.plot([self.true_temperature_cloud_top[0]-(x_lim[1]-x_lim[0])/40,self.true_temperature_cloud_top[0]+(x_lim[1]-x_lim[0])/40],[self.true_pressure_cloud_top[0],self.true_pressure_cloud_top[0]],color='C4',linestyle = '-',linewidth = 2, zorder=6)
            ax.plot([self.true_temperature_cloud_top[0],self.true_temperature_cloud_top[0]],[10**(np.log10(self.true_pressure_cloud_top[0])+y_bound),10**(np.log10(self.true_pressure_cloud_top[0])-y_bound)],color='C4',linestyle = '-',linewidth = 2, zorder=6)
            ax.plot(self.true_temperature_cloud_top[0],self.true_pressure_cloud_top[0],marker='s',color='C4',ms=5, zorder=6)

        if ax_arg is None:
            ax.set_xlabel('Temperature [K]')
            ax.set_ylabel('Pressure [bar]')
        ax.invert_yaxis()

        # Position the 2d histogram according to the input
        if ax_arg is None:
            if loc_surface == 'bottom left':
                ax2 = figure.add_axes([0.11, 0.12, 0.2, 0.4])
            elif loc_surface == 'bottom right':
                ax2 = figure.add_axes([1-0.11-0.2, 0.12, 0.2, 0.4])
            elif loc_surface == 'top left':
                ax2 = figure.add_axes([0.11, 1-0.12-0.4, 0.2, 0.4])
            elif loc_surface == 'top right':
                ax2 = figure.add_axes([1-0.12833-0.2, 1-0.12-0.2833, 0.21, 0.2833])
                #ax2 = figure.add_axes([1-0.11-0.2, 1-0.12-0.4, 0.2, 0.4])
        else:
            if loc_surface == 'bottom left':
                ax2 = ax.inset_axes([0.02*2/3, 0.02, 0.5*2/3, 0.5])
            elif loc_surface == 'bottom right':
                ax2 = ax.inset_axes([1-0.5*2/3, 0.02, 1-0.02*2/3, 0.5])
            elif loc_surface == 'top left':
                ax2 = ax.inset_axes([0.02*2/3, 1-0.5, 0.5*2/3, 1-0.02])
            elif loc_surface == 'top right':
                ax2 = ax.inset_axes([1-0.5*2/3-0.02*2/3, 1-0.5-0.02,0.5*2/3, 0.5])

        if self.settings['clouds'] == 'opaque':
            # make a 2d histogram of the surface pressures and temperatures
            Z,X,Y=np.histogram2d(self.temperature_cloud_top[:,0],np.log10(self.pressure_cloud_top[:,0]),bins=15)
            ax2.contourf((X[:-1]+X[1:])/2,10**((Y[:-1]+Y[1:])/2),Z.T,cmap='Greys',levels=np.array([0.05,0.15,0.3,0.45,0.6,0.75,0.95,1])*np.max(Z))

            # set the limits of the subplot axes
            #xlim = [X[0],X[-1]]
            #ax2.set_xlim(xlim)
            #ylim = [10**Y[-1],10**Y[0]]
            #ax2.set_ylim(ylim)

            # plot the true values that were used to generate the input spectrum
            ax2.plot(ax2.get_xlim(),[self.true_pressure_cloud_top[0],self.true_pressure_cloud_top[0]],color='C4',linestyle = '-',linewidth = 2)
            ax2.plot([self.true_temperature_cloud_top[0],self.true_temperature_cloud_top[0]],ax2.get_ylim(),color='C4',linestyle = '-',linewidth = 2)
            ax2.plot(self.true_temperature_cloud_top[0],self.true_pressure_cloud_top[0],marker='s',color='C4',ms=5)

            # plot the best fit median and mean surface pressures and temperatures
            ax2.scatter(self.best_temperature_cloud_top[0],self.best_pressure_cloud_top[0], marker='x',color='green')
            ax2.scatter(mean_temperature_cloud_top[0],mean_pressure_cloud_top[0], marker='x',color='darkorchid')
            ax2.scatter(median_temperature_cloud_top[0],median_pressure_cloud_top[0], marker='x',color='blue')

            # Define the Plot titles
            xlabel = r'$\mathrm{T_{Cloud\,Top}}\,\,\left[\mathrm{K}\right]$'
            ylabel = r'$\mathrm{P_{Cloud\,Top}}\,\,\left[\mathrm{bar}\right]$'
        else:
            # make a 2d histogram of the surface pressures and temperatures
            Z,X,Y=np.histogram2d(self.temperature[:,-1],self.pressure[:,-1],bins=15)
            ax2.contourf((X[:-1]+X[1:])/2,(Y[:-1]+Y[1:])/2,Z.T,cmap='Greys',levels=np.array([0.05,0.15,0.3,0.45,0.6,0.75,0.95,1])*np.max(Z))

            # set the limits of the subplot axes
            #xlim = [X[0],X[-1]]
            #ax2.set_xlim(xlim)
            #ylim = [Y[-1],Y[0]]
            #ax2.set_ylim(ylim)

            # plot the true values that were used to generate the input spectrum
            ax2.plot(ax2.get_xlim(),[self.input_pressure[-1],self.input_pressure[-1]],color='C3',linestyle = '-',linewidth = 2)
            ax2.plot([self.input_temperature[-1],self.input_temperature[-1]],ax2.get_ylim(),color='C3',linestyle = '-',linewidth = 2)
            ax2.plot(self.input_temperature[-1],self.input_pressure[-1],marker='o',color='C3',ms=5)

            # plot the best fit median and mean surface pressures and temperatures
            #ax2.scatter(self.best_temperature[-1],self.best_pressure[-1], marker='x',color='green')
            ax2.scatter(mean_temperature[-1],mean_pressure[-1], marker='x',color='darkorchid')
            ax2.scatter(median_temperature[-1],median_pressure[-1], marker='x',color='blue')

            # Define the Plot titles
            xlabel = r'$\mathrm{T_0}\,\,\left[\mathrm{K}\right]$'
            ylabel = r'$\mathrm{P_0}\,\,\left[\mathrm{bar}\right]$'
        xlim = ax2.get_xlim()
        ylim = ax2.get_ylim()
        xticks = [(1-pos)*xlim[0]+pos*xlim[1] for pos in [0.2,0.4,0.6,0.8]]
        yticks = [(1-pos)*ylim[0]+pos*ylim[1] for pos in [0.2,0.4,0.6,0.8]]
        
        if loc_surface == 'bottom left':
            ax2.set_ylabel(ylabel,va='top',rotation = 90)
            ax2.set_xlabel(xlabel,va='bottom')
            ax2.yaxis.set_label_position("right")
            ax2.yaxis.tick_right()
            ax2.xaxis.set_label_position("top")
            ax2.xaxis.tick_top()
        elif loc_surface == 'bottom right':
            ax2.set_ylabel(ylabel,rotation = 90)
            ax2.set_xlabel(xlabel,va='bottom')
            ax2.xaxis.set_label_position("top")
            ax2.xaxis.tick_top()
        elif loc_surface == 'top left':
            ax2.set_ylabel(ylabel,va='top',rotation = 90)
            ax2.set_xlabel(xlabel)
            ax2.yaxis.set_label_position("right")
            ax2.yaxis.tick_right()
        elif loc_surface == 'top right':
            ax2.set_ylabel(ylabel,rotation = 90)
            ax2.set_xlabel(xlabel)

        #roundx = np.abs(np.min([0,np.floor(np.log10(np.abs(xticks[1]-xticks[0])))-1]))
        #ax2.set_xticks(xticks)
        #ax2.set_xticklabels(np.round(xticks,int(roundx)),rotation=90)

        #roundy = np.abs(np.min([0,np.floor(np.log10(np.abs(yticks[1]-yticks[0])))-1]))
        #ax2.set_yticks(yticks)
        #ax2.set_yticklabels(np.round(yticks,int(roundy)))

        handles, labels = ax.get_legend_handles_labels()
        if ax_arg is not None:
            pass
        elif self.settings['clouds'] == 'opaque':
            lgd = ax.legend(handles+['CloudTopPressure','SurfacePressure'], labels+['True Cloud Top','True Surface'],handler_map={str:  rp_hndl.Handles()}, ncol=4,loc=[0.13,-0.25])
        else:
            lgd = ax.legend(handles+['SurfacePressure'], labels+['True Surface'],handler_map={str: rp_hndl.Handles()}, ncol=4,loc=[0.17,-0.25])

        if ax_arg is not None:
            return handles+['CloudTopPressure','SurfacePressure'], labels+['True Cloud Top','True Surface']
        elif save:
            plt.savefig(self.results_directory+'Plots/plot_pt_structure.pdf', bbox_inches='tight',bbox_extra_artists=(lgd,))
        else:
            return figure




    def PT_Envelope_Residual(self,ax = None, save=False, plot_clouds = False, color='C2', loc_surface='top right', x_lim =[0,1000], y_lim = [1e-6,1e4],
                    quantiles=[0.05,0.15,0.25,0.35,0.65,0.75,0.85,0.95],quantiles_title = None,skip=1,case_identifier = ''):

        #self.get_pt(skip=skip)
        if not hasattr(self, 'pressures'):
            self.Calc_PT_Profiles_uniform_bins(skip = skip)

        # find the quantiles for the different pressures and temperatures
        p_layers_quantiles = [np.nanquantile(self.pressure_full,q,axis=0) for q in quantiles]
        T_layers_quantiles = [np.nanquantile(self.temperature_full,q,axis=0)-np.nanquantile(self.temperature_full,0.5,axis=0) for q in quantiles]
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
            median_temperature_cloud_top, median_pressure_cloud_top = np.median(self.temperature_cloud_top,axis=0), np.median(self.pressure_cloud_top,axis=0)
            cloud_top_quantiles = [np.quantile(self.cloud_top,q,axis=0) for q in quantiles]





        # Generate colorlevels for the different quantiles
        color_levels, level_thresholds, N_levels = rp_col.color_levels(color,quantiles)





        # Start of the plotting
        alpha_step=2/len(quantiles)
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

        ax.semilogy([0,0], [0,1000],color ='black', linestyle=':', zorder=5)
        ax.annotate('Retrieved PT Median',[0-0.025*x_lim[1],10**(0.975*(np.log10(y_lim[1])-np.log10(y_lim[0]))+np.log10(y_lim[0]))],color = 'black',rotation=90,ha='right')

        # If wanted: plotting the retrieved cloud top
        if plot_clouds:
            for i in range(int(len(quantiles)/2)):
                alpha = 0.1+i*0.15
                if i == len(quantiles)-i-2:
                    ax.fill([-10000,10000,10000,-10000],[cloud_top_quantiles[i],cloud_top_quantiles[i],cloud_top_quantiles[i+1],cloud_top_quantiles[i+1]],color = 'black',alpha=alpha,lw = 0, zorder=5)
                else:
                    ax.fill([-10000,10000,10000,-10000],[cloud_top_quantiles[i],cloud_top_quantiles[i],cloud_top_quantiles[i+1],cloud_top_quantiles[i+1]],color = 'black',alpha=alpha,lw = 0, zorder=5)
                    ax.fill([-10000,10000,10000,-10000],[cloud_top_quantiles[-i-2],cloud_top_quantiles[-i-2],cloud_top_quantiles[-i-1],cloud_top_quantiles[-i-1]],color = 'black',alpha=alpha,lw = 0, zorder=5)
            ax.plot([-10000,10000],[median_pressure_cloud_top[0],median_pressure_cloud_top[0]],'k--', zorder=5)
            ax.annotate('Retrieved Median Cloud Top',[x_lim[0]+0.025*x_lim[1],10**(-0.025*(np.log10(y_lim[1])-np.log10(y_lim[0]))+np.log10(median_pressure_cloud_top[0]))],color = 'black',ha='left')

        # Plotting the true/input profile (interpolation for smoothing)
        y = np.nanquantile(self.temperature_full,0.5,axis=0)
        x = np.nanquantile(self.pressure_full,0.5,axis=0)
        yinterp = np.interp(self.input_pressure, x, y)
        ax.semilogy(gaussian_filter1d(self.input_temperature-yinterp,sigma = 100),(self.input_pressure),color ='black', label = 'Input Profile', zorder=6)

        # Plotting the true/input surface temperature/pressure
        ax.plot(self.input_temperature[-1]-yinterp[-1],self.input_pressure[-1],marker='s',color='C3',ms=7, markeredgecolor='black',lw=0, zorder=6,label = 'Input Surface')

        # If wanted: plotting the true/input cloud top temperature/pressure
        if plot_clouds:
            ind_ct = (np.argmin(np.abs(np.log10(self.true_pressure_cloud_top[0])-np.log10(np.nanquantile(self.pressure_full,0.5,axis=0)))))
            T_CT = np.nanquantile(self.temperature_full,0.5,axis=0)[ind_ct]
            if self.settings['clouds'] == 'opaque':
                   ax.plot(self.true_temperature_cloud_top[0]-T_CT ,self.true_pressure_cloud_top[0],marker='o',color='C1',lw=0,ms=7, markeredgecolor='black', zorder=6,label = 'Input Cloud Top')

        # Print the case identifier
        ax.annotate(case_identifier,[x_lim[1]-0.025*(x_lim[1]-x_lim[0]),10**(np.log10(y_lim[1])-0.025*(np.log10(y_lim[1])-np.log10(y_lim[0])))],ha='right',va='bottom')

        # If it is a single plot show the axes titles
        if ax_arg is None:
            ax.set_xlabel('Temperature Relative to Retrieved Median [K]')
            ax.set_ylabel('Pressure [bar]')
        
        # Set the limits for the plot axes
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.invert_yaxis()



        # Inlay plot
        # generate and position the inlay plot
        ax2 = rp_inlay.position_inlay(loc_surface,figure,ax_arg,ax)

        # Plotting the cloud top temperature/pressure
        if self.settings['clouds'] == 'opaque':
            # Define the plot titles
            ax2_xlabel = r'$T^\mathrm{cloud}_\mathrm{top}\,\,\left[\mathrm{K}\right]$'
            ax2_ylabel = r'$P^\mathrm{cloud}_\mathrm{top}\,\,\left[\mathrm{bar}\right]$'

            # Make a 2d histogram of the cloud top pressures and temperatures
            Z,X,Y=np.histogram2d(self.temperature_cloud_top[:,0],np.log10(self.pressure_cloud_top[:,0]),bins=30,range = [[np.min(self.temperature_cloud_top[:,0]),np.max(self.temperature_cloud_top[:,0])], [-5, 0]])

        else:
            # Define the plot titles
            ax2_xlabel = r'$\mathrm{T_0}\,\,\left[\mathrm{K}\right]$'
            ax2_ylabel = r'$\mathrm{P_0}\,\,\left[\mathrm{bar}\right]$'

            # Make a 2d histogram of the surface pressures and temperatures
            Z,X,Y=np.histogram2d(self.temperature[:,-1],np.log10(self.pressure[:,-1]),bins=30,range = [[np.min(self.temperature[:,-1]),np.max(self.temperature[:,-1])], [-5, 0]])

        # Generate the colormap and plot the contours of the 2d histogram
        map, norm, levels = rp_col.color_map(Z,color_levels,level_thresholds)
        ax2.contourf((X[:-1]+X[1:])/2,10**((Y[:-1]+Y[1:])/2),Z.T,cmap=map,norm=norm,levels=np.array(levels))

        # plot the true values that were used to generate the input spectrum
        ax2.semilogy(self.input_temperature,self.input_pressure,color ='black')
        ax2.semilogy(self.input_temperature[-1]-yinterp[-1],self.input_pressure[-1],marker='s',color='C3',lw=0,ms=7, markeredgecolor='black')
        try:
            ax2.semilogy(self.true_temperature_cloud_top[0],(self.true_pressure_cloud_top[0]),marker='o',color='C1',lw=0,ms=7, markeredgecolor='black')
        except:
            pass
        
        # Arange the ticks for the inlay
        rp_inlay.axesticks_inlay(ax2,ax2_xlabel,ax2_ylabel,loc_surface)

        # Set the limits and ticks for the axes
        # x axis
        ax2_xlim = [np.min(X[np.where(Z >= levels[0])[0]])*0.9,np.max(X[np.where(Z >= levels[0])[0]])*1.1]
        xticks = np.array([(1-pos)*ax2_xlim[0]+pos*ax2_xlim[1] for pos in [0.2,0.4,0.6,0.8]])
        roundx = np.abs(np.min([0,np.floor(np.log10(np.abs(xticks[1]-xticks[0])))-1]))
        ax2.set_xticks(xticks)
        if roundx>=0:
            ax2.set_xticklabels(((xticks*10**(-roundx)).astype(int)*10**(roundx)).astype(int),rotation=90)
        else:
            ax2.set_xticklabels(np.round(xticks,int(roundx)),rotation=90)
        ax2.set_xlim(ax2_xlim)

        # y axis
        ax2.set_yticks([1e-1,1e-2,1e-3,1e-4])
        ax2.set_ylim([1e0,1e-5])



        # Legend cosmetics
        handles, labels = ax.get_legend_handles_labels()
        if ax_arg is not None:
            pass
        elif plot_clouds:
            patch_handles = [rp_hndl.MulticolorPatch([tuple(color_levels[i, :]),tuple(3*[0.9-i*0.15])],[1,1]) for i in range(N_levels)]
            if quantiles_title is None:
                patch_labels = levels[:-1]
            else:
                patch_labels = quantiles_title
            lgd = ax.legend(handles+patch_handles,labels+patch_labels,\
                            handler_map={str:  rp_hndl.Handles(), rp_hndl.MulticolorPatch:  rp_hndl.MulticolorPatchHandler()}, ncol=1,loc='upper left',frameon=False)
        else:
            lgd = ax.legend(handles+['SurfacePressure'], labels+['True Surface'],handler_map={str:  rp_hndl.Handles()}, ncol=4,loc=[0.17,-0.25])



        # Save or pass back the figure
        if ax_arg is not None:
            return handles+['CloudTopPressure','SurfacePressure'], labels+['True Cloud Top','True Surface']
        elif save:
            plt.savefig(self.results_directory+'Plots/plot_pt_structure_residual.pdf', bbox_inches='tight',bbox_extra_artists=(lgd,))
            return figure, ax, ax2
        else:
            return figure, ax, ax2






    
    def PT_Histogram(self,ax = None, save=False, plot_clouds = False, color='C2', loc_surface='top right', x_lim =[0,1000], y_lim = [1e-6,1e4],skip=10):
    
        self.get_pt(skip=skip)

        # Start of the plotting
        ax_arg = ax
        if ax is None:
            figure = plt.figure(figsize = (10,5))
            ax = figure.add_axes([0.1, 0.1, 0.8, 0.8])
        else:
            pass
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)


        #plotting the uncertainty on the retrieved spectrum
        bins = np.array([np.linspace(x_lim[0],x_lim[1],400),np.logspace(np.log10(y_lim[0]),np.log10(y_lim[1]),400)])
        ax.hist2d(self.temperature_full.flatten()[~np.isnan(self.temperature_full.flatten())],self.pressure_full.flatten()[~np.isnan(self.pressure_full.flatten())],bins=bins,cmap='afmhot')
        ax.semilogy(self.input_temperature,self.input_pressure,color ='xkcd:azure', label = 'True Profile', zorder=3,lw=0.5)

        # True surface temperature and surface pressure
        ax.plot(x_lim,[self.input_pressure[-1],self.input_pressure[-1]],color='xkcd:red',linestyle = '-.', zorder=6,label = 'True Surface',lw=0.5)

        # True cloud top pressure and temperature
        if self.settings['clouds'] == 'opaque':
            ax.plot(x_lim,[self.true_pressure_cloud_top[0],self.true_pressure_cloud_top[0]],color='xkcd:magenta',linestyle = '--', zorder=6,label = 'True Cloud Top',lw=0.5)

        if ax_arg is None:
            ax.set_xlabel('Temperature [K]')
            ax.set_ylabel('Pressure [bar]')
        ax.invert_yaxis()


        handles, labels = ax.get_legend_handles_labels()
        if ax_arg is not None:
            pass
        else:
            lgd = ax.legend(handles, labels,handler_map={str:  rp_hndl.Handles()}, ncol=4,loc=[0.13,-0.25])

        if ax_arg is not None:
            return handles, labels
        elif save:
            plt.savefig(self.results_directory+'Plots/pt_hist.png', bbox_inches='tight',bbox_extra_artists=(lgd,),dpi = 450)
        else:
            return figure











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
        print('Starting spectrum calculation.')
        print('\t0.00 % of spectra calculated.', end = "\r")

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
                    self.wavelength = nc.c / self.rt_object.freq #self.dwlen[instrument]
                    self.flux_true = np.array(self.rt_object.flux)
                elif i == 1:
                    self.wavelength = nc.c / self.rt_object.freq #self.dwlen[instrument]
                    self.flux_best = np.array(self.rt_object.flux)
                elif not hasattr(self,'retrieved_fluxes'):
                    self.wavelength = nc.c / self.rt_object.freq #self.dwlen[instrument]
                    self.retrieved_fluxes = np.array([self.rt_object.flux])
                else:
                    self.retrieved_fluxes = np.append(self.retrieved_fluxes,np.array([self.rt_object.flux]),axis=0)
            
            # Print status of calculation
            if process == 0:
                t_end = t.time()
                remain_time = (t_end-t_start)/((i+1)/(ind_end-ind_start))-(t_end-t_start)
                print('\t'+str(np.round((i+1)/(ind_end-ind_start)*100,2))+' % of spectra calculated. Estimated time remaining: '+str(remain_time//3600)+' h, '+str((remain_time%3600)//60)+' min.         ', end = "\r")

        # Print status of calculation
        if process == 0:
            print('\nSpectrum calculation completed.')

        #return the calculated results
        pass


    

    def Flux_Error(self,ax = None,skip=10,quantiles = [0.01,0.10,0.25,0.75,0.90,0.99],save=False,pptx=True,plot_points=False,n_processes=10):
        '''
        This Function creates a plot that visualizes the absolute uncertainty on the
        retrieval results in comparison with the input spectrum for the retrieval, the
        retrieved best fit spectrum, the retrieved mean spectrum and the retrieved
        median spectrum.
        '''

        self.get_fluxes(skip=skip,n_processes=n_processes)
        
        # Determining the quantiles of the different retrieved spectra
        flux_quantiles = [np.quantile(self.retrieved_fluxes,q,axis=0) for q in quantiles]

        # Start generating the flux error plot
        ax_arg = ax
        if ax is None:
            if pptx:
                figure = plt.figure(figsize = (7,5))
                ax = figure.add_axes([0.12, 0.12, 0.87, 0.83])
                fs = 16
                ms=2
            else:
                figure = plt.figure(figsize = (10,5))
                ax = figure.add_axes([0.1, 0.1, 0.8, 0.8])
                fs = 8 
                ms = 1
        else:
            pass
        #plt.plot(self.wavelength,self.flux_true,'k-',linewidth = 0.5,label = 'True Value Spectrum')
        try:
            ax.errorbar(self.input_wavelength,self.input_flux,yerr=self.input_error,color = 'red',marker='o',ms=ms,ls='',lw=0.5,label = 'Input Spectrum',zorder=3)
            ax.plot(self.true_wavelength,self.true_flux,color = 'black',ls='-',label = 'True Spectrum',lw=0.5,zorder=2)
        except:
            ax.plot(self.input_wavelength,self.input_flux,color = 'black',ls='-',label = 'Input Spectrum',lw=0.5,zorder=2)
       
        # Plotting results for the best fit, the mean fit and the median fit
        #ax.plot(self.wavelength/1e-4,self.flux_best,color = 'red',ls ='-',label = 'Best fit Spectrum')
        #ax.plot(self.wavelength/1e-4,self.flux_mean,color = 'purple',ls ='-',label = 'Mean fit Spectrum')
        #ax.plot(self.wavelength/1e-4,self.flux_median,color = 'green',ls ='-',label = 'Median fit Spectrum')
        
        # Plotting the flux quantiles corresponding to the retrieved parameters
        num_quantiles = len(quantiles)
        for q in range(num_quantiles-1):
            alpha = 0.3 + 0.5*(1-abs(1-q/(num_quantiles//2-1)))
            if num_quantiles//2 > q:
                ax.fill_between(self.wavelength/1e-4,flux_quantiles[q],flux_quantiles[q+1],
                                facecolor='green',alpha=alpha,label = str(int(100*quantiles[q]))
                                +r'% - '+str(int(100*quantiles[-q-1]))+'%',zorder=1)
            else:
                ax.fill_between(self.wavelength/1e-4,flux_quantiles[q],flux_quantiles[q+1],
                                facecolor='green',alpha=alpha,zorder=1)
        
        # Formatting of the plot and adding titles and the legend
        ax.set_xlabel(r'Wavelength [$\mu$m]',size=fs)
        ax.set_ylabel(r'Flux $\left[\mathrm{\frac{erg}{cm^2\,s\,Hz}}\right]$',size=fs)
        ax.set_xlim([np.min(self.wavelength)/1e-4,np.max(self.wavelength)/1e-4])
        
        # Either save the plot or return the figure object to allow for further formatting by the user
        ax.tick_params(axis='both', labelsize = fs)
        ax.yaxis.offsetText.set_fontsize(fs)

        handles, labels = ax.get_legend_handles_labels()
        if ax_arg is not None:
            return handles, labels
        elif save:
            if pptx:
                ax.set_ylim([0,ax.get_ylim()[1]])
                ax.legend(loc='upper left',fontsize=fs)
                plt.savefig(self.results_directory+'Plots/PPTX_plot_flux_error.pdf')
            else:
                ax.legend(loc='best',fontsiz=fs)
                plt.savefig(self.results_directory+'Plots/plot_flux_error.pdf')
        else:
            ax.legend(loc='best')
            return figure

    def Flux_Error_noQuant(self,ax = None,skip=10,save=False,pptx=True,n_processes=10):
        '''
        This Function creates a plot that visualizes the absolute uncertainty on the
        retrieval results in comparison with the input spectrum for the retrieval, the
        retrieved best fit spectrum, the retrieved mean spectrum and the retrieved
        median spectrum.
        '''

        self.get_fluxes(skip=skip,n_processes=n_processes)

        # Start generating the flux error plot
        ax_arg = ax
        
        if ax is None:
            if pptx:
                figure = plt.figure(figsize = (7,5))
                ax = figure.add_axes([0.12, 0.12, 0.87, 0.83])
                fs = 16
                ms=2
            else:
                figure = plt.figure(figsize = (10,5))
                ax = figure.add_axes([0.1, 0.1, 0.8, 0.8])
                fs = 8 
                ms = 1
        else:
            pass
        #plt.plot(self.wavelength,self.flux_true,'k-',linewidth = 0.5,label = 'True Value Spectrum')
        try:
            ax.errorbar(self.input_wavelength,self.input_flux,yerr=self.input_error,color = 'red',marker='o',ms=ms,ls='',lw=0.5,label = 'Input Spectrum',zorder=3)
            ax.plot(self.true_wavelength,self.true_flux,color = 'black',ls='-',label = 'True Spectrum',lw=0.5,zorder=1)
            ax.fill_between(self.true_wavelength,self.true_flux-self.true_error,self.true_flux+self.true_error, facecolor='grey',alpha=0.3,zorder=1)
        except:
            ax.plot(self.input_wavelength,self.input_flux,color = 'black',ls='-',label = 'Input Spectrum',lw=0.5,zorder=1)
            ax.fill_between(self.input_wavelength,self.input_flux-self.input_error,self.input_flux+self.input_error, facecolor='grey',alpha=0.3,zorder=1)




        # Plotting results for the best fit, the mean fit and the median fit
        #ax.plot(self.wavelength/1e-4,self.flux_best,color = 'red',ls ='-',label = 'Best fit Spectrum',lw=0.75,zorder=2)
        #ax.plot(self.wavelength/1e-4,self.flux_mean,color = 'purple',ls ='-',label = 'Mean fit Spectrum',lw=0.75,zorder=2)
        #ax.plot(self.wavelength/1e-4,self.flux_median,color = 'green',ls ='-',label = 'Median fit Spectrum',lw=0.75,zorder=2)

        # Formatting of the plott and adding titles and the legend
        ax.set_xlabel(r'Wavelength [$\mu$m]',size=fs)
        ax.set_ylabel(r'Flux $\left[\mathrm{\frac{erg}{cm^2\,s\,Hz}}\right]$',size=fs)
        ax.set_xlim([np.min(self.wavelength)/1e-4,np.max(self.wavelength)/1e-4])

        # Either save the plot or return the figure object to allow for further formatting by the user
        ax.tick_params(axis='both', labelsize = fs)
        ax.yaxis.offsetText.set_fontsize(fs)
        handles, labels = ax.get_legend_handles_labels()
        if ax_arg is not None:
            return handles, labels
        elif save:
            ax.legend(loc='best')
            plt.savefig(self.results_directory+'Plots/plot_flux_error_noquant.png',dpi = 450)
        else:
            ax.legend(loc='best')
            return figure


    def Input_Flux(self,ax = None,skip=10,save=False,pptx=True,n_processes=10):
        '''
        This Function creates a plot that visualizes the absolute uncertainty on the
        retrieval results in comparison with the input spectrum for the retrieval, the
        retrieved best fit spectrum, the retrieved mean spectrum and the retrieved
        median spectrum.
        '''

        self.get_fluxes(skip=skip,n_processes=n_processes)

        # Start generating the flux error plot
        ax_arg = ax
        if ax is None:
            if pptx:
                figure = plt.figure(figsize = (7,5))
                ax = figure.add_axes([0.12, 0.12, 0.87, 0.83])
                fs = 16
                ms=2
            else:
                figure = plt.figure(figsize = (10,5))
                ax = figure.add_axes([0.1, 0.1, 0.8, 0.8])
                fs = 8 
                ms = 1
        else:
            pass
        #plt.plot(self.wavelength,self.flux_true,'k-',linewidth = 0.5,label = 'True Value Spectrum')
        try:
            ax.errorbar(self.input_wavelength,self.input_flux,yerr=self.input_error,color = 'C3',marker='o',ms=ms,ls='',lw=0.5,label = 'Input Spectrum',zorder=3)
            ax.plot(self.true_wavelength,self.true_flux,color = 'black',ls='-',label = 'True Spectrum',lw=0.5,zorder=1)
            ax.fill_between(self.true_wavelength,self.true_flux-self.true_error,self.true_flux+self.true_error, facecolor='grey',alpha=0.3,zorder=1)
        except:
            ax.plot(self.input_wavelength,self.input_flux,color = 'black',ls='-',label = 'Input Spectrum',lw=0.5,zorder=1)
            ax.fill_between(self.input_wavelength,self.input_flux-self.input_error,self.input_flux+self.input_error, facecolor='grey',alpha=0.3,zorder=1)


        # Formatting of the plott and adding titles and the legend
        ax.set_xlabel(r'Wavelength [$\mu$m]',size = fs)
        ax.set_ylabel(r'Flux $\left[\mathrm{\frac{erg}{cm^2\,s\,Hz}}\right]$',size=fs)
        ax.set_xlim([np.min(self.wavelength)/1e-4,np.max(self.wavelength)/1e-4])
        ax.tick_params(axis='both', labelsize = fs)
        ax.yaxis.offsetText.set_fontsize(fs)
        # Either save the plot or return the figure object to allow for further formatting by the user
        handles, labels = ax.get_legend_handles_labels()
        if ax_arg is not None:
            return handles, labels
        elif save:
            if pptx:
                ax.legend(loc='upper left',fontsize=fs)
                plt.savefig(self.results_directory+'Plots/PPTX_Input_Spectrum.pdf',dpi = 450)
            else:
                ax.legend(loc='best',fontsize=fs)
                plt.savefig(self.results_directory+'Plots/Input_Spectrum.pdf',dpi = 450)
        else:
            ax.legend(loc='best')
            return figure



    def Rel_Flux_Error(self,skip=10,quantiles = [0.0014,0.0228,0.1587,0.8413,0.9772,0.9986],quantiles_title = [r'$5-95\%$',r'$15-85\%$',r'$25-75\%$',r'$35-65\%$'],
                        save=False,ax=None,plot_best_fits=False,y_lim=[-4,4],color = 'C2',case_identifier='',n_processes=10):
        '''
        This Function creates a plot that visualizes the relative uncertainty on the
        retrieval results in comparison with the input spectrum for the retrieval.
        ((F_retrieved-F_input)/F_input). We also plot the same for the retrieved
        best fit spectrum, the retrieved mean spectrum and the retrieved
        median spectrum.
        '''

        self.get_fluxes(skip=skip,n_processes=n_processes)

        # Determining the quantiles of the different retrieved spectra
        flux_quantiles = [np.quantile(self.retrieved_fluxes,q,axis=0) for q in quantiles]

        # Start generating the relative flux error plot


        # Start of the plotting
        ax_arg = ax
        alpha_step=2/len(quantiles)
        if ax is None:
            figure = plt.figure(figsize =(12,2)) #13.6,
            ax = figure.add_axes([0.1, 0.1, 0.8, 0.8])
        else:
            pass
        #ax.set_xlim(x_lim)
        #ax.set_ylim(y_lim)
        ax.plot(self.input_wavelength,self.input_flux*0,color = 'black',ls =':')#,label = 'Input Spectrum')
        x_lim = [np.min(self.wavelength)/1e-4,np.max(self.wavelength)/1e-4]

        # Plotting results for the best fit, the mean fit and the median fit
        ax.fill_between(self.input_wavelength,-self.input_error/self.input_flux*100,self.input_error/self.input_flux*100, facecolor='black',alpha=0.2,label = 'Noise')
        if plot_best_fits:
            ax.plot(self.wavelength/1e-4,(self.flux_best-self.input_flux)/self.input_flux,color = 'C3',ls ='--',label = 'Best fit Spectrum')
            ax.plot(self.wavelength/1e-4,(self.flux_mean-self.input_flux)/self.input_flux,color = 'C4',ls ='-.',label = 'Mean fit Spectrum')
            ax.plot(self.wavelength/1e-4,(self.flux_median-self.input_flux)/self.input_flux,color = 'C2',ls ='-',label = 'Median fit Spectrum')
        
        # Plotting the flux quantiles corresponding to the retrieved parameters
        ax.fill_between(self.wavelength/1e-4,(flux_quantiles[0]-self.input_flux)/self.input_flux*100,
                        (flux_quantiles[-1]-self.input_flux)/self.input_flux*100, zorder=2,facecolor='white')
        for q in range(int(len(quantiles)/2)):
            if q == len(quantiles)-q-2:
                print()
                ax.fill_between(self.wavelength/1e-4,(flux_quantiles[q]-self.input_flux)/self.input_flux*100,
                                (flux_quantiles[q+1]-self.input_flux)/self.input_flux*100, zorder=2,facecolor=color
                                ,alpha=alpha_step+q*alpha_step,label = quantiles_title[q])
            else:
                ax.fill_between(self.wavelength/1e-4,(flux_quantiles[q]-self.input_flux)/self.input_flux*100,
                                (flux_quantiles[q+1]-self.input_flux)/self.input_flux*100, zorder=2,facecolor=color
                                ,alpha=alpha_step+q*alpha_step,label = quantiles_title[q])

                ax.fill_between(self.wavelength/1e-4,(flux_quantiles[-q-1]-self.input_flux)/self.input_flux*100,
                                (flux_quantiles[-q-2]-self.input_flux)/self.input_flux*100, zorder=2,facecolor=color
                                ,alpha=alpha_step+q*alpha_step)
        
        # Formatting of the plott and adding titles and the legend        13.6,2
        print(case_identifier)
        print([x_lim[1]-4/5*(x_lim[1]-x_lim[0]),y_lim[1]-4/5*(y_lim[1]-y_lim[0])])
        ax.annotate(case_identifier,[x_lim[1]-1/(13.6/2*15/2)*(x_lim[1]-x_lim[0]),y_lim[1]-14/15*(y_lim[1]-y_lim[0])],ha='right',va='bottom')
        patch_handle = [rp_hndl.MulticolorPatch(['black'],[0.2])]+[rp_hndl.MulticolorPatch([color],[0.2+i*alpha_step]) for i in range(len(quantiles_title))]
        patch_label = [r'LIFE Noise']+list(quantiles_title)
        lgd = ax.legend(patch_handle,patch_label,\
                        handler_map={str:  rp_hndl.Handles(), rp_hndl.MulticolorPatch:  rp_hndl.MulticolorPatchHandler()}, ncol=5,loc='upper center',frameon=False)

        #ax.legend(loc='best',frameon=False)
        ax.set_xlabel(r'Wavelength [$\mu$m]')
        ax.set_ylabel(r'Retrieval Residual [%]')
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        
        # Either save the plot or return the figure object to allow for further formatting by the user
        if save:
            print(self.results_directory+'Plots/plot_rel_flux_error_skip_'+str(skip)+'.pdf')
            plt.savefig(self.results_directory+'Plots/plot_rel_flux_error_skip_'+str(skip)+'.pdf',bbox_inches='tight')
        else:
            return figure

    def Rel_Flux_Error_Noise(self,skip=10,quantiles = [0.0014,0.0228,0.1587,0.8413,0.9772,0.9986],quantiles_title = [r'$5-95\%$',r'$15-85\%$',r'$25-75\%$',r'$35-65\%$'],
                        save=False,ax=None,plot_best_fits=False,alpha_step = 0.15,y_lim=[-2,2],color='C2',n_processes=10):
        '''
        This Function creates a plot that visualizes the relative uncertainty on the
        retrieval results in comparison with the input spectrum for the retrieval.
        ((F_retrieved-F_input)/F_input). We also plot the same for the retrieved
        best fit spectrum, the retrieved mean spectrum and the retrieved
        median spectrum.
        '''

        self.get_fluxes(skip=skip,n_processes=n_processes)

        # Determining the quantiles of the different retrieved spectra
        flux_quantiles = [np.quantile(self.retrieved_fluxes,q,axis=0) for q in quantiles]

        # Start generating the relative flux error plot


        # Start of the plotting
        ax_arg = ax
        if ax is None:
            figure = plt.figure() #figsize = (10,5))
            ax = figure.add_axes([0.1, 0.1, 0.8, 0.8])
        else:
            pass
        #ax.set_xlim(x_lim)
        #ax.set_ylim(y_lim)
        ax.plot(self.input_wavelength,self.input_flux*0,color = 'black',ls =':')#,label = 'Input Spectrum')

        # Plotting results for the best fit, the mean fit and the median fit
        #ax.fill_between(self.input_wavelength,-self.input_error/self.input_flux*100,self.input_error/self.input_flux*100, facecolor='black',alpha=0.2,label = 'Noise')
        if plot_best_fits:
            ax.plot(self.wavelength/1e-4,(self.flux_best-self.input_flux)/self.input_flux,color = 'C3',ls ='--',label = 'Best fit Spectrum')
            ax.plot(self.wavelength/1e-4,(self.flux_mean-self.input_flux)/self.input_flux,color = 'C4',ls ='-.',label = 'Mean fit Spectrum')
            ax.plot(self.wavelength/1e-4,(self.flux_median-self.input_flux)/self.input_flux,color = 'C2',ls ='-',label = 'Median fit Spectrum')
        
        # Plotting the flux quantiles corresponding to the retrieved parameters
        ax.fill_between(self.wavelength/1e-4,(flux_quantiles[0]-self.input_flux)/self.input_flux*100,
                        (flux_quantiles[-1]-self.input_flux)/self.input_flux*100, zorder=2,facecolor='white')
        for q in range(int(len(quantiles)/2)):
            if q == len(quantiles)-q-2:
                print()
                ax.fill_between(self.wavelength/1e-4,(flux_quantiles[q]-self.input_flux)/self.input_error*100,
                                (flux_quantiles[q+1]-self.input_flux)/self.input_error*100, zorder=2,facecolor=color
                                ,alpha=0.1+q*alpha_step,label = quantiles_title[q])
            else:
                ax.fill_between(self.wavelength/1e-4,(flux_quantiles[q]-self.input_flux)/self.input_error*100,
                                (flux_quantiles[q+1]-self.input_flux)/self.input_error*100, zorder=2,facecolor=color
                                ,alpha=0.1+q*alpha_step,label = quantiles_title[q])

                ax.fill_between(self.wavelength/1e-4,(flux_quantiles[-q-1]-self.input_flux)/self.input_error*100,
                                (flux_quantiles[-q-2]-self.input_flux)/self.input_error*100, zorder=2,facecolor=color
                                ,alpha=0.1+q*alpha_step)
        
        # Formatting of the plott and adding titles and the legend

        patch_handle = [rp_hndl.MulticolorPatch([color],0.1+i*alpha_step) for i in range(len(quantiles_title))]+[]
        patch_label = quantiles_title
        lgd = ax.legend(patch_handle,patch_label,\
                            handler_map={str:  rp_hndl.Handles(), rp_hndl.MulticolorPatch:  rp_hndl.MulticolorPatchHandler()}, ncol=1,loc='upper left',frameon=False)

        ax.legend(loc='best',frameon=False)
        ax.set_xlabel(r'Wavelength [$\mu$m]')
        ax.set_ylabel(r'Retrieval Residual [\%]')
        ax.set_xlim([np.min(self.wavelength)/1e-4,np.max(self.wavelength)/1e-4])
        # ax.set_ylim(y_lim)
        
        # Either save the plot or return the figure object to allow for further formatting by the user
        if save:
            plt.savefig(self.results_directory+'Plots/plot_rel_flux_error_noise_skip_'+str(skip)+'.pdf',bbox_inches='tight')
        else:
            return figure



    def Rel_Flux_Error_noQuant(self,skip=10,n_processes=10,save=False):
       '''
       This Function creates a plot that visualizes the relative uncertainty on the
       retrieval results in comparison with the input spectrum for the retrieval.
       ((F_retrieved-F_input)/F_input). We also plot the same for the retrieved
       best fit spectrum, the retrieved mean spectrum and the retrieved
       median spectrum.
       '''

       self.get_fluxes(skip=skip,n_processes=n_processes)

       # Start generating the relative flux error plot
       figure = plt.figure()
       ind  = np.intersect1d(self.wavelength,self.input_wavelength,return_indices=True)[1]
       #plt.plot(self.input_wavelength,self.input_flux*0,color = 'black',ls ='-',label = 'Input Spectrum')


       # Plotting results for the best fit, the mean fit and the median fit
       plt.fill_between(self.input_wavelength,-self.input_error/self.input_flux,self.input_error/self.input_flux, facecolor='grey',alpha=0.3)
       plt.plot(self.input_wavelength,self.input_flux*0,color = 'black',ls ='-',lw=0.75)
       plt.plot(self.wavelength/1e-4,(self.flux_best-self.input_flux)/self.input_flux,color = 'red',lw=0.75,ls ='-',label = 'Best fit Spectrum')
       plt.plot(self.wavelength/1e-4,(self.flux_mean-self.input_flux)/self.input_flux,color = 'purple',lw=0.75,ls ='-',label = 'Mean fit Spectrum')
       plt.plot(self.wavelength/1e-4,(self.flux_median-self.input_flux)/self.input_flux,color = 'green',lw=0.75,ls ='-',label = 'Median fit Spectrum')
       plt.ylim([-0.075,0.075])

       # Formatting of the plott and adding titles and the legend
       plt.legend(loc='best')
       plt.xlabel(r'Wavelength [$\mu$m]')
       plt.ylabel(r'$1-F_\mathrm{retrieved}/F_\mathrm{input}$')
       plt.xlim([np.min(self.wavelength)/1e-4,np.max(self.wavelength)/1e-4])

       # Either save the plot or return the figure object to allow for further formatting by the user
       if save:
               plt.savefig(self.results_directory+'Plots/plot_rel_flux_error_noquant.png',dpi = 450)

       else:
           return figure









    def Plot_Ret_Bond_Albedo(self,L_star,err_L_star,sep_planet,err_sep_planet,A_Bond_true = None,T_equ_true= None,
                            skip=1,quantiles=[0.16, 0.5, 0.84],bins=50,save=False,plot=True):
        
        self.get_pt(skip=skip)

        if self.settings['clouds'] == 'opaque': #if len(self.cloud_vars) != 0:
            ret_surface_T = self.temperature_cloud_top[:,0]
            #ret_surface_T = self.temperature[:,-1]
        else:
            ret_surface_T = self.temperature[:,-1]
        self.ret_surface_T = ret_surface_T

        # Generating random data for the stellar luminosity and the panet separation
        L_star_data = L_star + err_L_star*np.random.randn(*ret_surface_T.shape)
        sep_planet_data = sep_planet + err_sep_planet*np.random.randn(*ret_surface_T.shape)

        # Defining constants needed for the calculations
        L_sun = 3.826*1e26
        AU = 1.495978707*1e11
        sigma_SBoltzmann = 5.670374419*1e-8

        # Converting stellar luminosity and planet separation to SI
        L_star_data_SI = L_star_data * L_sun
        sep_planet_data_SI = sep_planet_data * AU

        # Calculate the bond albedo
        self.A_Bond_ret = 1 - 16*np.pi*sep_planet_data_SI**2*sigma_SBoltzmann*ret_surface_T**4/L_star_data_SI




        
        if plot == True:
            # Define the figure and the font size fs depending on the number of
            # overall retrieved parameters
            data = np.vstack([L_star_data,sep_planet_data,ret_surface_T,self.A_Bond_ret]).T
            titles = [r'$\mathrm{L_{Star}}$',r'$\mathrm{a_{Planet}}$',r'$\mathrm{T_{eq,\,Planet}}$',r'$\mathrm{A_{B,\,Planet}}$']
            units = [r'$\left[\mathrm{L}_\odot\right]$',r'$\left[\mathrm{AU}\right]$',r'$\left[\mathrm{K}\right]$','']
            truths = [L_star,sep_planet,T_equ_true,A_Bond_true]
            precision = 2

            fig = self.Corner(data,titles,truths=truths,units=units,bins=bins,quantiles=quantiles)

            # Save the figure or retrun the figure object
            if save:
                plt.savefig(self.results_directory+'Plots/plot_Bond_Albedo.png',dpi = 400)
            else:
                return fig
        else:
            pass






    def Plot_Ice_Surface_Test(self,MMW_atm=44,MMW_H2O=18,ax=None,skip=1,bins=50,save=False):
        
        if not hasattr(self, 'pressure'):
            self.Calc_PT_Profiles(skip = skip)
        if self.settings['clouds'] == 'opaque': #if len(self.cloud_vars) != 0:
            ret_surface_T = self.temperature_cloud_top[:,0]
            ret_surface_p = self.pressure_cloud_top[:,0]
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


    













class grid_plotting():

    def __init__(self, results_directory):
        '''
        This function reads the input.ini file as well as the retrieval
        results files of imterest to us and stores the read in data
        in order to generate the retrieval plots of interest to us.
        '''

        self.results_directory = results_directory
        self.grid_results = {}
        self.model = set()
        self.wln = set()
        self.res = set()
        self.snr = set()

        # Iterating over all directories in the grid results folders
        for model_item in os.listdir(results_directory):
            model_type = os.path.join(results_directory, model_item)
            if os.path.isdir(model_type):
                self.grid_results[model_item]={}
                self.model.add(model_item)
                for item in os.listdir(model_type):
                    model_run = os.path.join(model_type, item)
                    if os.path.isdir(model_run):
                        split = model_run.split('_')

                        # Initiating instances of the retrieval plotting class and storing the data in a dict
                        results = retrieval_plotting(model_run+'/')
                        self.wln.add(split[-1])
                        self.res.add(int(split[-2][1:]))
                        self.snr.add(int(split[-3][2:]))
                        try:
                            self.grid_results[model_item][split[-1]][int(split[-2][1:])][int(split[-3][2:])]=results
                        except:
                            try:
                                self.grid_results[model_item][split[-1]][int(split[-2][1:])]={}
                                self.grid_results[model_item][split[-1]][int(split[-2][1:])][int(split[-3][2:])]=results
                            except:
                                self.grid_results[model_item][split[-1]]={}
                                self.grid_results[model_item][split[-1]][int(split[-2][1:])]={}
                                self.grid_results[model_item][split[-1]][int(split[-2][1:])][int(split[-3][2:])]=results




    def PT_grid(self,plot_clouds=False,Hist = False):
        
        # Iterate over models and wavelength ranges
        for model in self.model:
            for wln in self.wln:

                fig,axs = plt.subplots(len(self.res),len(self.snr),figsize=(15,10))
                plt.subplots_adjust(hspace=0.03,wspace=0.025)
                
                # Iterate over spectral resolutions and wavelength ranges
                count_res = 0
                for res in sorted(self.res):
                    count_snr = 0
                    for snr in sorted(self.snr):
                        print()
                        print('PT-Profile Calculation for:')
                        print(model,wln,res,snr)
                        if Hist == True:
                            handles, labels = self.grid_results[model][wln][res][snr].PT_Histogram(ax = axs[count_res,count_snr], plot_clouds = plot_clouds)
                        else:
                            #try:
                            # Subroutine to plot the subplots
                            if 'Cloudfree' in model:
                                handles, labels = self.grid_results[model][wln][res][snr].PT_Envelope(ax = axs[count_res,count_snr], plot_clouds = False)
                            else:
                                handles, labels = self.grid_results[model][wln][res][snr].PT_Envelope(ax = axs[count_res,count_snr], plot_clouds = plot_clouds)
                            #except:
                            #    pass
                        
                        lim = axs[count_res,count_snr].get_ylim()
                        axs[count_res,count_snr].set_yticks(10**(np.log10(lim[0])+np.array([0.1,0.3,0.5,0.7,0.9])*(np.log10(lim[1])-np.log10(lim[0]))))
                        if count_snr == 0:
                            axs[count_res,count_snr].set_ylabel(r'R = '+str(res))
                        else:
                            axs[count_res,count_snr].set_yticklabels([])


                        lim = axs[count_res,count_snr].get_xlim()
                        axs[count_res,count_snr].set_xticks(lim[0]+np.array([0.1,0.3,0.5,0.7,0.9])*(lim[1]-lim[0]))
                        if count_res == len(self.res)-1:
                            axs[count_res,count_snr].set_xlabel(r'S/N = '+str(snr))
                        else:
                            axs[count_res,count_snr].set_xticklabels([])
                        count_snr += 1
                    count_res += 1
                    
                ylabel = fig.text(0.07,0.5, 'Pressure [bar]', va='center', rotation='vertical')
                xlabel = fig.text(0.515,0.05, 'Temperature [K]', ha='center')

                try:
                    legend=fig.legend(handles, labels,handler_map={str:  rp_hndl.Handles()}, ncol=4, bbox_to_anchor=(0.7,0.04),borderaxespad=0.0)
                    title = fig.text(0.5,0.9,wln+r' $\mu\mathrm{m}$',ha='center',fontsize=20)
                    if Hist == True:
                        plt.savefig(self.results_directory+'/Plots/'+model+'/ret_pt_hist_'+wln+'.png',bbox_extra_artists=[legend,xlabel,ylabel,title], bbox_inches='tight',dpi=600)
                    else:
                        plt.savefig(self.results_directory+'/Plots/'+model+'/ret_pt_'+wln+'.png',bbox_extra_artists=[legend,xlabel,ylabel,title], bbox_inches='tight',dpi=600)
                except:
                    title = fig.text(0.5,0.9,wln+r' $\mu\mathrm{m}$',ha='center',fontsize=20)
                    if Hist == True:
                        plt.savefig(self.results_directory+'/Plots/'+model+'/ret_pt_hist_'+wln+'.png',bbox_extra_artists=[xlabel,ylabel,title], bbox_inches='tight',dpi=600)
                    else:
                        plt.savefig(self.results_directory+'/Plots/'+model+'/ret_pt_'+wln+'.png',bbox_extra_artists=[xlabel,ylabel,title], bbox_inches='tight',dpi=600)



    def Ice_Surface(self):
        
        # Iterate over models and wavelength ranges
        for model in self.model:
            for wln in self.wln:

                fig,axs = plt.subplots(len(self.res),len(self.snr),figsize=(15,10))
                plt.subplots_adjust(hspace=0.03,wspace=0.025)
                
                # Iterate over spectral resolutions and wavelength ranges
                count_res = 0
                for res in sorted(self.res):
                    count_snr = 0
                    for snr in sorted(self.snr):
                        print()
                        print('PT-Profile Calculation for:')
                        print(model,wln,res,snr)
                        #try:
                        # Subroutine to plot the subplots
                        handles, labels = self.grid_results[model][wln][res][snr].Plot_Ice_Surface_Test(ax=axs[count_res,count_snr])
                        #except:
                        #    pass
                        
                        lim = axs[count_res,count_snr].get_ylim()
                        #axs[count_res,count_snr].set_yticks(10**(np.log10(lim[0])+np.array([0.1,0.3,0.5,0.7,0.9])*(np.log10(lim[1])-np.log10(lim[0]))))
                        if count_snr == 0:
                            axs[count_res,count_snr].set_ylabel(r'R = '+str(res))
                        axs[count_res,count_snr].set_yticklabels([])


                        lim = axs[count_res,count_snr].get_xlim()
                        #axs[count_res,count_snr].set_xticks(lim[0]+np.array([0.1,0.3,0.5,0.7,0.9])*(lim[1]-lim[0]))
                        if count_res == len(self.res)-1:
                            axs[count_res,count_snr].set_xlabel(r'S/N = '+str(snr))
                        else:
                            axs[count_res,count_snr].set_xticklabels([])
                        count_snr += 1
                    count_res += 1
                    
                #ylabel = fig.text(0.07,0.5, 'Pressure [bar]', va='center', rotation='vertical')
                xlabel = fig.text(0.515,0.05, r'Abundance log$_10$', ha='center')

                try:
                    legend=fig.legend(handles, labels,handler_map={str:  rp_hndl.Handles()}, ncol=4, bbox_to_anchor=(0.7,0.04),borderaxespad=0.0)
                    title = fig.text(0.5,0.9,wln+r' $\mu\mathrm{m}$',ha='center',fontsize=20)
                    plt.savefig(self.results_directory+'/'+model+'/ice_surface_exclusion_'+wln+'.png',bbox_extra_artists=[legend,xlabel,title], bbox_inches='tight',dpi=600)
                except:
                    title = fig.text(0.5,0.9,wln+r' $\mu\mathrm{m}$',ha='center',fontsize=20)
                    plt.savefig(self.results_directory+'/'+model+'/ice_surface_exclusion_'+wln+'.png',bbox_extra_artists=[xlabel,title], bbox_inches='tight',dpi=600)




    def Spectrum_grid(self,skip = 10, NoQuant = False):
        
        # Iterate over models and wavelength ranges
        for model in self.model:
            for wln in self.wln:
                print(len(self.res),len(self.snr))
                fig,axs = plt.subplots(len(self.res),len(self.snr),figsize=(15,10))
                plt.subplots_adjust(hspace=0.03,wspace=0.025)
                
                
                # Iterate over spectral resolutions and wavelength ranges
                count_res = 0
                for res in sorted(self.res):
                    count_snr = 0
                    for snr in sorted(self.snr):
                        try:
                            # Subroutine to plot the subplots
                            if NoQuant:
                                handles, labels = self.grid_results[model][wln][res][snr].Flux_Error_noQuant(ax = axs[count_res,count_snr],skip=skip)
                            else:
                                handles, labels = self.grid_results[model][wln][res][snr].Flux_Error(ax = axs[count_res,count_snr],skip=skip)       
                        except:
                            pass
                        
                        lim = list(axs[0,0].get_ylim())
                        print(lim)
                        lim[0]=0
                        axs[count_res,count_snr].set_yticks(lim[0]+np.array([0.1,0.3,0.5,0.7,0.9])*(lim[1]-lim[0]))
                        axs[count_res,count_snr].set_ylim(lim)
                        if count_snr == 0:
                            axs[count_res,count_snr].set_ylabel(r'R = '+str(res))
                        else:
                            axs[count_res,count_snr].set_yticklabels([])

                        lim = axs[0,0].get_xlim()
                        axs[count_res,count_snr].set_xticks(lim[0]+np.array([0.1,0.3,0.5,0.7,0.9])*(lim[1]-lim[0]))
                        axs[count_res,count_snr].set_xlim(lim)
                        if count_res == len(self.res)-1:
                            axs[count_res,count_snr].set_xlabel(r'S/N = '+str(snr))
                        else:
                            axs[count_res,count_snr].set_xticklabels([])
                        count_snr += 1
                    count_res += 1
                
                ylabel = fig.text(0.07,0.5, r'$\mathrm{Flux}$', va='center', rotation='vertical')
                xlabel = fig.text(0.515,0.05, r'Wavelength [$\mu$m]', ha='center')

                try:
                    legend=fig.legend(handles, labels,handler_map={str:  rp_hndl.Handles()}, ncol=4, bbox_to_anchor=(0.7,0.04),borderaxespad=0.0)
                    title = fig.text(0.5,0.9,wln+r' $\mu\mathrm{m}$',ha='center',fontsize=20)
                    if NoQuant:
                        plt.savefig(self.results_directory+'/'+model+'/ret_spect_noQuant_'+wln+'.png',bbox_extra_artists=[xlabel,ylabel,title,legend], bbox_inches='tight',dpi=600)
                    else:
                        plt.savefig(self.results_directory+'/'+model+'/ret_spect_'+wln+'.png',bbox_extra_artists=[xlabel,ylabel,title,legend], bbox_inches='tight',dpi=600)
                except:
                    plt.savefig(self.results_directory+'/'+model+'/ret_spect_'+wln+'.png',bbox_extra_artists=[xlabel,ylabel,title], bbox_inches='tight',dpi=600)


                


    def Parameters_Comparison(self, parameters=['R_pl','M_pl','P_equ','T_equ','A_Bond'], priors = [True, True, False, False, False], units=['$\mathrm{R_\oplus}$','dex','dex','K',''], span = [0.2,0.5,0.5,20,0.1], titles = None, bond_params = [1,0.1,0.723,0.0723,0.77,226,5e-2],filename=None):

        # Parameters for plotting
        ms = 8
        lw = 2
        eb = 0.025
        markers = ['o','s','D','v']
        colors = ['C3','C2','C0','C1']
        
        # Iterate over models and wavelength ranges
        for model in self.model:
            for wln in self.wln:

                # Initialize a new figure
                plt.figure(figsize = (10,6))
                plt.title(str(model)+', '+str(wln),fontsize=20)
                plt.yticks([],fontsize=20,rotation = 90)
                xticks = []

                # Iterate over the specified parameters
                count_param = 0
                for param in parameters:
                    #Plot the background
                    plt.plot([2*count_param+1,2*count_param+1],[-2,3],'-k',alpha=1)
                    plt.fill_betweenx([-2,3],2*count_param+1-0.5,2*count_param+1+0.5,color='darkcyan',alpha=0.1,lw=0)
                    plt.annotate(r'$-$'+str(span[count_param])+' '+str(units[count_param]),[2*count_param+1-0.5,-1.15],fontsize=15,rotation = 90,va='center',ha = 'left',rotation_mode='anchor')
                    plt.annotate(r'$+$'+str(span[count_param])+' '+str(units[count_param]),[2*count_param+1+0.5,-1.15],fontsize=15,rotation = 90,va='center',ha = 'left',rotation_mode='anchor')
                    xticks += [2*count_param+1]
                    

                    # Iterate over spectral resolutions and wavelength ranges
                    count_snr = 0
                    for snr in sorted(self.snr):
                        # SNR annotation
                        plt.annotate(r'$\mathrm{S/N}=$'+str(snr),[-0.9,0.32+count_snr*0.67],fontsize=15,rotation = 0,va='center',ha = 'left',rotation_mode='anchor')

                        count_res = 0
                        for res in sorted(self.res):
                            # Important for plotting
                            offset = 1/3+count_snr*2/3+(count_res-1.5)/8

                            # Check if the wanted variables were retrieved for
                            try:
                                # Link the parameters to the corresponding position in the data
                                keys = list(self.grid_results[model][wln][res][snr].params.keys())
                                ind = np.where(np.array(keys)==str(param))[0][0]

                                # Load the data and perform calculations if necessary
                                if units[count_param]=='dex':
                                    post = np.log10(self.grid_results[model][wln][res][snr].equal_weighted_post[:,ind])
                                    true = np.log10(self.grid_results[model][wln][res][snr].truths[ind])
                                else:
                                    post = self.grid_results[model][wln][res][snr].equal_weighted_post[:,ind]
                                    true = self.grid_results[model][wln][res][snr].truths[ind]
                                prior = self.grid_results[model][wln][res][snr].priors_range[ind]

                            # What to do If they were not retrieved for
                            except:
                                if param == 'A_Bond':
                                    # If not done already calculate the bond albedo from the data
                                    if not hasattr(self.grid_results[model][wln][res][snr], 'A_Bond_ret'):
                                        print()
                                        print('Bond Albedo Calculation for:')
                                        print(model,wln,res,snr)
                                        self.grid_results[model][wln][res][snr].Plot_Ret_Bond_Albedo(*bond_params[:4],A_Bond_true=bond_params[4],T_equ_true=bond_params[5],plot=False)
                                    post = self.grid_results[model][wln][res][snr].A_Bond_ret
                                    true = bond_params[4]

                                elif param in ['T0','T_eq','P_eq']:
                                    # If not done already calculate the PT profiles from the data
                                    if not hasattr(self.grid_results[model][wln][res][snr], 'pressure'):
                                        print()
                                        print('PT-Profile Calculation for:')
                                        print(model,wln,res,snr)
                                        self.grid_results[model][wln][res][snr].Calc_PT_Profiles(skip = 1)
                                    if param in ['T0','T_eq']:
                                        if self.grid_results[model][wln][res][snr].settings['clouds'] == 'opaque': #len(self.grid_results[model][wln][res][snr].cloud_vars) != 0:
                                            post = self.grid_results[model][wln][res][snr].temperature_cloud_top[:,0]
                                            true = bond_params[5]
                                        else:
                                            post = self.grid_results[model][wln][res][snr].temperature[:,-1]
                                            true = bond_params[5]
                                    else:
                                        if self.grid_results[model][wln][res][snr].settings['clouds'] == 'opaque':
                                            post = np.log10(self.grid_results[model][wln][res][snr].pressure_cloud_top[:,0])
                                            true = np.log10(bond_params[6])
                                        else:
                                            post = np.log10(self.grid_results[model][wln][res][snr].pressure[:,-1])
                                            true = np.log10(bond_params[6])
                                
                                # If none of the above cases the variable cannot be plotted
                                else:
                                    print('Error: Parameter '+ str(param) +' not defined in input file config.ini.')
                                    sys.exit()
                            
                            # Compute the quantiles
                            q50 = np.quantile(post-true,0.5,axis=0)
                            q84 = np.quantile(post-true,0.84,axis=0)
                            q16 = np.quantile(post-true,0.16,axis=0)

                            # Plot the data according to the quantiles
                            center = 0.5*q50/(span[count_param])+2*count_param+1
                            plt.plot([center-0.5*(q50-q84)/(span[count_param]),center-0.5*(q50-q16)/(span[count_param])],[offset,offset],ls='-',color=colors[count_res],linewidth=lw)
                            plt.plot([center-0.5*(q50-q16)/(span[count_param]),center-0.5*(q50-q16)/(span[count_param])],[eb+offset,-eb+offset],ls='-',color=colors[count_res],linewidth=lw)
                            plt.plot([center-0.5*(q50-q84)/(span[count_param]),center-0.5*(q50-q84)/(span[count_param])],[eb+offset,-eb+offset],ls='-',color=colors[count_res],linewidth=lw)
                            plt.plot([center,center],[offset,offset],color=colors[count_res],marker=markers[count_snr],markersize = ms,markeredgecolor='black',markeredgewidth=lw/4)

                            # If wanted plot the prior distributions
                            if priors[count_param] == True:
                                offset = -1/3+3/16
                                center = 0.5*(prior[0]-true)/(span[count_param])+2*count_param+1
                                plt.plot([center+0.5*prior[1]/(span[count_param]),center+0.5*prior[1]/(span[count_param])],[eb+offset,-eb+offset],ls='-',color='black',linewidth=lw)
                                plt.plot([center-0.5*prior[1]/(span[count_param]),center-0.5*prior[1]/(span[count_param])],[eb+offset,-eb+offset],ls='-',color='black',linewidth=lw)
                                plt.plot([center+0.5*prior[1]/(span[count_param]),center-0.5*prior[1]/(span[count_param])],[offset,offset],ls='-',color='black',linewidth=lw)
                                plt.plot([center],[offset],color='black',marker='o',markersize = ms)
                
                            count_res += 1
                        count_snr += 1
                    count_param += 1
                if titles is None:
                    plt.xticks(ticks=xticks,labels=parameters,fontsize=20)
                else:
                    plt.xticks(ticks=xticks,labels=titles,fontsize=20)

                # Final cosmetics
                lgd=plt.legend(['Markers:','SNR5','SNR10','SNR15','SNR20','','Line Colors:','R20','R35','R50','R100','Prior'],\
                    ['','S/N = 5','S/N = 10','S/N = 15','S/N = 20','','','R = 20','R = 35','R = 50','R = 100','Prior'],\
                    handler_map={str:  rp_hndl.Handles()},handlelength=4,ncol=2, bbox_to_anchor=(0.25, -0.65, 0.5, 0.5), borderaxespad=0.1,loc='center',fontsize=18)
                plt.xlim([-1,2*count_param+0.75])
                plt.ylim([-1.2,1/3+count_snr*2/3-1.5/8])

                # Save the plot
                if filename is None:
                    plt.savefig(self.results_directory+'/'+model+'/ret_planet_params_'+wln+'.png',dpi=450,bbox_extra_artists=[lgd], bbox_inches='tight')
                else:
                    plt.savefig(self.results_directory+'/'+model+'/'+str(filename)+'_'+wln+'.png',dpi=450,bbox_extra_artists=[lgd], bbox_inches='tight')















    def Parameters_Hist(self, parameters=['R_pl','M_pl','H2O','CO2','CO','H2SO484(c)_am_top_pressure','H2SO484(c)_am','H2SO484(c)_am_thickness','H2SO484(c)_am_particle_radius','H2SO484(c)_am_sigma_lnorm','A_Bond'], priors = [True, True, False, False, False], units=['$\mathrm{R_\oplus}$','dex','dex','dex','dex','dex','dex','dex','dex','','',''], span = [0.2,0.5,0.5,20,0.1], titles = None, bond_params = [1,0.05,0.723,0.05*0.723,0.77,226,5e-2],filename=None):

        # Parameters for plotting
        ms = 8
        lw = 2
        eb = 0.025
        markers = ['o','s','D','v']
        colors = ['C3','C2','C0','C1']
        
        # Iterate over models and wavelength ranges
        for model in self.model:
            for wln in self.wln:

                # Iterate over the specified parameters
                count_param = 0
                for param in parameters:

                    # Initialize a new figure
                    plt.figure(figsize=(3,3))
                    plt.title(param + ' ' + str(model)+', '+str(wln),fontsize=20)
                    #plt.yticks([],fontsize=20,rotation = 90)
                    #xticks = []

                    #Plot the background
                    #plt.plot([2*count_param+1,2*count_param+1],[-2,3],'-k',alpha=1)
                    #plt.fill_betweenx([-2,3],2*count_param+1-0.5,2*count_param+1+0.5,color='darkcyan',alpha=0.1,lw=0)
                    #plt.annotate(r'$-$'+str(span[count_param])+' '+str(units[count_param]),[2*count_param+1-0.5,-1.15],fontsize=15,rotation = 90,va='center',ha = 'left',rotation_mode='anchor')
                    #plt.annotate(r'$+$'+str(span[count_param])+' '+str(units[count_param]),[2*count_param+1+0.5,-1.15],fontsize=15,rotation = 90,va='center',ha = 'left',rotation_mode='anchor')
                    #xticks += [2*count_param+1]
                    

                    # Iterate over spectral resolutions and wavelength ranges
                    count_snr = 0
                    for snr in sorted(self.snr):
                        # SNR annotation
                        #plt.annotate(r'$\mathrm{S/N}=$'+str(snr),[-0.9,0.32+count_snr*0.67],fontsize=15,rotation = 0,va='center',ha = 'left',rotation_mode='anchor')

                        count_res = 0
                        for res in [50]: #sorted(self.res):
                            # Important for plotting
                            offset = 1/3+count_snr*2/3+(count_res-1.5)/8

                            # Check if the wanted variables were retrieved for
                            Data_found = True
                            try:
                                # Link the parameters to the corresponding position in the data
                                keys = list(self.grid_results[model][wln][res][snr].params.keys())
                                ind = np.where(np.array(keys)==str(param))[0][0]

                                # Load the data and perform calculations if necessary
                                if units[count_param]=='dex':
                                    post = np.log10(self.grid_results[model][wln][res][snr].equal_weighted_post[:,ind])
                                    true = np.log10(self.grid_results[model][wln][res][snr].truths[ind])
                                else:
                                    post = self.grid_results[model][wln][res][snr].equal_weighted_post[:,ind]
                                    true = self.grid_results[model][wln][res][snr].truths[ind]
                                prior = self.grid_results[model][wln][res][snr].priors_range[ind]

                            # What to do If they were not retrieved for
                            except:
                                if param == 'A_Bond':
                                    # If not done already calculate the bond albedo from the data
                                    if not hasattr(self.grid_results[model][wln][res][snr], 'A_Bond_ret'):
                                        print()
                                        print('Bond Albedo Calculation for:')
                                        print(model,wln,res,snr)
                                        self.grid_results[model][wln][res][snr].Plot_Ret_Bond_Albedo(*bond_params[:4],A_Bond_true=bond_params[4],T_equ_true=bond_params[5],plot=False)
                                    post = self.grid_results[model][wln][res][snr].A_Bond_ret
                                    true = bond_params[4]

                                elif param in ['T0','T_eq','P_eq']:
                                    # If not done already calculate the PT profiles from the data
                                    if not hasattr(self.grid_results[model][wln][res][snr], 'pressure'):
                                        print()
                                        print('PT-Profile Calculation for:')
                                        print(model,wln,res,snr)
                                        self.grid_results[model][wln][res][snr].Calc_PT_Profiles(skip = 1)
                                    if param in ['T0','T_eq']:
                                        if self.grid_results[model][wln][res][snr].settings['clouds'] == 'opaque': #len(self.grid_results[model][wln][res][snr].cloud_vars) != 0:
                                            post = self.grid_results[model][wln][res][snr].temperature_cloud_top[:,0]
                                            true = bond_params[5]
                                        else:
                                            post = self.grid_results[model][wln][res][snr].temperature[:,-1]
                                            true = bond_params[5]
                                    else:
                                        if self.grid_results[model][wln][res][snr].settings['clouds'] == 'opaque':
                                            post = np.log10(self.grid_results[model][wln][res][snr].pressure_cloud_top[:,0])
                                            true = np.log10(bond_params[6])
                                        else:
                                            post = np.log10(self.grid_results[model][wln][res][snr].pressure[:,-1])
                                            true = np.log10(bond_params[6])
                                
                                # If none of the above cases the variable cannot be plotted
                                else:
                                    print('Error: Parameter '+ str(param) +' not defined in input file config.ini.')
                                    Data_found = False
                            
                            if Data_found:
                                plt.hist(post,bins = 20,density = true,histtype='step',label=r'$\mathrm{S/N}=$'+str(snr))
                            
                            
                            """
                            # Plot the data according to the quantiles
                            center = 0.5*q50/(span[count_param])+2*count_param+1
                            plt.plot([center-0.5*(q50-q84)/(span[count_param]),center-0.5*(q50-q16)/(span[count_param])],[offset,offset],ls='-',color=colors[count_res],linewidth=lw)
                            plt.plot([center-0.5*(q50-q16)/(span[count_param]),center-0.5*(q50-q16)/(span[count_param])],[eb+offset,-eb+offset],ls='-',color=colors[count_res],linewidth=lw)
                            plt.plot([center-0.5*(q50-q84)/(span[count_param]),center-0.5*(q50-q84)/(span[count_param])],[eb+offset,-eb+offset],ls='-',color=colors[count_res],linewidth=lw)
                            plt.plot([center,center],[offset,offset],color=colors[count_res],marker=markers[count_snr],markersize = ms,markeredgecolor='black',markeredgewidth=lw/4)
                            """
                            """
                            # If wanted plot the prior distributions
                            if priors[count_param] == True:
                                offset = -1/3+3/16
                                center = 0.5*(prior[0]-true)/(span[count_param])+2*count_param+1
                                plt.plot([center+0.5*prior[1]/(span[count_param]),center+0.5*prior[1]/(span[count_param])],[eb+offset,-eb+offset],ls='-',color='black',linewidth=lw)
                                plt.plot([center-0.5*prior[1]/(span[count_param]),center-0.5*prior[1]/(span[count_param])],[eb+offset,-eb+offset],ls='-',color='black',linewidth=lw)
                                plt.plot([center+0.5*prior[1]/(span[count_param]),center-0.5*prior[1]/(span[count_param])],[offset,offset],ls='-',color='black',linewidth=lw)
                                plt.plot([center],[offset],color='black',marker='o',markersize = ms)
                            """
                            count_res += 1
                        count_snr += 1
                    count_param += 1
                    #if titles is None:
                    #    plt.xticks(ticks=xticks,labels=parameters,fontsize=20)
                    #else:
                    #    plt.xticks(ticks=xticks,labels=titles,fontsize=20)

                    # Final cosmetics
                    #lgd=plt.legend(['Markers:','SNR5','SNR10','SNR15','SNR20','','Line Colors:','R20','R35','R50','R100','Prior'],\
                    #    ['','S/N = 5','S/N = 10','S/N = 15','S/N = 20','','','R = 20','R = 35','R = 50','R = 100','Prior'],\
                    #    handler_map={str:  rp_hndl.Handles()},handlelength=4,ncol=2, bbox_to_anchor=(0.25, -0.65, 0.5, 0.5), borderaxespad=0.1,loc='center',fontsize=18)
                    #plt.xlim([-1,2*count_param+0.75])
                    #plt.ylim([-1.2,1/3+count_snr*2/3-1.5/8])

                    # Save the plot
                    ylim = plt.gca().get_ylim()
                    plt.plot([true,true],plt.gca().get_ylim(),'k--',label = 'Input')
                    plt.ylim(ylim)
                    plt.legend(loc = 'best')
                    if filename is None:
                        plt.savefig(self.results_directory+'/'+model+'/hist_'+str(param)+'_R'+str(res)+'_'+wln+'.pdf', bbox_inches='tight')
                    else:
                        plt.savefig(self.results_directory+'/'+model+'/'+str(filename)+'_'+wln+'.png',dpi=450,bbox_extra_artists=[lgd], bbox_inches='tight')


















    def Parameters_Hist_Top(self, parameters=['R_pl','M_pl','H2O','CO2','CO','H2SO484(c)_am'], ranges = [0.1,0.1,1,1,1,1], units=['$\mathrm{R_\oplus}$','dex','dex','dex','dex','dex','dex','dex','dex','','',''],bins = 100, titles = None, bond_params = [1,0.05,0.723,0.05*0.723,0.77,226,5e-2],filename=None):

        # Parameters for plotting
        ms = 8
        lw = 2
        eb = 0.025
        markers = ['o','s','D','v']
        colors = ['C3','C2','C0','C1']
        
        # Iterate over models and wavelength ranges
        for model in self.model:
            for wln in self.wln:

                # Iterate over the specified parameters
                count_param = 0
                for param in parameters:
                    # Iterate over spectral resolutions and wavelength ranges
                    m_hist = []
                    m_bins = []
                    for snr in sorted(self.snr):
                        # SNR annotation
                        for res in sorted(self.res): #sorted(self.res):
                            # Check if the wanted variables were retrieved for
                            Data_found = True
                            if True: # try:
                                # Link the parameters to the corresponding position in the data
                                keys = list(self.grid_results[model][wln][res][snr].params.keys())
                                ind = np.where(np.array(keys)==str(param))[0][0]

                                # Load the data and perform calculations if necessary
                                if units[count_param]=='dex':
                                    post = np.log10(self.grid_results[model][wln][res][snr].equal_weighted_post[:,ind])
                                    true = np.log10(self.grid_results[model][wln][res][snr].truths[ind])
                                else:
                                    post = self.grid_results[model][wln][res][snr].equal_weighted_post[:,ind]
                                    true = self.grid_results[model][wln][res][snr].truths[ind]

                                print(self.grid_results[model][wln][res][snr].priors[ind])
                                if self.grid_results[model][wln][res][snr].priors[ind] in ['G','LG']:
                                    prior_t = self.grid_results[model][wln][res][snr].priors_range[ind]
                                    prior = [prior_t[0]-3*prior_t[1],prior_t[0]+3*prior_t[1]]
                                else:
                                    prior = self.grid_results[model][wln][res][snr].priors_range[ind]
                                print(prior)

                            # What to do If they were not retrieved for
                            else: #except:
                                if param == 'A_Bond':
                                    # If not done already calculate the bond albedo from the data
                                    if not hasattr(self.grid_results[model][wln][res][snr], 'A_Bond_ret'):
                                        print()
                                        print('Bond Albedo Calculation for:')
                                        print(model,wln,res,snr)
                                        self.grid_results[model][wln][res][snr].Plot_Ret_Bond_Albedo(*bond_params[:4],A_Bond_true=bond_params[4],T_equ_true=bond_params[5],plot=False)
                                    post = self.grid_results[model][wln][res][snr].A_Bond_ret
                                    true = bond_params[4]
                                    prior = [0,1]

                                elif param in ['T0','T_eq','P_eq']:
                                    # If not done already calculate the PT profiles from the data
                                    if not hasattr(self.grid_results[model][wln][res][snr], 'pressure'):
                                        print()
                                        print('PT-Profile Calculation for:')
                                        print(model,wln,res,snr)
                                        self.grid_results[model][wln][res][snr].Calc_PT_Profiles(skip = 1)
                                    if param in ['T0','T_eq']:
                                        if self.grid_results[model][wln][res][snr].settings['clouds'] == 'opaque': #len(self.grid_results[model][wln][res][snr].cloud_vars) != 0:
                                            post = self.grid_results[model][wln][res][snr].temperature_cloud_top[:,0]
                                            true = bond_params[5]
                                        else:
                                            post = self.grid_results[model][wln][res][snr].temperature[:,-1]
                                            true = bond_params[5]
                                    else:
                                        if self.grid_results[model][wln][res][snr].settings['clouds'] == 'opaque':
                                            post = np.log10(self.grid_results[model][wln][res][snr].pressure_cloud_top[:,0])
                                            true = np.log10(bond_params[6])
                                        else:
                                            post = np.log10(self.grid_results[model][wln][res][snr].pressure[:,-1])
                                            true = np.log10(bond_params[6])
                                
                                # If none of the above cases the variable cannot be plotted
                                else:
                                    print('Error: Parameter '+ str(param) +' not defined in input file config.ini.')
                                    Data_found = False
                            
                            if Data_found:
                                m_hi, m_bi = np.histogram((post-prior[0])/(prior[1]-prior[0]),bins=[1/bins*ind for ind in range(bins+1)],density=True)
                                print(m_bi)
                                m_hist += [m_hi]
                                m_bins += [m_bi]


                    fig,ax = plt.subplots(4,1, figsize = (4,4))
                    plt.subplots_adjust(hspace=0.05)
                    
                    maxim = np.max(np.array(m_hist))
                    count = 0
                    count_snr = 0
                    for snr in sorted(self.snr):
                        count_res = 0
                        for res in sorted(self.res):
                            print(count)
                            #maxim = np.max(np.array(m_hist[count]))
                            for k in range(bins):
                                ax[len(self.snr)-1-count_snr].fill([m_bins[count][k],m_bins[count][k+1],m_bins[count][k+1],m_bins[count][k]],[count_res,count_res,count_res+1,count_res+1],color = 'k',alpha = (m_hist[count]/maxim)[k],lw=0)
                                ax[len(self.snr)-1-count_snr].plot([0,1],[count_res,count_res],'k-')
                            count_res += 1
                            count += 1

                        ax[len(self.snr)-1-count_snr].set_yticks([i+0.5 for i in range(count_res)])
                        ax[len(self.snr)-1-count_snr].set_yticklabels(['R '+str(i) for i in sorted(self.res)])
                        ax[len(self.snr)-1-count_snr].plot([(true-prior[0])/(prior[1]-prior[0]),(true-prior[0])/(prior[1]-prior[0])],[0,count_res],color = 'C3',ls = '-',lw = 1.5,label = r'Input')
                        ax[len(self.snr)-1-count_snr].plot([(true+ranges[count_param]-prior[0])/(prior[1]-prior[0]),(true+ranges[count_param]-prior[0])/(prior[1]-prior[0])],[0,count_res],color = 'C3',ls = ':',lw = 1.5,label = r'Input')
                        ax[len(self.snr)-1-count_snr].plot([(true-ranges[count_param]-prior[0])/(prior[1]-prior[0]),(true-ranges[count_param]-prior[0])/(prior[1]-prior[0])],[0,count_res],color = 'C3',ls = ':',lw = 1.5,label = r'Input')
                        #ax[i].plot([0.3,0.3],[0,count_res],color = 'C3',ls = ':',lw = 1.5)
                        #ax[i].plot([0.7,0.7],[0,count_res],color = 'C3',ls = ':',lw = 1.5,label = r'$\pm1$ dex')
                        ax[len(self.snr)-1-count_snr].plot([0,1],[count_res,count_res],'k-')
                        ax[len(self.snr)-1-count_snr].set_xlim([0,1])
                        ax[len(self.snr)-1-count_snr].set_ylim([0,count_res])
                        ax[len(self.snr)-1-count_snr].set_xticks([])
                        ax[len(self.snr)-1-count_snr].set_ylabel('S/N '+str(snr)+'\n')

                        count_snr += 1

                    if filename is None:
                        plt.savefig(self.results_directory+'/'+model+'/hist_sc_top_'+str(param)+'_'+wln+'.pdf', bbox_inches='tight')
                    else:
                        plt.savefig(self.results_directory+'/'+model+'/'+str(filename)+'_'+wln+'.png',dpi=450,bbox_extra_artists=[lgd], bbox_inches='tight')
                    count_param += 1


























    def Model_Comparison(self):
        
        # Iterate over models and wavelength ranges
        for model in self.model:
            for model_compare in self.model:
                if not model == model_compare:
                    print(str(model)+' vs '+str(model_compare))
                    # Define a matrix to stare the Bayes factors             
                    Mat = np.zeros(((len(self.res)+1)*((len(self.wln)+1)//2)-1,(len(self.snr)+1)*((len(self.wln)+1)//2)-1))-100

                    count_wln = 0
                    for wln in sorted(self.wln):
                        print(wln)
                        try:        
                            # Iterate over spectral resolutions and wavelength ranges
                            count_snr = 0
                            for snr in sorted(self.snr):
                                count_res = 0
                                for res in sorted(self.res):
                                    print(snr, res)
                                    # Load the data
                                    post = self.grid_results[model][wln][res][snr].evidence
                                    post_compare = self.grid_results[model_compare][wln][res][snr].evidence

                                    # Calculate the bayes factor and store the result in the matrix
                                    K = 0.4342944819*(float(post_compare[0])-float(post[0]))
                                    print(5*(count_wln//2)+count_res,5*(count_wln%2)+count_snr,K)
                                    Mat[5*(count_wln//2)+count_res,5*(count_wln%2)+count_snr] = K
                                    count_res += 1
                                count_snr += 1
                            count_wln += 1
                        except:
                            pass

                    # Define the color map according to jeffre's scale
                    cmap = col.ListedColormap(['azure','#2ca02c','#2ca02c80','#2ca02c60','#2ca02c40','#d6272840','#d6272860','#d6272880','#d62728'])
                    bounds=[-1000,-99,-2,-1,-0.5,0.0,0.5,1,2,99]
                    norm = col.BoundaryNorm(bounds, cmap.N)

                    # Initial plot configuration
                    fig,ax = plt.subplots(1)
                    fig.patch.set_facecolor('azure')
                    ax.axis('off')

                    # Plot the matrix
                    ax.matshow(Mat,cmap=cmap, norm=norm)
                    ax.vlines([-0.5+i for i in range(np.shape(Mat)[1]+1)],-0.5,np.shape(Mat)[0]-0.5,'azure',alpha=1,lw=2)
                    ax.hlines([-0.5+i for i in range(np.shape(Mat)[0]+1)],-0.5,np.shape(Mat)[0]-0.5,'azure',alpha=1,lw=2)

                    x_ax=[1.5,6.5]
                    y_ax=[1.5,6.5]
                
                    for pos in range(len(x_ax)):
                        for i in range(len(self.res)):
                            ax.text(-0.6, x_ax[pos]-(len(self.res)-1)/2.0-0.05+i,str(sorted(self.res)[i]),rotation = 0,rotation_mode='default',ha='right',va='center',fontsize=7)
                        ax.text(-1.7, x_ax[pos],'R',rotation = 90,rotation_mode='default',ha='center',va='center',fontsize=8)
                    
                    for pos in range(len(y_ax)):
                        for i in range(len(self.snr)):
                            ax.text(y_ax[pos]-(len(self.snr)-1)/2.0+i,-0.7,str(sorted(self.snr)[i]),rotation = 0,rotation_mode='default',ha='center',va='center',fontsize=7)
                        ax.text(y_ax[pos],-1.5,'S/N',rotation = 0,rotation_mode='default',ha='center',va='center',fontsize=8)

                    for i in range(len(self.wln)):
                        print(sorted(self.wln))
                        ax.text(y_ax[i%2]+2,x_ax[i//2]+2.15,sorted(self.wln)[i],rotation = 0,rotation_mode='default',ha='right',va='center',fontsize=8)
                    
                    for a in range(np.shape(Mat)[0]):
                        for b in range(np.shape(Mat)[1]):
                            if Mat[a,b]==-100:
                                ax.annotate('',[b,a+0.05],ha='center', va='center',color='white',fontweight = 'bold',fontsize=10)
                            else:
                                ax.annotate(str(np.round(Mat[a,b],1)),[b,a+0.05],ha='center', va='center',color='white',fontweight = 'bold',fontsize=7)

                    #plt.savefig('Results/Abundances/Summary/'+Wlen_grid[i]+'.png',dpi=450,bbox_inches='tight', facecolor='azure')
                    title = plt.title('Bayes factor for model '+str(model_compare)+'.', y=1.15)
                    plt.savefig(self.results_directory+'/'+model+'/Baye_'+model+'-'+model_compare+'.png',dpi=450, bbox_inches='tight', facecolor='azure')

    def Model_Comparison_Wlen(self):
        
        # Iterate over models and wavelength ranges
        for wln in sorted(self.wln):
            for model in self.model:
                # Define a matrix to store the Bayes factors
                print((len(self.res)+1)*((len(self.model))//2)-1,(len(self.snr)+1)*2-1)           
                Mat = np.zeros(((len(self.res)+1)*((len(self.model))//2)-1,(len(self.snr)+1)*2-1))-100
                
                count_model_compare = 0
                combinations = []
                for model_compare in self.model:
                    if not model == model_compare:
                        print(str(model)+' vs '+str(model_compare))
                        #try:        
                        # Iterate over spectral resolutions and wavelength ranges
                        count_snr = 0
                        for snr in sorted(self.snr):
                            count_res = 0
                            for res in sorted(self.res):
                                print(snr, res)
                                # Load the data
                                post = self.grid_results[model][wln][res][snr].evidence
                                post_compare = self.grid_results[model_compare][wln][res][snr].evidence

                                # Calculate the bayes factor and store the result in the matrix
                                K = 0.4342944819*(float(post_compare[0])-float(post[0]))
                                print(5*(count_model_compare//2)+count_res,5*(count_model_compare%2)+count_snr,K)
                                Mat[5*(count_model_compare//2)+count_res,5*(count_model_compare%2)+count_snr] = K
                                count_res += 1
                            count_snr += 1
                        count_model_compare += 1
                        combinations += [str(model)+' vs '+str(model_compare)]
                        #except:
                        #    pass
            

                # Define the color map according to jeffre's scale
                cmap = col.ListedColormap(['azure','#2ca02c','#2ca02c80','#2ca02c60','#2ca02c40','#d6272840','#d6272860','#d6272880','#d62728'])
                bounds=[-1000,-99,-2,-1,-0.5,0.0,0.5,1,2,99]
                norm = col.BoundaryNorm(bounds, cmap.N)

                # Initial plot configuration
                fig,ax = plt.subplots(1)
                fig.patch.set_facecolor('azure')
                ax.axis('off')

                # Plot the matrix
                ax.matshow(Mat,cmap=cmap, norm=norm)
                ax.vlines([-0.5+i for i in range(np.shape(Mat)[1]+1)],-0.5,np.shape(Mat)[0]-0.5,'azure',alpha=1,lw=2)
                ax.hlines([-0.5+i for i in range(np.shape(Mat)[0]+1)],-0.5,np.shape(Mat)[1]-0.5,'azure',alpha=1,lw=2)

                x_ax=[1.5,6.5,11.5]
                y_ax=[1.5,6.5]
                
                for pos in range(len(x_ax)):
                    for i in range(len(self.res)):
                        ax.text(-0.6, x_ax[pos]-(len(self.res)-1)/2.0-0.05+i,str(sorted(self.res)[i]),rotation = 0,rotation_mode='default',ha='right',va='center',fontsize=7)
                    ax.text(-1.7, x_ax[pos],'R',rotation = 90,rotation_mode='default',ha='center',va='center',fontsize=8)
                    
                for pos in range(len(y_ax)):
                    for i in range(len(self.snr)):
                        ax.text(y_ax[pos]-(len(self.snr)-1)/2.0+i,-0.7,str(sorted(self.snr)[i]),rotation = 0,rotation_mode='default',ha='center',va='center',fontsize=7)
                    ax.text(y_ax[pos],-1.5,'S/N',rotation = 0,rotation_mode='default',ha='center',va='center',fontsize=8)

                for i in range(len(self.model)-1):
                    print(sorted(self.model))
                    print(i,y_ax,i%2,x_ax,i//2,combinations)
                    ax.text(y_ax[i%2]+2,x_ax[i//2]+2.15,combinations[i],rotation = 0,rotation_mode='default',ha='right',va='center',fontsize=8)
                    
                for a in range(np.shape(Mat)[0]):
                    for b in range(np.shape(Mat)[1]):
                        if Mat[a,b]==-100:
                            ax.annotate('',[b,a+0.05],ha='center', va='center',color='white',fontweight = 'bold',fontsize=10)
                        else:
                            ax.annotate(str(np.round(Mat[a,b],1)),[b,a+0.05],ha='center', va='center',color='white',fontweight = 'bold',fontsize=7)

                #plt.savefig('Results/Abundances/Summary/'+Wlen_grid[i]+'.png',dpi=450,bbox_inches='tight', facecolor='azure')
                #title = plt.title('Bayes factor for model '+str(model_compare)+'.', y=1.15)
                plt.savefig(self.results_directory+'/'+model+'/Baye_'+str(wln)+'.png',dpi=450, bbox_inches='tight', facecolor='azure')





    def Model_Comparison_All(self):
        
        # Iterate over models and wavelength ranges
        for model in sorted(self.model):

            # Define a matrix to store the Bayes factors         
            Mat = np.zeros(((len(self.res)+1)*(len(self.model)-1)-1+3,(len(self.snr)+1)*len(self.wln)-1))-100

            
            combinations = []

            count_wln=0
            for wln in sorted(self.wln):
                count_model_compare = 0
                for model_compare in sorted(self.model):
                    if not model == model_compare:
                        print(str(model)+' vs '+str(model_compare))
                        #try:        
                        # Iterate over spectral resolutions and wavelength ranges
                        count_snr = 0
                        for snr in sorted(self.snr):
                            count_res = 0
                            for res in sorted(self.res):
                                print(snr, res)
                                # Load the data
                                post = self.grid_results[model][wln][res][snr].evidence
                                post_compare = self.grid_results[model_compare][wln][res][snr].evidence

                                # Calculate the bayes factor and store the result in the matrix
                                K = 0.4342944819*(float(post_compare[0])-float(post[0]))
                                print(5*(count_model_compare)+count_res,5*(count_model_compare%2)+count_snr,K)
                                Mat[5*(count_model_compare)+count_res,5*(count_wln)+count_snr] = K
                                count_res += 1
                            count_snr += 1
                        count_model_compare += 1
                        combinations += [str(model)+' vs\n'+str(model_compare)]
                count_wln += 1
            
            combinations = [r'Opaque $\mathrm{H_2SO_4}$ vs.'+'\nCloud-Free',r'Opaque $\mathrm{H_2SO_4}$ vs.'+'\nTransparent '+r'$\mathrm{H_2SO_4}$',r'Opaque $\mathrm{H_2SO_4}$ vs.'+'\nOpaque '+r'$\mathrm{H_2O}$']


            #Legend_Values
            Mat[-2,6]=-0.25
            Mat[-2,5]=-0.75
            Mat[-2,4]=-1.25
            Mat[-2,3]=-2.25

            Mat[-2,7]=0.25
            Mat[-2,8]=0.75
            Mat[-2,9]=1.25
            Mat[-2,10]=2.25

            # Define the color map according to jeffre's scale
            cmap = col.ListedColormap(['azure','#2ca02c','#2ca02c80','#2ca02c60','#2ca02c40','#d6272840','#d6272860','#d6272880','#d62728'])
            bounds=[-1000,-99,-2,-1,-0.5,0.0,0.5,1,2,99]
            norm = col.BoundaryNorm(bounds, cmap.N)

            # Initial plot configuration
            lw = 1
            fig,ax = plt.subplots(1,linewidth=2*lw, edgecolor="#007272")
            fig.patch.set_facecolor('azure')
            ax.axis('off')

            # Plot the matrix
            ax.matshow(Mat,cmap=cmap, norm=norm)
            ax.vlines([-0.5+i for i in range(np.shape(Mat)[1]+1)],-0.5,np.shape(Mat)[0]-0.5,'azure',alpha=1,lw=2)
            ax.vlines([13.55],-0.5,np.shape(Mat)[0]-0.5,'azure',alpha=1,lw=2)
            ax.hlines([-0.5+i for i in range(np.shape(Mat)[0]+1)],-0.5,np.shape(Mat)[1]-0.5,'azure',alpha=1,lw=2)

            x_ax=[1.5,6.5,11.5]
            y_ax=[1.5,6.5,11.5]
                
            for pos in range(len(x_ax)):
                for i in range(len(self.res)):
                    ax.text(-0.6, x_ax[pos]-(len(self.res)-1)/2.0+i,str(sorted(self.res)[i]),rotation = 0,rotation_mode='default',ha='right',va='center',fontsize=6,color='#007272')
                ax.text(-1.75, x_ax[pos],'R',rotation = 90,rotation_mode='default',ha='center',va='center',fontsize=6,color='#007272')
                ax.text(-2.8, x_ax[pos],combinations[pos],rotation = 90,rotation_mode='default',ha='center',va='center',fontsize=6,color='#007272')
                    
            for pos in range(len(y_ax)):
                for i in range(len(self.snr)):
                    ax.text(y_ax[pos]-(len(self.snr)-1)/2.0+i,-0.7,str(sorted(self.snr)[i]),rotation = 0,rotation_mode='default',ha='center',va='center',fontsize=6,color='#007272')
                ax.text(y_ax[pos],-1.4,'S/N',rotation = 0,rotation_mode='default',ha='center',va='center',fontsize=6,color='#007272')
                ax.text(y_ax[pos],-2.2,str(sorted(self.wln)[pos])+r' $\mu$m',rotation = 0,rotation_mode='default',ha='center',va='center',fontsize=6,color='#007272')
                    
            for a in range(np.shape(Mat)[0]-3):
                for b in range(np.shape(Mat)[1]):
                    if Mat[a,b]==-100:
                        ax.annotate('',[b,a+0.05],ha='center', va='center',color='white',fontweight = 'bold',fontsize=10)
                    else:
                        ax.annotate(str(np.round(Mat[a,b],1)),[b,a+0.05],ha='center', va='center',color='white',fontweight = 'bold',fontsize=5)

            
            ax.vlines([3.5+i for i in range(np.shape(Mat)[1]-7)],np.shape(Mat)[0]-2.5,np.shape(Mat)[0]-1.5,color='#007272',alpha=1,lw=lw)
            ax.vlines([6.5],np.shape(Mat)[0]-2,np.shape(Mat)[0]-1,color='#007272',lw=lw)
            ax.plot([2.72,6.365],[np.shape(Mat)[0]-1.25,np.shape(Mat)[0]-1.25],color='#007272',ls='-',lw=lw)
            ax.plot([2.9,2.7,2.9],[np.shape(Mat)[0]-1.15,np.shape(Mat)[0]-1.25,np.shape(Mat)[0]-1.35],color='#007272',ls='-',lw=lw)
            ax.plot([6.63,10.28],[np.shape(Mat)[0]-1.25,np.shape(Mat)[0]-1.25],color='#007272',ls='-',lw=lw)
            ax.plot([10.1,10.3,10.1],[np.shape(Mat)[0]-1.15,np.shape(Mat)[0]-1.25,np.shape(Mat)[0]-1.35],color='#007272',ls='-',lw=lw)

            for i in range(101):
                ax.fill([6.45-i/200,6.45-(i+1)/200,6.45-(i+1)/200,6.45-i/200],[np.shape(Mat)[0]-1.15,np.shape(Mat)[0]-1.15,np.shape(Mat)[0]-1.35,np.shape(Mat)[0]-1.35],'azure',alpha=1-i/100,lw=0,zorder=100)
                ax.fill([6.55+i/200,6.55+(i+1)/200,6.55+(i+1)/200,6.55+i/200],[np.shape(Mat)[0]-1.15,np.shape(Mat)[0]-1.15,np.shape(Mat)[0]-1.35,np.shape(Mat)[0]-1.35],'azure',alpha=1-i/100,lw=0,zorder=100)
                



            ax.text(3.5,np.shape(Mat)[0]-2.6,'-2.0',rotation = 90,rotation_mode='default',ha='center',va='bottom',fontsize=5,color='#007272')
            ax.text(4.5,np.shape(Mat)[0]-2.6,'-1.0',rotation = 90,rotation_mode='default',ha='center',va='bottom',fontsize=5,color='#007272')
            ax.text(5.5,np.shape(Mat)[0]-2.6,'-0.5',rotation = 90,rotation_mode='default',ha='center',va='bottom',fontsize=5,color='#007272')
            ax.text(6.5,np.shape(Mat)[0]-2.6,'0.0',rotation = 90,rotation_mode='default',ha='center',va='bottom',fontsize=5,color='#007272')
            ax.text(7.5,np.shape(Mat)[0]-2.6,'0.5',rotation = 90,rotation_mode='default',ha='center',va='bottom',fontsize=5,color='#007272')
            ax.text(8.5,np.shape(Mat)[0]-2.6,'1.0',rotation = 90,rotation_mode='default',ha='center',va='bottom',fontsize=5,color='#007272')
            ax.text(9.5,np.shape(Mat)[0]-2.6,'2.0',rotation = 90,rotation_mode='default',ha='center',va='bottom',fontsize=5,color='#007272')

            ax.text(2.3,np.shape(Mat)[0]-2,'Color Coding\n of '+r'$\mathrm{log_{10}(K)}$',rotation = 0,rotation_mode='default',ha='right',va='center',fontsize=6,color='#007272')
            ax.text(6,np.shape(Mat)[0]-0.9,r'Opaque $\mathrm{H_2SO_4}$',rotation = 0,rotation_mode='default',ha='right',va='center',fontsize=5,color='#007272')
            ax.text(7,np.shape(Mat)[0]-0.9,r'Other Models',rotation = 0,rotation_mode='default',ha='left',va='center',fontsize=5,color='#007272')
            #plt.savefig('Results/Abundances/Summary/'+Wlen_grid[i]+'.png',dpi=450,bbox_inches='tight', facecolor='azure')
            #title = plt.title('Bayes factor for model '+str(model_compare)+'.', y=1.15)
            plt.savefig(self.results_directory+'/'+model+'/Baye_Total.pdf', bbox_inches='tight', facecolor='azure', edgecolor=fig.get_edgecolor())










    def Species_Grid(self,parameters=['H2O','CO2','CO','H2SO484(c)'],units=['$\mathrm{R_\oplus}$','dex','K',''],span = [0.1,0.5,10,0.1],titles = None):
        
        # Iterate over models and wavelength ranges
        for model in self.model:
            for wln in self.wln:
                plt.figure(figsize = (10,6))
                plt.yticks([],fontsize=20,rotation = 90)
                xticks = []

                count_param = 0
                for param in parameters:
                    #Plot the background
                    plt.plot([2*count_param+1,2*count_param+1],[-2,3],'-k',alpha=1)
                    plt.fill_betweenx([-2,3],2*count_param+1-0.5,2*count_param+1+0.5,color='darkcyan',alpha=0.1,lw=0)
                    plt.annotate(r'$-$'+str(span[count_param])+' '+str(units[count_param]),[2*count_param+1-0.5,-1.15],fontsize=15,rotation = 90,va='center',ha = 'left',rotation_mode='anchor')
                    plt.annotate(r'$+$'+str(span[count_param])+' '+str(units[count_param]),[2*count_param+1+0.5,-1.15],fontsize=15,rotation = 90,va='center',ha = 'left',rotation_mode='anchor')
                    xticks += [2*count_param+1]
                    

                    # Iterate over spectral resolutions and wavelength ranges
                    count_snr = 0
                    for snr in sorted(self.snr):
                        plt.annotate(r'$\mathrm{S/N}=$'+str(snr),[-0.9,0.32+count_snr*0.67],fontsize=15,rotation = 0,va='center',ha = 'left',rotation_mode='anchor')

                        count_res = 0
                        for res in sorted(self.res):
                           

                            keys = list(self.grid_results[model][wln][res][snr].params.keys())

                            post = self.grid_results[model][wln][res][snr].equal_weighted_post
                            true = self.grid_results[model][wln][res][snr].truths
                            post = self.grid_results[model][wln][res][snr].evidence

                            print()
                            print(model,res,snr)
                            print()

                            count_res += 1
                        count_snr += 1
                    count_param += 1
                plt.xticks(ticks=xticks,labels=parameters,fontsize=20)
                plt.xlim([-1,8.75])
    
    
    def Constraint_Analysis(self,parameters=['H2O','CO2','H2SO484(c)_am','CO','R_pl','M_pl'],relative = None, units=['dex','dex','dex','dex','$\mathrm{R_\oplus}$','$\mathrm{M_\oplus}$'], span = [0.5,0.5,0.5,0.5,0.2,0.2], titles = None,Group = True,limits = None,plot_classification=True,p0_SSG=[8,6,0.007,0.5,0.5],s_max=2,s_ssg_max=5):

        # Parameters for plotting
        ms = 8
        lw = 2
        eb = 0.025
        markers = ['o','s','D','v']
        colors = ['C3','C2','C0','C1']

        # Iterate over models and wavelength ranges and fit the posterior models
        for model in self.model:
            for wln in self.wln:
                for snr in sorted(self.snr):
                    for res in sorted(self.res):
                        self.grid_results[model][wln][res][snr].Posterior_Classification(parameters=parameters,relative=relative,limits = None,plot_classification=plot_classification,p0_SSG=p0_SSG,s_max=s_max,s_ssg_max=s_ssg_max)
                
                # Start of the plotting
                if Group == True:
                    plt.figure(figsize = (10,15))
                    plt.yticks(ticks=[5,3,1],labels=list(parameters[:3]),rotation=90,fontsize=20,rotation_mode='default',va = 'center')
                    pass

                count_param = 0
                for param in parameters[:3]:
                    keys = list(self.grid_results[model][wln][res][snr].params.keys())
                    ind = np.where(np.array(keys)==str(param))[0][0]
                    truth = np.log10(self.grid_results[model][wln][res][snr].truths[ind])

                    plt.xticks([-7,-6,-5,-4,-3,-2,-1,0],fontsize=20,rotation = 90)
                    plt.ylabel('Atmospheric Species',fontsize=20,labelpad =25)
                    if relative is None:
                        plt.xlabel(r'Abundance $\mathrm{log_{10}}$',fontsize=20,labelpad =20)
                    else:
                        plt.xlabel(r'Abundance relative to ' + str(relative) + r'$\mathrm{log_{10}}$',fontsize=20,labelpad =20)



                    plt.plot([-7,0],[2+2*(2-count_param),2+2*(2-count_param)],'-k',alpha=1)
                    plt.plot([truth,truth],[2*(2-count_param),2+2*(2-count_param)],'k-')
                    plt.fill_betweenx([2*(2-count_param),2+2*(2-count_param)],truth-0.5,truth+0.5,color='darkcyan',alpha=0.1,lw=0)
                    plt.fill_betweenx([2*(2-count_param),2+2*(2-count_param)],truth-1.0,truth+1.0,color='darkcyan',alpha=0.1,lw=0)
                    plt.fill_betweenx([2*(2-count_param),2+2*(2-count_param)],truth-1.5,truth+1.5,color='darkcyan',alpha=0.1,lw=0)
                    
                    count_snr = 0
                    for snr in sorted(self.snr):
                        count_res = 0
                        for res in sorted(self.res):
                            try:
                                Offset = 1/4+count_snr*1/2+(count_res-1.5)/10
                                print(self.grid_results[model][wln][res][snr].best_post_model[param])
                                self.Limit_Plot(count_param,2,self.grid_results[model][wln][res][snr].best_post_model[param],Offset,ms,lw,eb,colors[count_res],markers[count_snr])
                            except:
                                pass
                            count_res +=1
                        count_snr += 1
                    count_param += 1
                plt.title('Wavelength Range: '+wln, fontsize = 20)
                plt.show()




    def Limit_Plot(self,n,N,model,SNR_O,ms,lw,eb,c,m,centre =-3.5 ,un_c_len=2):
        x=np.linspace(-15,15,1000)     
        print(model)

        if model[0] == 'F':
            plt.plot([centre,centre-un_c_len],[2*(N-n)+SNR_O,2*(N-n)+SNR_O],ls=':',color=c,linewidth=lw)
            plt.plot([centre-un_c_len+3*eb,centre-un_c_len,centre-un_c_len+3*eb],[2*(N-n)+eb+SNR_O,2*(N-n)+SNR_O,2*(N-n)-eb+SNR_O],ls='-',color=c,linewidth=lw)
            plt.plot([centre+un_c_len,centre],[2*(N-n)+SNR_O,2*(N-n)+SNR_O],ls=':',color=c,linewidth=lw)
            plt.plot([centre+un_c_len-3*eb,centre+un_c_len,centre+un_c_len-3*eb],[2*(N-n)+eb+SNR_O,2*(N-n)+SNR_O,2*(N-n)-eb+SNR_O],ls='-',color=c,linewidth=lw)
            plt.plot(centre,2*(N-n)+SNR_O,color=c,marker = m,markersize = ms,markeredgecolor='black',markeredgewidth=lw/4)

        if model[0] == 'SS':
            HM = -model[1][1]/model[1][0]
            s = r_post.Inv_Model_SoftStep(0.16*model[1][-1],model[1][0],model[1][1],model[1][2])
            plt.plot([HM,HM-un_c_len],[2*(N-n)+SNR_O ,2*(N-n)+SNR_O ],ls='-',color=c,linewidth=lw)
            plt.plot([HM-un_c_len+3*eb,HM-un_c_len,HM-un_c_len+3*eb],[2*(N-n)+eb+SNR_O ,2*(N-n)+SNR_O ,2*(N-n)-eb+SNR_O],ls='-',color=c,linewidth=lw)
            plt.plot([HM,s],[2*(N-n)+SNR_O ,2*(N-n)+SNR_O],ls='-',color=c,linewidth=lw)
            plt.plot([s,s],[2*(N-n)+eb+SNR_O,2*(N-n)-eb+SNR_O],ls='-',color=c,linewidth=lw)
            plt.plot(HM,2*(N-n)+SNR_O ,color=c,marker = m,markersize = ms,markeredgecolor='black',markeredgewidth=lw/4)

        if model[0] == 'SSG':
            line = r_post.Model_SoftStepG(x,*list(model[1]))
            ind = np.argmax(line)
            x_up = x[np.where(x>x[ind])]
            xp = x_up[np.argmin(np.abs(r_post.Model_SoftStepG(x_up,*model[1])-1/2*np.max(line)))]
            x_down = x[np.where(x<x[ind])]
            xm = x_down[np.argmin(np.abs(r_post.Model_SoftStepG(x_down,*model[1])-(np.max(line)/2+model[1][2]/2)))]
            plt.plot([xp,xm],[2*(N-n)+SNR_O,2*(N-n)+SNR_O],ls='-',color=c,linewidth=lw)
            plt.plot([xp,xp],[2*(N-n)+SNR_O-eb,2*(N-n)+SNR_O+eb],ls='-',color=c,linewidth=lw)
            plt.plot([xm,xm],[2*(N-n)+SNR_O-eb,2*(N-n)+SNR_O+eb],ls='-',color=c,linewidth=lw)
            plt.plot([xm,xm-un_c_len],[2*(N-n)+SNR_O,2*(N-n)+SNR_O],ls=':',color=c,linewidth=lw)
            plt.plot([xm-un_c_len+3*eb,xm-un_c_len,xm-un_c_len+3*eb],[2*(N-n)+SNR_O-eb,2*(N-n)+SNR_O,2*(N-n)+SNR_O+eb],ls='-',color=c,linewidth=lw)
            plt.plot(x[ind],2*(N-n)+SNR_O,color=c,marker = m,markersize = ms,markeredgecolor='black',markeredgewidth=lw/4)

        if model[0] == 'G':
            s=model[1][-1]
            mean=model[1][-2]
            q50 = mean #np.quantile(data,0.5,axis=0)
            q84 = mean+s #np.quantile(data,0.84,axis=0)
            q16 = mean-s #np.quantile(data,0.16,axis=0)
            plt.plot([q84,q16],[2*(N-n)+SNR_O,2*(N-n)+SNR_O],ls='-',color=c,linewidth=lw)
            plt.plot([q16,q16],[2*(N-n)+eb+SNR_O,2*(N-n)-eb+SNR_O],ls='-',color=c,linewidth=lw)
            plt.plot([q84,q84],[2*(N-n)+eb+SNR_O,2*(N-n)-eb+SNR_O],ls='-',color=c,linewidth=lw)
            plt.plot([q50,q50],[2*(N-n)+SNR_O,2*(N-n)+SNR_O],color=c,marker = m,markersize = ms,markeredgecolor='black',markeredgewidth=lw/4)

        if model[0] == 'USS':
            HM = -model[1][1]/model[1][0]
            s = r_post.Inv_Model_SoftStep(0.16*model[1][-1],model[1][0],model[1][1],model[1][2])
            plt.plot([-HM,-HM+un_c_len],[2*(N-n)+SNR_O ,2*(N-n)+SNR_O ],ls='-',color=c,linewidth=lw)
            plt.plot([-HM+un_c_len+3*eb,-HM+un_c_len,-HM+un_c_len+3*eb],[2*(N-n)+eb+SNR_O ,2*(N-n)+SNR_O ,2*(N-n)-eb+SNR_O],ls='-',color=c,linewidth=lw)
            plt.plot([-HM,s],[2*(N-n)+SNR_O ,2*(N-n)+SNR_O],ls='-',color=c,linewidth=lw)
            plt.plot([s,s],[2*(N-n)+eb+SNR_O,2*(N-n)-eb+SNR_O],ls='-',color=c,linewidth=lw)
            plt.plot(-HM,2*(N-n)+SNR_O ,color=c,marker = m,markersize = ms,markeredgecolor='black',markeredgewidth=lw/4)
            



    """
    
    def Basic_Background(self,parameters=['R_pl','M_pl','T_eq','A_Bond'],units=['$\mathrm{R_\oplus}$','dex','K',''],span = [0.1,0.5,10,0.1],titles = None):
        
        # Iterate over models and wavelength ranges
        for model in self.model:
            for wln in self.wln:
                plt.figure(figsize = (10,6))
                plt.yticks([],fontsize=20,rotation = 90)
                xticks = []

                count_param = 0
                for param in parameters:
                    #Plot the background
                    plt.plot([2*count_param+1,2*count_param+1],[-2,3],'-k',alpha=1)
                    plt.fill_betweenx([-2,3],2*count_param+1-0.5,2*count_param+1+0.5,color='darkcyan',alpha=0.1,lw=0)
                    plt.annotate(r'$-$'+str(span[count_param])+' '+str(units[count_param]),[2*count_param+1-0.5,-1.15],fontsize=15,rotation = 90,va='center',ha = 'left',rotation_mode='anchor')
                    plt.annotate(r'$+$'+str(span[count_param])+' '+str(units[count_param]),[2*count_param+1+0.5,-1.15],fontsize=15,rotation = 90,va='center',ha = 'left',rotation_mode='anchor')
                    xticks += [2*count_param+1]
                    

                    # Iterate over spectral resolutions and wavelength ranges
                    count_snr = 0
                    for snr in sorted(self.snr):
                        plt.annotate(r'$\mathrm{S/N}=$'+str(snr),[-0.9,0.32+count_snr*0.67],fontsize=15,rotation = 0,va='center',ha = 'left',rotation_mode='anchor')

                        count_res = 0
                        for res in sorted(self.res):
                           

                            keys = list(self.grid_results[model][wln][res][snr].params.keys())

                            post = self.grid_results[model][wln][res][snr].equal_weighted_post
                            true = self.grid_results[model][wln][res][snr].truths
                            post = self.grid_results[model][wln][res][snr].evidence

                            print()
                            print(model,res,snr)
                            print()

                            count_res += 1
                        count_snr += 1
                    count_param += 1
                plt.xticks(ticks=xticks,labels=parameters,fontsize=20)
                plt.xlim([-1,8.75])

 
        
        
        markers = ['o','s','D','v']
        colors = ['C3','C2','C0','C1']
        eb=0.025

    """