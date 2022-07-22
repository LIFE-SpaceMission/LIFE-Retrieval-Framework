__author__ = "Konrad"
__copyright__ = "Copyright 2022, Konrad"
__maintainer__ = "Björn S. Konrad"
__email__ = "konradb@phys.ethz.ch"
__status__ = "Development"

# Standard Libraries
import itertools
import sys, os
from xml.etree.ElementInclude import XINCLUDE_INCLUDE
import matplotlib.colors as col
import matplotlib.pyplot as plt
import numpy as np
from pkg_resources import ResolutionError
from scipy.ndimage.filters import gaussian_filter1d
import pandas as pd
import matplotlib.patheffects as PathEffects

# Import additional external files
import retrieval_plotting as rp
from retrieval_support import retrieval_posteriors as r_post
from retrieval_plotting_support import retrieval_plotting_handlerbase as rp_hndl
from retrieval_plotting_support import retrieval_plotting_posteriors as rp_posteriors





class grid_plotting():

    def __init__(self, results_directory, plots_dir='',object_info=None):
        '''
        This function reads the input.ini file as well as the retrieval
        results files of imterest to us and stores the read in data
        in order to generate the retrieval plots of interest to us.
        '''

        self.grid_results = {}
        self.grid_results['item_classification'] = {}
        self.grid_results['rp_object'] = {}

        directories = results_directory
        self.results_directory = plots_dir
        if not os.path.exists(plots_dir+'Grid_Plots/'):
            os.makedirs(plots_dir+'Grid_Plots/')

        # Iterating over all directories in the grid results folders
        for model_item in directories:
            self.grid_results['item_classification'][model_item] = {}

            # Initiating instances of the retrieval plotting class and storing the data in a dict
            self.grid_results['rp_object'][model_item] = rp.retrieval_plotting(model_item)

            # If no object classification library is is provided provided automatically generate a class linrary
            if object_info is None:
                self.grid_results['item_classification'][model_item]['Wavelength'] = (self.grid_results['rp_object'][model_item].config_file.items('LAMBDA RANGE')[0][1]
                                                        + '-' + self.grid_results['rp_object'][model_item].config_file.items('LAMBDA RANGE')[1][1])
                self.grid_results['item_classification'][model_item]['R'] = [i[1] for i in self.grid_results['rp_object'][model_item].config_file.items('CHEMICAL COMPOSITION PARAMETERS') if i[0] == 'settings_resolution'][0]
                self.grid_results['item_classification'][model_item]['SN'] = [i[2:] for i in model_item.split('/')[-2].split('_') if 'SN' in i][0]
                self.grid_results['item_classification'][model_item]['Model'] = model_item.split('/')[-3]
            else:
                self.grid_results['item_classification'] = object_info.copy()

        # genereate lists of the different categories and subcategories of parameters
        self.categories = list(sorted(set(list(self.grid_results['item_classification'][model_item].keys()))))
        self.sub_categories = {}
        for category in self.categories:
            if category in ['R','SN']:
                self.sub_categories[category] = sorted(set([int(self.grid_results['item_classification'][i][category]) for i in self.grid_results['item_classification'].keys()]))
                self.sub_categories[category] = [str(i) for i in self.sub_categories[category]]
            else:
                self.sub_categories[category] = sorted(set([self.grid_results['item_classification'][i][category] for i in self.grid_results['item_classification'].keys()]))
        


    # Generates the grid structure for the plots
    def _grid_generator(self,plot_type,x_category=None,y_category=None,overplot_category=None,return_all=False):
        # Get lists with the categories for the x and the y axis
        local_categories = self.categories.copy()
        if x_category is None:
            x_ind = 0
            x_cat = ['']
        else:
            x_cat = self.sub_categories[x_category]
            local_categories = [i for i in local_categories if i != x_category]
        if y_category is None:
            y_ind = 0
            y_cat = ['']
        else:
            y_cat = self.sub_categories[y_category]
            local_categories = [i for i in local_categories if i != y_category]
        if overplot_category is None:
            o_ind = 0
            o_cat = ['']
        else:
            o_cat = self.sub_categories[overplot_category]
            local_categories = [i for i in local_categories if i != overplot_category]

        # Find the dimensions of the lists
        x_dim = len(x_cat)
        y_dim = len(y_cat)
        o_dim = len(o_cat)

        # Get the names of all Categories that are neither the x, y, or overplot category
        local_sub_categories = {}
        for i in local_categories:
            local_sub_categories[i] = self.sub_categories[i]
 
        # Get all combinations of Categories that are neither the x, y, or overplot category 
        keys, values = zip(*local_sub_categories.items())
        combinations_local_sub_categories = [dict(zip(keys, v)) for v in itertools.product(*values)]

        # Get lists of the runnames for each combination of category
        run_categorization = np.zeros((len(combinations_local_sub_categories),y_dim,x_dim,o_dim), dtype=object)
        for ind in range(len(combinations_local_sub_categories)):
            for run in self.grid_results['rp_object'].keys():
                equal = [self.grid_results['item_classification'][run][key]==combinations_local_sub_categories[ind][key] for key in combinations_local_sub_categories[ind].keys()]
                if all(equal):
                    if x_category is not None:
                        x_ind = [i for i in range(x_dim) if x_cat[i] == self.grid_results['item_classification'][run][x_category]][0]
                    if y_category is not None:
                        y_ind = [i for i in range(y_dim) if y_cat[i] == self.grid_results['item_classification'][run][y_category]][0]
                    if overplot_category is not None:
                        o_ind = [i for i in range(o_dim) if o_cat[i] == self.grid_results['item_classification'][run][overplot_category]][0]

                    # Save the runname at the correct position in the array
                    run_categorization[ind,y_ind,x_ind,o_ind] = run

        # Generate the directory to save the plots in
        if overplot_category is None:
            save_directory = self.results_directory+'Grid_Plots/'+plot_type+'/Fixed_'+'_'.join(local_categories)+'/'
        else:
            save_directory = self.results_directory+'Grid_Plots/'+plot_type+'/Fixed_'+'_'.join(local_categories)+'/Overplot_'+overplot_category+'/'

        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        if return_all:
            return save_directory, x_dim, x_cat, y_dim, y_cat, o_dim, o_cat, combinations_local_sub_categories, run_categorization
        else:    
            return save_directory, x_dim, y_dim, o_dim, combinations_local_sub_categories, run_categorization





    """
    #################################################################################
    #                                                                               #
    #   Routine for generating posterior plots.                                     #
    #                                                                               #
    #################################################################################
    """



    def _get_posterior_data(self,log_pressures,log_mass,log_abundances,log_particle_radii,plot_pt,plot_physparam,plot_chemcomp,plot_clouds,plot_bond=None,BB_fit_range=None):
        # Get all of the unique keys we want to plot the posteriors of
        chemcomp_params, cloud_params, phys_params, pt_params = [], [], [], []
        local_grid_results = {}

        for run in self.grid_results['rp_object'].keys():
            
            chemcomp_params += [[]]
            cloud_params += [[]]
            phys_params += [[]]
            pt_params += [[]]

            # classification of the parameters
            names = self.grid_results['rp_object'][run].params_names.copy()
            for i in names:
                if self.grid_results['rp_object'][run].params[names[i]]['type'] == 'CHEMICAL COMPOSITION PARAMETERS':
                    chemcomp_params[-1] = chemcomp_params[-1]+[i]
                if self.grid_results['rp_object'][run].params[names[i]]['type'] == 'CLOUD PARAMETERS':
                    cloud_params[-1] = cloud_params[-1]+[i]
                if self.grid_results['rp_object'][run].params[names[i]]['type'] == 'PHYSICAL PARAMETERS':
                    phys_params[-1] = phys_params[-1]+[i]
                if self.grid_results['rp_object'][run].params[names[i]]['type'] == 'TEMPERATURE PARAMETERS':
                    pt_params[-1] = pt_params[-1]+[i]
                else:
                    pass

            # Generate local copies of the posterior data 
            local_grid_results[run]={}
            local_grid_results[run]['local_equal_weighted_post'] = np.copy(self.grid_results['rp_object'][run].equal_weighted_post[:,:-1])
            local_grid_results[run]['local_truths'] = self.grid_results['rp_object'][run].truths.copy()
            local_grid_results[run]['local_titles'] = self.grid_results['rp_object'][run].titles.copy()
            local_grid_results[run]['local_posterior_keys'] = list(self.grid_results['rp_object'][run].params.keys())

            # Adust the local copy of the posteriors according to the users desires
            local_grid_results[run]['local_equal_weighted_post'], local_grid_results[run]['local_truths'], local_grid_results[run]['local_titles'] = \
                self.grid_results['rp_object'][run].Scale_Posteriors(local_grid_results[run]['local_equal_weighted_post'],local_grid_results[run]['local_truths'],
                                                                    local_grid_results[run]['local_titles'], log_pressures=log_pressures, log_mass=log_mass,
                                                                    log_abundances=log_abundances, log_particle_radii=log_particle_radii)

            # If wanted add the bond albedo and the equilibrium temperature to the plot
            if plot_bond is not None:
                A_Bond_true, T_equ_true = self.grid_results['rp_object'][run].Plot_Ret_Bond_Albedo(*plot_bond[:-2],A_Bond_true=plot_bond[-1],T_equ_true=plot_bond[-2],plot = False,fit_BB=BB_fit_range)
                local_grid_results[run]['local_equal_weighted_post'] = np.append(local_grid_results[run]['local_equal_weighted_post'], self.grid_results['rp_object'][run].ret_opaque_T,axis=1)
                local_grid_results[run]['local_equal_weighted_post'] = np.append(local_grid_results[run]['local_equal_weighted_post'], self.grid_results['rp_object'][run].A_Bond_ret,axis=1)
                local_grid_results[run]['local_truths'] += [T_equ_true,A_Bond_true]
                local_grid_results[run]['local_titles'] += [r'$\mathrm{T_{eq,\,Planet}}$',r'$\mathrm{A_{B,\,Planet}}$']
                local_grid_results[run]['local_posterior_keys'] += ['T_eq', 'A_B']

        # Get the unique posterior keys and return
        # Todo: add a routine if there is no maximal category  
        unique_posterior_keys = []
        if plot_pt:
            unique_posterior_keys += max(pt_params, key=len)
        if plot_physparam:
            unique_posterior_keys += max(phys_params, key=len)
        if plot_chemcomp:
            unique_posterior_keys += max(chemcomp_params, key=len)
        if plot_clouds:
            unique_posterior_keys += max(cloud_params, key=len)
        if plot_bond is not None:
            unique_posterior_keys += ['T_eq', 'A_B']

        return local_grid_results, unique_posterior_keys

        

    def posteriors_custom_grid(self,x_category=None,y_category=None,overplot_category=None,overplot_identifiers=None,posterior_identifiers=None,subfig_size=[2.5,3],sharex=True,sharey=True,
                    log_pressures=True, log_mass=True,log_abundances=True, log_particle_radii=True,colors=None,hist_settings=None,bins=20,true_profile=False,
                    plot_pt=True,plot_physparam=True,plot_chemcomp=True,plot_clouds=True,plot_bond=None,BB_fit_range=None,collapse = True):

        # Generate the grid of runs for plotting
        save_directory, x_dim, x_cat, y_dim, y_cat, o_dim, o_cat, combinations_local_sub_categories, run_categorization = \
                                    self._grid_generator('Posteriors',x_category=x_category,y_category=y_category,overplot_category=overplot_category,return_all=True)
        
        if collapse:
            run_categorization_collapsed = np.zeros((len(combinations_local_sub_categories),y_dim+x_dim,1,o_dim), dtype=object)
            for p in range(len(combinations_local_sub_categories)):
                for i in range(x_dim):
                    run_categorization_collapsed[p,i*y_dim:(i+1)*y_dim,0,:] = run_categorization[p,:,i,:]
            y_dim = x_dim+y_dim
            x_dim = 1
            run_categorization = run_categorization_collapsed.copy()


        # Get all of the unique keys as well as local copies of the posteriors
        local_grid_results,unique_posterior_keys = self._get_posterior_data(log_pressures,log_mass,log_abundances,log_particle_radii,
                                                                            plot_pt,plot_physparam,plot_chemcomp,plot_clouds,plot_bond=plot_bond,BB_fit_range=BB_fit_range)

        # Loop over all of the unique retieved variables             
        for key in unique_posterior_keys:
            print('Plotting the results for: '+ key)
            self._posterior_plotting(combinations_local_sub_categories,run_categorization,local_grid_results,save_directory,
                                    x_dim,x_cat,y_dim,y_cat,o_dim,o_cat,unique_posterior_keys,colors,hist_settings,
                                    overplot_identifiers,posterior_identifiers,bins,subfig_size,true_profile,key_in=key,sharex=sharex,sharey=sharey)
            print('Done plotting the posteriors for: '+key)



    def posteriors_grid(self,x_size = 1,overplot_category=None,overplot_identifiers=None,posterior_identifiers=None,subfig_size=[2.5,3],
                    log_pressures=True, log_mass=True,log_abundances=True, log_particle_radii=True,colors=None,hist_settings=None,bins=20,true_profile=False,
                    plot_pt=True,plot_physparam=True,plot_chemcomp=True,plot_clouds=True,plot_bond=None,BB_fit_range=None):

        # Secify what posteriors are in the grid:
        plotted_posts = ''
        if plot_pt:
            plotted_posts += '_pt'
        if plot_physparam:
            plotted_posts += '_phsparam'
        if plot_chemcomp:
            plotted_posts += '_abund'
        if plot_clouds:
            plotted_posts += '_clouds'
        if plot_bond is not None:
            plotted_posts += '_albedo'

        # Generate the grid of runs for plotting
        save_directory, x_dim, x_cat, y_dim, y_cat, o_dim, o_cat, combinations_local_sub_categories, run_categorization = \
                                    self._grid_generator('Posteriors',overplot_category=overplot_category,return_all=True)

        # Get all of the unique keys as well as local copies of the posteriors
        local_grid_results,unique_posterior_keys = self._get_posterior_data(log_pressures,log_mass,log_abundances,log_particle_radii,
                                                                            plot_pt,plot_physparam,plot_chemcomp,plot_clouds,plot_bond=plot_bond,BB_fit_range=BB_fit_range)
        x_dim = x_size
        y_dim = len(unique_posterior_keys)//x_size+1

        # Create the grid plot
        print('Plotting the posterior grids.')
        self._posterior_plotting(combinations_local_sub_categories,run_categorization,local_grid_results,save_directory,
                                    x_dim,x_cat,y_dim,y_cat,o_dim,o_cat,unique_posterior_keys,colors,hist_settings,
                                    overplot_identifiers,posterior_identifiers,bins,subfig_size,true_profile,plotted_posts=plotted_posts)
        print('Done plotting the posterior grids.')



    def _posterior_plotting(self,combinations_local_sub_categories,run_categorization,local_grid_results,save_directory,
                                    x_dim,x_cat,y_dim,y_cat,o_dim,o_cat,unique_posterior_keys,colors,hist_settings,
                                    overplot_identifiers,posterior_identifiers,bins,subfig_size,true_profile,key_in=None,plotted_posts='',sharex=False,sharey=False):

        # Increase the recurion limit
        sys.setrecursionlimit(10**9)

        # we use this as marker for which case we are in
        # posteriors_custom_grid: passes a key
        # posteriors_grid: does not
        key = key_in

        # Loop over the various plots to do
        for ind in range(len(combinations_local_sub_categories)):
            case = combinations_local_sub_categories[ind]

            # Initialize a new figure for the plot
            fig,ax = plt.subplots(y_dim,x_dim,figsize = (x_dim*subfig_size[0],y_dim*subfig_size[1]),
                            sharex=sharex,sharey=sharey,squeeze=False)
            if sharex:
                plt.subplots_adjust(hspace=0,wspace=0)
            else:
                plt.subplots_adjust(hspace=0.3,wspace=0.1)
            legend_plotted = False

            xlim = [1e100,-1e100]
            ylim = [0,0]
            # Loop over the x and y axis of the plots
            for y in range(y_dim):
                for x in range(x_dim):
                    if (key_in is not None) or ((y*x_dim+x)<len(unique_posterior_keys)):
                        if key_in is None:
                            # Define the key if none was provided
                            key = unique_posterior_keys[y*x_dim+x]
                            xlim = [1e100,-1e100]
                            ylim = [0,0]
                        for o in range(o_dim):
                            if key_in is None:
                                run = run_categorization[ind,0,0,o]
                            else:
                                run = run_categorization[ind,y,x,o]

                            for post in range(len(local_grid_results[run]['local_posterior_keys'])):
                                # If we are at the correct post index
                                if key == (list(self.grid_results['rp_object'][run].params_names.keys())+['T_eq', 'A_B'])[post]:

                                    # Choose the correct color and hatches
                                    if colors is None:
                                        color = 'k'
                                    else:
                                        color = colors[[i for i in colors.keys() if i in run][0]]
                                    if hist_settings is None:
                                        hist_setting = {'histtype':'stepfilled'}
                                    else:
                                        hist_setting = hist_settings[[i for i in hist_settings.keys() if i in run][0]]
                                    if overplot_identifiers is None:
                                        overplot_identifier = o_cat[o]
                                    else:
                                        overplot_identifier = overplot_identifiers[[i for i in overplot_identifiers.keys() if i in run][0]]
                                    if posterior_identifiers is None:
                                        posterior_identifier = key
                                    else:
                                        posterior_identifier = posterior_identifiers[[i for i in posterior_identifiers.keys() if i in run][0]][post]

                                    # Plot the posterior histogram
                                    if xlim[0] != 1e100:
                                        h = ax[y,x].hist(local_grid_results[run]['local_equal_weighted_post'][:,post],range=xlim,color=color,density=True,bins=bins,**hist_setting,label=overplot_identifier)
                                    else:
                                        h = ax[y,x].hist(local_grid_results[run]['local_equal_weighted_post'][:,post],color=color,density=True,bins=bins,**hist_setting,label=overplot_identifier)

                                    # Update the limits for the plots
                                    xlim = [min(h[1][0],xlim[0]),max(h[1][-1],xlim[1])]
                                    ylim = [0,max(1.3*np.max(h[0]),ylim[1])]
                                    run_last = run
                                    post_last = post

                        # Plot the truth or the profiles for the abundances
                        profiles=''
                        if true_profile:
                            df = pd.read_csv(self.grid_results['rp_object'][run_last].settings['input_profile'])
                            if key in df.columns:
                                ax2=ax[x,y].twinx()
                                ax2.semilogy(np.log10(df[key]),df['P(bar)'],color='k',ls = '--',label = 'Truth')
                                ax2.set_ylim(df['P(bar)'].max(),df['P(bar)'].min())
                                ax2.set_ylabel('Pressure (bar)')
                                profiles='_profile'
                            else:
                                ax[x,y].vlines(local_grid_results[run_last]['local_truths'][post_last],ylim[0],ylim[1],color='k',ls = '--',label = 'Input')

                        # Plot the truth
                        ax[y,x].vlines(local_grid_results[run_last]['local_truths'][post_last],-10000,10000,color='k',ls = '--',label = 'Truth',zorder = 4)

                        # Labels, Limits and legends
                        ax[y,x].set_xlim(xlim)
                        ax[y,x].set_ylim(ylim)
                        ax[y,x].set_yticks([])

                        if (not sharex) or (y == y_dim-1):
                            ax[y,x].set_xlabel(posterior_identifier)
                        else:
                            ax[y,x].set_yticks([])


                        #if key_in is not None:
                        #    ax[y,x].legend(frameon = False,loc='upper left')
                    else:
                        ax[y,x].axis('off')
                        if (y*x_dim+x)==len(unique_posterior_keys):
                            handles,labels = ax[1,0].get_legend_handles_labels()
                            ax[y,x].legend(handles,labels, frameon = False,loc='center')
                            legend_plotted=True

            if not legend_plotted:
                handles,labels = ax[0,0].get_legend_handles_labels()

            # Check if directory for saving exists
            title = '_'.join([list(case.keys())[i]+list(case.values())[i]for i in range(len(case))])
            if not os.path.exists(save_directory+'/'+title+'/'):
                os.makedirs(save_directory+'/'+title+'/')

            # Save the plots
            if key_in is None:

                plt.savefig(save_directory+'/'+title+'/Grid_Posterior'+profiles+plotted_posts+'.pdf', bbox_inches='tight')
                plt.clf()
            else:
                plt.subplots_adjust(top = 0.99, bottom = 0.05, right = 0.96, left = 0.04)
                xlim = ax[3,0].get_xlim()
                ylim = ax[3,0].get_ylim()
                
                ax[0,0].text(xlim[0]+0.975*(xlim[1]-xlim[0]),ylim[0]+0.975*(ylim[1]-ylim[0]), list(case.values())[0] + r' $\mu$m'+'\n'+r' $\mathrm{R}=50$'+'\n'+r'$\mathrm{S/N}=10$',ha = 'right',va='top')
                ax[1,0].text(xlim[0]+0.975*(xlim[1]-xlim[0]),ylim[0]+0.975*(ylim[1]-ylim[0]), list(case.values())[0] + r' $\mu$m'+'\n'+r' $\mathrm{R}=50$'+'\n'+r'$\mathrm{S/N}=20$',ha = 'right',va='top')
                ax[2,0].text(xlim[0]+0.975*(xlim[1]-xlim[0]),ylim[0]+0.975*(ylim[1]-ylim[0]), list(case.values())[0] + r' $\mu$m'+'\n'+r' $\mathrm{R}=100$'+'\n'+r'$\mathrm{S/N}=10$',ha = 'right',va='top')
                ax[3,0].text(xlim[0]+0.975*(xlim[1]-xlim[0]),ylim[0]+0.975*(ylim[1]-ylim[0]), list(case.values())[0] + r' $\mu$m'+'\n'+r' $\mathrm{R}=100$'+'\n'+r'$\mathrm{S/N}=20$',ha = 'right',va='top')
                
                ax[0,0].set_xlim(xlim)
                ax[0,0].set_ylim(ylim)
                ax[1,0].set_xlim(xlim)
                ax[1,0].set_ylim(ylim)
                ax[2,0].set_xlim(xlim)
                ax[2,0].set_ylim(ylim)
                ax[3,0].set_xlim(xlim)
                ax[3,0].set_ylim(ylim)
                
                plt.margins(0,0)

                plt.savefig(save_directory+'/'+title+'/'+key+'_Posterior'+profiles+plotted_posts+'.pdf')#, bbox_inches='tight')
                plt.clf()

            # Save the legend as separate pdf if it was not plotted
            if not legend_plotted:
                fig = plt.figure()
                fig.legend(handles,labels, frameon = False,loc='center',ncol = len(handles))
                plt.savefig(save_directory+'/'+title+'/'+key+'_Posterior_Legend.pdf', bbox_inches='tight')
                plt.clf()
            

        # Reduce the recusion limit
        sys.setrecursionlimit(1000)





    """
    #################################################################################
    #                                                                               #
    #   Routine for generating Spectrum plots.                                      #
    #                                                                               #
    #################################################################################
    """



    def Spectrum_Grid(self,x_category=None,y_category=None,subfig_size=[6,4],sharex=True,sharey=True,
                            plot_residual=False,quantiles=[0.05,0.15,0.25,0.35,0.65,0.75,0.85,0.95],
                            quantiles_title=[r'$5-95\%$',r'$15-85\%$',r'$25-75\%$',r'$35-65\%$'],colors=None,
                            case_identifiers=None,legend_loc = 'best'):
        
        # Generate the grid of runs for plotting        
        if type(x_category)==int:
            save_directory, x_dim, y_dim, o_dim, combinations_local_sub_categories, run_categorization = \
                self._grid_generator('Spectra',y_category=y_category)
            x_dim = x_category
            y_dim = ((y_dim-1)//x_dim+1)
        elif type(y_category)==int:
            save_directory, x_dim, y_dim, o_dim, combinations_local_sub_categories, run_categorization = \
                self._grid_generator('Spectra',x_category=x_category)
            y_dim = y_category
            x_dim = ((x_dim-1)//y_dim+1)
        else:
            save_directory, x_dim, y_dim, o_dim, combinations_local_sub_categories, run_categorization = \
                self._grid_generator('Spectra',x_category=x_category,y_category=y_category)

        # Loop over the various plots to do
        for ind in range(len(combinations_local_sub_categories)):
            case = combinations_local_sub_categories[ind]

            # Initialize a new figure for the plot
            fig,ax = plt.subplots(y_dim,x_dim,figsize = (x_dim*subfig_size[0],y_dim*subfig_size[1]),
                        sharex=sharex,sharey=sharey,squeeze=False)
            plt.subplots_adjust(hspace=0.1,wspace=0.1)

            # Loop over the x and y axis of the plots
            for x in range(x_dim):
                for y in range(y_dim):
                    try:
                        if type(x_category)==int:
                            run = run_categorization[ind,y+x*x_category,0,0]
                        elif type(y_category)==int:
                            run = run_categorization[ind,0,x+y*y_category,0]
                        else:
                            run = run_categorization[ind,y,x,0]

                        # Choose the correct color and hatches
                        if colors is None:
                            color = 'k'
                        else:
                            color = colors[[i for i in colors.keys() if i in run][0]]
                        if case_identifiers is None:
                            case_identifier = ''
                        else:
                            case_identifier = case_identifiers[[i for i in case_identifiers.keys() if i in run][0]]

                        # Plot the spectrum in the corresponding subplot
                        self.grid_results['rp_object'][run].Flux_Error(plot_residual=plot_residual,ax=ax[y,x],save =True,quantiles=quantiles,quantiles_title=quantiles_title,
                                        plot_noise = True, plot_true_spectrum = True, legend_loc = legend_loc,color = color,noise_title = 'LIFE Noise',case_identifier=case_identifier)
                    except:
                        ax[y,x].axis('off')

            # Label the axes of the subplots
            for i in range(y_dim):
                if plot_residual:
                    ax[i,0].set_ylabel(r'Retrieval Residual $\left[\%\right]$')
                else:
                    ax[i,0].set_ylabel(r'Flux at 10 pc $\left[\mathrm{\frac{erg}{s\,Hz\,m^2}}\right]$')
            for i in range(x_dim):
                ax[-1,i].set_xlabel(r'Wavelength [$\mu$m]')

            # Save the plots
            title = '_'.join([list(case.keys())[i]+list(case.values())[i]for i in range(len(case))])
            if plot_residual:
                plt.savefig(save_directory+title+'Spectra_Residual.pdf', bbox_inches='tight')
            else:
                plt.savefig(save_directory+title+'Spectra.pdf', bbox_inches='tight')





    """
    #################################################################################
    #                                                                               #
    #   Routines for generating PT profile plots.                                   #
    #                                                                               #
    #################################################################################
    """



    def PT_Grid(self,x_category=None,y_category=None,subfig_size=[8,6],sharex=True,sharey=True,plot_residual=False,plot_clouds=False,
                    quantiles=[0.05,0.15,0.25,0.35,0.65,0.75,0.85,0.95],quantiles_title=[r'$5-95\%$',r'$15-85\%$',r'$25-75\%$',r'$35-65\%$'],
                    colors=None,case_identifiers=None,legend_loc = 'best', x_lim =[0,1000], y_lim = [1e-6,1e4],true_cloud_top=None):

        # Generate the grid of runs for plotting        
        if type(x_category)==int:
            save_directory, x_dim, y_dim, o_dim, combinations_local_sub_categories, run_categorization = \
                self._grid_generator('PT_profiles',y_category=y_category)
            x_dim = x_category
            y_dim = ((y_dim-1)//x_dim+1)
        elif type(y_category)==int:
            save_directory, x_dim, y_dim, o_dim, combinations_local_sub_categories, run_categorization = \
                self._grid_generator('PT_profiles',x_category=x_category)
            y_dim = y_category
            x_dim = ((x_dim-1)//y_dim+1)
        else:
            save_directory, x_dim, y_dim, o_dim, combinations_local_sub_categories, run_categorization = \
                self._grid_generator('PT_profiles',x_category=x_category,y_category=y_category)

        # Generate the plots
        for ind in range(len(combinations_local_sub_categories)):
            case = combinations_local_sub_categories[ind]

            # Initialize a new figure for the plot
            fig,ax = plt.subplots(y_dim,x_dim,figsize = (x_dim*subfig_size[0],y_dim*subfig_size[1]),
                        sharex=sharex,sharey=sharey,squeeze=False)
            plt.subplots_adjust(hspace=0.06,wspace=0.04)

            # Loop over the x and y axis of the plots
            for y in range(y_dim):
                for x in range(x_dim):
                    try:
                        if type(x_category)==int:
                            run = run_categorization[ind,y+x*x_category,0,0]
                        elif type(y_category)==int:
                            run = run_categorization[ind,0,x+y*y_category,0]
                        else:
                            run = run_categorization[ind,y,x,0]
                            

                        # Choose the correct color and hatches
                        if colors is None:
                            color = 'k'
                        else:
                            color = colors[[i for i in colors.keys() if i in run][0]]
                        if case_identifiers is None:
                            case_identifier = ''
                        else:
                            case_identifier = case_identifiers[[i for i in case_identifiers.keys() if i in run][0]]

                        # Plot the PT profile in the corresponding subplot. If requested, try plotting clouds
                        try:
                            self.grid_results['rp_object'][run].PT_Envelope(plot_residual=plot_residual, plot_clouds = plot_clouds, x_lim=x_lim, y_lim=y_lim, quantiles=quantiles, quantiles_title=quantiles_title,
                                    inlay_loc='upper right', bins_inlay = 20,figure = fig, ax = ax[y,x], color=color, case_identifier=case_identifier, legend_loc=legend_loc,true_cloud_top=true_cloud_top)
                        except:
                            self.grid_results['rp_object'][run].PT_Envelope(plot_residual=plot_residual, plot_clouds = False, x_lim=x_lim, y_lim=y_lim, quantiles=quantiles, quantiles_title=quantiles_title,
                                    inlay_loc='upper right', bins_inlay = 20,figure = fig, ax = ax[y,x], color=color, case_identifier=case_identifier, legend_loc=legend_loc,true_cloud_top=true_cloud_top)
                    except:
                        ax[y,x].axis('off')
                        
            # Label the axes of the subplots
            for i in range(y_dim):
                ax[i,0].set_ylabel(r'Pressure [bar]')
            for i in range(x_dim):
                if plot_residual:
                    ax[-1,i].set_xlabel(r'Temperature Relative to Retrieved Median [K]')
                else:
                    ax[-1,i].set_xlabel(r'Temperature [K]')

            # Save the plots
            title = '_'.join([list(case.keys())[i]+list(case.values())[i]for i in range(len(case))])
            if plot_residual:
                plt.savefig(save_directory+title+'PT_Residual.pdf', bbox_inches='tight')
            else:
                plt.savefig(save_directory+title+'PT.pdf', bbox_inches='tight')





    """
    #################################################################################
    #                                                                               #
    #   Routines for generating Model Comaprison plots.                             #
    #                                                                               #
    #################################################################################
    """



    def Grid_Model_Comparison(self,x_category=None,y_category=None,x_cat_label=None,y_cat_label=None,model_category='Model',model_compare=None,facecolor='white',case_identifiers=None):
        # Generate the grid of runs for plotting
        save_directory, m_dim, m_cat, x_dim, x_cat, y_dim, y_cat, combinations_local_sub_categories, run_categorization = \
            self._grid_generator('Model_Compare',x_category=model_category,y_category=x_category,overplot_category=y_category,return_all=True)

        # Default model compar is first model
        if model_compare is None:
            model_compare = m_cat[0]
        
        # Define a matrix to store the Bayes factors
        K_Matrix = np.zeros(((y_dim+1)*(m_dim-1)-1,(x_dim+1)*len(combinations_local_sub_categories)-1))-100

        # Iterate over models and wavelength ranges and set he matrix elements
        n_cases = len(combinations_local_sub_categories)
        for ind in range(len(combinations_local_sub_categories)):
            m_corr = 0
            m_compare = [i for i in range(m_dim) if m_cat[i]==model_compare][0]
            m_combinations = []

            #Loop over all elements nor equal to m_corr
            for m in range(m_dim):
                if m == m_compare:
                    m_corr -= 1
                else:
                    for x in range(x_dim):
                        for y in range(y_dim):
                            # Get the evidences for the runs
                            evidence = self.grid_results['rp_object'][run_categorization[ind,x,m,y]].evidence
                            evidence_compare = self.grid_results['rp_object'][run_categorization[ind,x,m_compare,y]].evidence

                            # Calculate the bayes factor and store the result in the matrix
                            K = 0.4342944819*(float(evidence_compare[0])-float(evidence[0]))
                            K_Matrix[(y_dim+1)*(m+m_corr)+y,(x_dim+1)*(ind)+x] = K
                          
                    # Save the name of the case combination
                    if case_identifiers is None:
                        m_combinations += [model_compare+' vs.\n'+m_cat[m]]
                    else:
                        m_combinations += [case_identifiers[[i for i in case_identifiers.keys() if i in run_categorization[ind,0,m_compare,0]][0]]+\
                            ' vs.\n'+case_identifiers[[i for i in case_identifiers.keys() if i in run_categorization[ind,0,m,0]][0]]]

            # Define the color map according to jeffrey's scale
            cmap = col.ListedColormap(['white','#d62728','#d6272880','#d6272860','#d6272840','#2ca02c40','#2ca02c60','#2ca02c80','#2ca02c'])
            bounds=[-1000,-99,-2,-1,-0.5,0.0,0.5,1,2,99]
            norm = col.BoundaryNorm(bounds, cmap.N)


        # Initial plot configuration
        lw = 2
        C_y = np.shape(K_Matrix)[0]+5/1.5
        fig,ax = plt.subplots(2,figsize = (0.6*np.shape(K_Matrix)[1],0.6*np.shape(K_Matrix)[0]),gridspec_kw={'height_ratios': [np.shape(K_Matrix)[0]/C_y,5/1.5/C_y]})#,linewidth=2*lw, edgecolor="#007272")
        plt.subplots_adjust(hspace=0,wspace=0)
        fig.patch.set_facecolor(facecolor)
        ax[0].axis('off')
        ax[1].axis('off')

        # Plot the matrix and separate the different elements
        ax[0].matshow(K_Matrix,cmap=cmap, norm=norm)
        ax[0].vlines([-0.5+i for i in range(np.shape(K_Matrix)[1]+1)],-0.5,np.shape(K_Matrix)[0]-0.5,color=facecolor,lw=lw)
        ax[0].hlines([-0.5+i for i in range(np.shape(K_Matrix)[0]+1)],-0.5,np.shape(K_Matrix)[1]-0.5,color=facecolor,lw=lw)

        # Annotate the y-axis
        y_ax=[y_dim/2-0.5+i*(y_dim+1)for i in range(m_dim-1)]
        y_shift = -((y_dim+1)%2)*0.5-(y_dim-1)//2
        for pos_y in range(m_dim-1):
            for ind_y in range(y_dim):
                ax[0].text(-0.6, y_ax[pos_y]+y_shift+ind_y+0.03,y_cat[ind_y],rotation = 0,rotation_mode='default',ha='right',va='center')
            if y_cat_label is None:
                ax[0].text(-1.65, y_ax[pos_y],y_category,rotation = 90,rotation_mode='default',ha='center',va='center')
            else:
                ax[0].text(-1.65, y_ax[pos_y],y_cat_label,rotation = 90,rotation_mode='default',ha='center',va='center')
            ax[0].text(-2.55, y_ax[pos_y],m_combinations[pos_y],rotation = 90,rotation_mode='default',ha='center',va='center')

        # Annotate the x-axis
        x_ax=[x_dim/2-0.5+i*(x_dim+1)for i in range(n_cases)]
        x_shift = -((x_dim+1)%2)*0.5-(x_dim-1)//2
        for pos_x in range(n_cases):
            for ind_x in range(x_dim):
                ax[0].text(x_ax[pos_x]+x_shift+ind_x,-0.7,x_cat[ind_x],rotation = 0,rotation_mode='default',ha='center',va='center')
            if x_cat_label is None:
                ax[0].text(x_ax[pos_x],-1.35,x_category,rotation = 0,rotation_mode='default',ha='center',va='center') #ax[0].text(x_ax[pos_x],-1.35,x_category,rotation = 0,rotation_mode='default',ha='center',va='center')
            else:
                ax[0].text(x_ax[pos_x],-1.35,x_cat_label,rotation = 0,rotation_mode='default',ha='center',va='center') #ax[0].text(x_ax[pos_x],-1.35,x_category,rotation = 0,rotation_mode='default',ha='center',va='center')
            ax[0].text(x_ax[pos_x],-2.15,[list(combinations_local_sub_categories[i].keys())[0]+'\n'+list(combinations_local_sub_categories[i].values())[0]+' $\mu$m'\
                 for i in range(len(combinations_local_sub_categories))][pos_x],rotation = 0,rotation_mode='default',ha='center',va='center')

    
        # Write the Bayes factor in the fields
        for pos_y in range(np.shape(K_Matrix)[0]):
            for pos_x in range(np.shape(K_Matrix)[1]):
                    if K_Matrix[pos_y,pos_x]!=-100:
                        txt = ax[0].annotate(str(np.round(K_Matrix[pos_y,pos_x],1)),[pos_x,pos_y+0.03],ha='center',va='center',color='white',fontweight = 'bold')
                        # Place an edge around the text
                        if K_Matrix[pos_y,pos_x]<0:
                            txt.set_path_effects([PathEffects.withStroke(linewidth=0.7, foreground='#d62728')])
                        else:
                            txt.set_path_effects([PathEffects.withStroke(linewidth=0.7, foreground='#2ca02c')])

    
        # Plot for the color scale of the Bayes' factor
        Plot_Matrix = np.zeros((5,8))-100
        bounds = [-2.0,-1.0,-0.5,0.0,0.5,1.0,2.0]
        Plot_Matrix[-2,:]=[i-0.25 for i in bounds]+[bounds[-1]+0.25]

        # Cosmetics for the reference scale (nines arrows...)
        ax[1].matshow(Plot_Matrix,cmap=cmap, norm=norm)
        ax[1].hlines([2.5,3.5],-0.5,7.5,color=facecolor,lw=2*lw)
        ax[1].vlines([i-0.5 for i in range(0,9)],0,4,color=facecolor,lw=2*lw)
        ax[1].vlines([i-0.5 for i in range(1,8)],2.47,3.53,color='k',lw=lw/2)
        ax[1].vlines(3.5,2.47,4.2,color='k',lw=lw/2)
        for ind_bound in range(len(bounds)):
            ax[1].text(0.57+ind_bound,2.3,str(bounds[ind_bound]),rotation = 90,rotation_mode='default',ha='center',va='bottom')

        ax[1].plot([-0.28,3.365],[3.85,3.85],color='k',ls='-',lw=lw/2)
        ax[1].plot([3.635,7.5-0.28],[3.85,3.85],color='k',ls='-',lw=lw/2)
        ax[1].plot([-0.1,-0.35,-0.1],[3.75,3.85,3.95],color='k',ls='-',lw=lw/2)
        ax[1].plot([7.5-0.4,7.5-0.15,7.5-0.4],[3.75,3.85,3.95],color='k',ls='-',lw=lw/2)
        for i in range(101):
            ax[1].fill([3.4-i/100,3.4-(i+1)/100,3.4-(i+1)/100,3.4-i/100],[5-1.05,5-1.05,5-1.25,5-1.25],color=facecolor,alpha=1-i/100,lw=0,zorder=5)
            ax[1].fill([3.6+i/100,3.6+(i+1)/100,3.6+(i+1)/100,3.6+i/100],[5-1.05,5-1.05,5-1.25,5-1.25],color=facecolor,alpha=1-i/100,lw=0,zorder=5)
        
        # Annotation
        ax[1].text(3.5,0.8,'Color Coding of '+r'$\mathrm{log_{10}(K)}$',rotation = 0,rotation_mode='default',ha='center',va='center',color='k')
        ax[1].text(3,4.3,r'Other Models',rotation = 0,rotation_mode='default',ha='right',va='center',color='k')
        ax[1].text(4,4.3,case_identifiers[[i for i in case_identifiers.keys() if i in run_categorization[ind,0,m_compare,0]][0]]\
            ,rotation = 0,rotation_mode='default',ha='left',va='center',color='k')
        #plt.savefig('Results/Abundances/Summary/'+Wlen_grid[i]+'.png',dpi=450,bbox_inches='tight', facecolor='azure')
        #title = plt.title('Bayes factor for model '+str(model_compare)+'.', y=1.15)

        plt.savefig(save_directory+'/'+model_compare+'baye_total.pdf', bbox_inches='tight', facecolor=facecolor)

































































































































    """


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

    """

                


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
                                        self.grid_results[model][wln][res][snr].get_pt
                                    if param in ['T0','T_eq']:
                                        if self.grid_results[model][wln][res][snr].settings['clouds'] == 'opaque': #len(self.grid_results[model][wln][res][snr].cloud_vars) != 0:
                                            post = self.grid_results[model][wln][res][snr].temperature_cloud_top
                                            true = bond_params[5]
                                        else:
                                            post = self.grid_results[model][wln][res][snr].temperature[:,-1]
                                            true = bond_params[5]
                                    else:
                                        if self.grid_results[model][wln][res][snr].settings['clouds'] == 'opaque':
                                            post = np.log10(self.grid_results[model][wln][res][snr].pressure_cloud_top)
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















    def Parameters_Hist(self, parameters=['R_pl','M_pl','H2O','CO2','CO','H2SO484(c)_am_top_pressure','H2SO484(c)_am','H2SO484(c)_am_thickness','H2SO484(c)_am_particle_radius','H2SO484(c)_am_sigma_lnorm','A_Bond'], priors = [True, True, False, False, False], units=['$\mathrm{R_\oplus}$','dex','dex','dex','dex','dex','dex','dex','dex','','',''],
                        span = [0.2,0.5,0.5,20,0.1], titles = None, bond_params = [1,0.05,0.723,0.05*0.723,0.77,226,5e-2],filename=None,skip=1,n_processes=50):

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
                                        self.grid_results[model][wln][res][snr].get_pt(skip=skip,n_processes=n_processes)
                                    if param in ['T0','T_eq']:
                                        if self.grid_results[model][wln][res][snr].settings['clouds'] == 'opaque': #len(self.grid_results[model][wln][res][snr].cloud_vars) != 0:
                                            post = self.grid_results[model][wln][res][snr].temperature_cloud_top
                                            true = bond_params[5]
                                        else:
                                            post = self.grid_results[model][wln][res][snr].temperature[:,-1]
                                            true = bond_params[5]
                                    else:
                                        if self.grid_results[model][wln][res][snr].settings['clouds'] == 'opaque':
                                            post = np.log10(self.grid_results[model][wln][res][snr].pressure_cloud_top)
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


















    def Parameters_Hist_Top(self, parameters=['R_pl','M_pl','H2O','CO2','CO','H2SO484(c)_am'], ranges = [0.1,0.1,1,1,1,1], units=['$\mathrm{R_\oplus}$','dex','dex','dex','dex','dex','dex','dex','dex','','',''],bins = 100,
                            titles = None, bond_params = [1,0.05,0.723,0.05*0.723,0.77,226,5e-2],filename=None,skip=1,n_processes=50):

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
                                        self.grid_results[model][wln][res][snr].get_pt(skip=skip,n_processes=n_processes)
                                    if param in ['T0','T_eq']:
                                        if self.grid_results[model][wln][res][snr].settings['clouds'] == 'opaque': #len(self.grid_results[model][wln][res][snr].cloud_vars) != 0:
                                            post = self.grid_results[model][wln][res][snr].temperature_cloud_top
                                            true = bond_params[5]
                                        else:
                                            post = self.grid_results[model][wln][res][snr].temperature[:,-1]
                                            true = bond_params[5]
                                    else:
                                        if self.grid_results[model][wln][res][snr].settings['clouds'] == 'opaque':
                                            post = np.log10(self.grid_results[model][wln][res][snr].pressure_cloud_top)
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
            



























    def Grid_Model_ComparisonT(self,x_category=None,y_category=None,x_cat_label=None,y_cat_label=None,model_category='Model',model_compare=None,facecolor='white',case_identifiers=None):
        # Generate the grid of runs for plotting
        save_directory, m_dim, m_cat, x_dim, x_cat, y_dim, y_cat, combinations_local_sub_categories, run_categorization = \
            self._grid_generator('Model_Compare',x_category=model_category,y_category=x_category,overplot_category=y_category,return_all=True)

        # Default model compar is first model
        if model_compare is None:
            model_compare = m_cat[0]
        
        # Define a matrix to store the Bayes factors
        K_Matrix = np.zeros(((y_dim+1)*len(combinations_local_sub_categories)-1,(x_dim+1)*(m_dim-1)-1))-100

        # Iterate over models and wavelength ranges and set he matrix elements
        n_cases = len(combinations_local_sub_categories)
        for ind in range(len(combinations_local_sub_categories)):
            m_corr = 0
            m_compare = [i for i in range(m_dim) if m_cat[i]==model_compare][0]
            m_combinations = []

            #Loop over all elements nor equal to m_corr
            for m in range(m_dim):
                if m == m_compare:
                    m_corr -= 1
                else:
                    for x in range(x_dim):
                        for y in range(y_dim):
                            # Get the evidences for the runs
                            evidence = self.grid_results['rp_object'][run_categorization[ind,x,m,y]].evidence
                            evidence_compare = self.grid_results['rp_object'][run_categorization[ind,x,m_compare,y]].evidence

                            # Calculate the bayes factor and store the result in the matrix
                            K = 0.4342944819*(float(evidence_compare[0])-float(evidence[0]))
                            K_Matrix[(y_dim+1)*(ind)+y,(x_dim+1)*(m+m_corr)+x] = K
                          
                    # Save the name of the case combination
                    if case_identifiers is None:
                        m_combinations += [model_compare+' vs.\n'+m_cat[m]]
                    else:
                        m_combinations += [case_identifiers[[i for i in case_identifiers.keys() if i in run_categorization[ind,0,m_compare,0]][0]]+\
                            ' vs.\n'+case_identifiers[[i for i in case_identifiers.keys() if i in run_categorization[ind,0,m,0]][0]]]

            # Define the color map according to jeffrey's scale
            cmap = col.ListedColormap(['white','#d62728','#d6272880','#d6272860','#d6272840','#2ca02c40','#2ca02c60','#2ca02c80','#2ca02c'])
            bounds=[-1000,-99,-2,-1,-0.5,0.0,0.5,1,2,99]
            norm = col.BoundaryNorm(bounds, cmap.N)

        #K_Matrix = K_Matrix.T

        # Initial plot configuration
        lw = 2
        C_y = np.shape(K_Matrix)[1]+6/1.5
        fig,ax = plt.subplots(1,2,figsize = (0.65*np.shape(K_Matrix)[1],0.65*np.shape(K_Matrix)[0]),gridspec_kw={'width_ratios': [np.shape(K_Matrix)[1]/C_y,6/1.5/C_y]})#,linewidth=2*lw, edgecolor="#007272")
        plt.subplots_adjust(hspace=0,wspace=0)
        fig.patch.set_facecolor(facecolor)
        ax[0].axis('off')
        ax[1].axis('off')

        # Plot the matrix and separate the different elements
        ax[0].matshow(K_Matrix,cmap=cmap, norm=norm)
        ax[0].vlines([-0.5+i for i in range(np.shape(K_Matrix)[1]+1)],-0.5,np.shape(K_Matrix)[0]-0.5,color=facecolor,lw=lw)
        ax[0].hlines([-0.5+i for i in range(np.shape(K_Matrix)[0]+1)],-0.5,np.shape(K_Matrix)[1]-0.5,color=facecolor,lw=lw)

        # Annotate the y-axis
        y_ax=[y_dim/2-0.5+i*(y_dim+1)for i in range(n_cases)]
        y_shift = -((y_dim+1)%2)*0.5-(y_dim-1)//2
        for pos_y in range(n_cases):
            for ind_y in range(y_dim):
                ax[0].text(-0.6, y_ax[pos_y]+y_shift+ind_y+0.03,y_cat[ind_y],rotation = 0,rotation_mode='default',ha='right',va='center')
            if y_cat_label is None:
                ax[0].text(-1.65, y_ax[pos_y],y_category,rotation = 90,rotation_mode='default',ha='center',va='center')
            else:
                ax[0].text(-1.65, y_ax[pos_y],y_cat_label,rotation = 90,rotation_mode='default',ha='center',va='center')
            ax[0].text(-2.55, y_ax[pos_y],[list(combinations_local_sub_categories[i].keys())[0]+'\n'+list(combinations_local_sub_categories[i].values())[0]+' $\mu$m'\
                 for i in range(len(combinations_local_sub_categories))][pos_y],rotation = 90,rotation_mode='default',ha='center',va='center')

        # Annotate the x-axis
        x_ax=[x_dim/2-0.5+i*(x_dim+1)for i in range(m_dim-1)]
        x_shift = -((x_dim+1)%2)*0.5-(x_dim-1)//2
        for pos_x in range(m_dim-1):
            for ind_x in range(x_dim):
                ax[0].text(x_ax[pos_x]+x_shift+ind_x,-0.7,x_cat[ind_x],rotation = 0,rotation_mode='default',ha='center',va='center')
            if x_cat_label is None:
                ax[0].text(x_ax[pos_x],-1.35,x_category,rotation = 0,rotation_mode='default',ha='center',va='center') #ax[0].text(x_ax[pos_x],-1.35,x_category,rotation = 0,rotation_mode='default',ha='center',va='center')
            else:
                ax[0].text(x_ax[pos_x],-1.35,x_cat_label,rotation = 0,rotation_mode='default',ha='center',va='center') #ax[0].text(x_ax[pos_x],-1.35,x_category,rotation = 0,rotation_mode='default',ha='center',va='center')
            ax[0].text(x_ax[pos_x],-2.15,m_combinations[pos_x],rotation = 0,rotation_mode='default',ha='center',va='center')

    
        # Write the Bayes factor in the fields
        for pos_y in range(np.shape(K_Matrix)[0]):
            for pos_x in range(np.shape(K_Matrix)[1]):
                    if K_Matrix[pos_y,pos_x]!=-100:
                        txt = ax[0].annotate(str(np.round(K_Matrix[pos_y,pos_x],1)),[pos_x,pos_y+0.03],ha='center',va='center',color='white',fontweight = 'bold')
                        # Place an edge around the text
                        if K_Matrix[pos_y,pos_x]<0:
                            txt.set_path_effects([PathEffects.withStroke(linewidth=0.7, foreground='#d62728')])
                        else:
                            txt.set_path_effects([PathEffects.withStroke(linewidth=0.7, foreground='#2ca02c')])

    
        # Plot for the color scale of the Bayes' factor
        Plot_Matrix = np.zeros((6,8))-100
        bounds = [-2.0,-1.0,-0.5,0.0,0.5,1.0,2.0]
        Plot_Matrix[-2,:]=[i-0.25 for i in bounds]+[bounds[-1]+0.25]
        Plot_Matrix = Plot_Matrix.T

        # Cosmetics for the reference scale (nines arrows...)
        ax[1].matshow(Plot_Matrix,cmap=cmap, norm=norm)
        ax[1].vlines([3.5,4.5],-0.5,7.5,color=facecolor,lw=2*lw)
        ax[1].hlines([i-0.5 for i in range(0,9)],0,5,color=facecolor,lw=2*lw)
        ax[1].hlines([i-0.5 for i in range(1,8)],3.47,4.53,color='k',lw=lw/2)
        ax[1].hlines(3.5,3.47,5.2,color='k',lw=lw/2)
        for ind_bound in range(len(bounds)):
            ax[1].text(3.3,0.57+ind_bound,str(bounds[ind_bound]),rotation = 0,ha='right',va='center')

        ax[1].plot([4.85,4.85],[-0.28,3.365],color='k',ls='-',lw=lw/2)
        ax[1].plot([4.85,4.85],[3.635,7.5-0.28],color='k',ls='-',lw=lw/2)
        ax[1].plot([4.75,4.85,4.95],[-0.1,-0.35,-0.1],color='k',ls='-',lw=lw/2)
        ax[1].plot([4.75,4.85,4.95],[7.5-0.4,7.5-0.15,7.5-0.4],color='k',ls='-',lw=lw/2)
        for i in range(101):
            ax[1].fill([6-1.05,6-1.05,6-1.25,6-1.25],[3.4-i/100,3.4-(i+1)/100,3.4-(i+1)/100,3.4-i/100],color=facecolor,alpha=1-i/100,lw=0,zorder=5)
            ax[1].fill([6-1.05,6-1.05,6-1.25,6-1.25],[3.6+i/100,3.6+(i+1)/100,3.6+(i+1)/100,3.6+i/100],color=facecolor,alpha=1-i/100,lw=0,zorder=5)
        
        # Annotation
        ax[1].text(1.8,3.5,'Color Coding of '+r'$\mathrm{log_{10}(K)}$',rotation = 90,rotation_mode='default',ha='center',va='center',color='k')
        ax[1].text(5.3,3,r'Other Models',rotation = 90,ha='center',va='bottom',color='k')
        ax[1].text(5.33,4,case_identifiers[[i for i in case_identifiers.keys() if i in run_categorization[ind,0,m_compare,0]][0]]\
            ,rotation = 90,ha='center',va='top',color='k')
        #plt.savefig('Results/Abundances/Summary/'+Wlen_grid[i]+'.png',dpi=450,bbox_inches='tight', facecolor='azure')
        #title = plt.title('Bayes factor for model '+str(model_compare)+'.', y=1.15)

        plt.savefig(save_directory+'/'+model_compare+'baye_total.pdf', bbox_inches='tight', facecolor=facecolor)





















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