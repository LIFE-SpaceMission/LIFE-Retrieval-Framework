__author__ = "Konrad"
__copyright__ = "Copyright 2022, Konrad"
__maintainer__ = "Björn S. Konrad"
__email__ = "konradb@phys.ethz.ch"
__status__ = "Development"

# Standard Libraries
import sys, os
import matplotlib.colors as col
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

# Import additional external files
import retrieval_plotting as rp
from retrieval_support import retrieval_posteriors as r_post
from retrieval_plotting_support import retrieval_plotting_handlerbase as rp_hndl





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
                        results = rp.retrieval_plotting(model_run+'/')
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




    def Spectrum_grid(self,skip = 1, NoQuant = False):
        
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