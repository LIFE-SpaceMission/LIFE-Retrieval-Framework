import sys

# import the trequired codes from the terrestrial retrieval package
sys.path.append("/home/konradb/Documents/Terrestrial_Retrieval/")
import retrieval_plotting as rp
import retrieval_plotting_grid as rpg
from retrieval_plotting_support import retrieval_plotting_adjust_config as rp_config

import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u



# initialize rp object with the directory of the retrieval run
results = rp.retrieval_plotting('/home/ipa/quanz/user_accounts/konradb/chains/Test_Unit_Retrieval/')





# Corner Plot
# Set custom titles for the units and for the parameter names
unit_titles = {'R_pl':'$\mathrm{R_{Earth}}$','M_pl':'$\mathrm{M_{Earth}}$'}
titles = {'R_pl':'$\mathrm{R_{pl}}$'}

# set units to plot. All units not set are plotted in retrieval units, if units = 'input' then
# everything is plotted in retrieval units
units = {'R_pl':u.km,'M_pl':u.t}

# make the corner plot
results.Posteriors(save=True,                       #Save the plot
                    plot_corner=True,               #Corner plot. If false the individual posteriors are plotted
                    plot_pt=True,                   #Plot the PT parameters
                    plot_physparam=True,            #Plot the physical parameters (e.g. R_pl)
                    plot_clouds=True,               #Plot the cloud parameters
                    plot_chemcomp=True,             #Plot the chemical composition parameters
                    plot_scatt=True,                #Plot the scattering parameters
                    log_pressures=True,             #Plot all pressures in log
                    log_mass=True,                  #Plot all masses in log
                    log_abundances=True,            #Plot all abundances in log
                    log_particle_radii=True,        #Plot all cloud particle radii in log
                    bins=40,                        #Bins for the Histograms
                    quantiles1d=[0.16, 0.5, 0.84],  #Quantiles in the 1D Histograms
                    color='#009e73',                #Color of the histograms
                    add_table=True,                 #Plot a table listing the retrieved values
                    color_truth='k',                #Color of the true values
                    units=units,                    #Units to plot. If not provided Retrieval units
                    unit_titles=unit_titles,        #Titles of units to plot. If not provided astropy generated titles
                    titles=titles)                  #Titles of parameters. If not provided generated automatically





# Flux envelope code
results.Flux_Error(skip =1,                                                                 #Skip value for the spectrum calculation
                    n_processes=50,                                                         #Number of cores to calculate the spectra on
                    reevaluate_spectra=False,                                               #Force the recalculation of the spectra
                    save =True,                                                             #Save the plot
                    figsize=(15.275, 3.02),                                                 #Size of the figure
                    x_lim = None,                                                           #Custom limits for the x axis
                    y_lim = None,                                                           #Custom limits for the y axis
                    log_x=False,                                                            #x axis in log-scale
                    log_y=False,                                                            #y axis in log-scale
                    plot_residual = False,                                                  #Plot the true flux of the residual relative to the input flux
                    median_only=False,                                                      #plot only the retrieved median without the envelopes
                    quantiles=[0.05,0.15,0.25,0.35,0.65,0.75,0.85,0.95],                    #Plotted flux quantiles
                    quantiles_title=[r'$5-95\%$',r'$15-85\%$',r'$25-75\%$',r'$35-65\%$'],   #Titles for the flux quantiles
                    color='C2',                                                             #Color for the quantiles
                    plot_noise = True,                                                      #Plot the noise as shaded area
                    plot_datapoints = False,                                                #Plot input as datapoints with errorbars
                    plot_true_spectrum = True,                                              #Plot the input spectrum as line
                    noise_title = 'Photon Noise',                                           #Title for the noise
                    legend_loc = 'upper center',                                            #Position of the legend
                    split_instruments=True,                                                 #Rebin the flux to the individual instruments
                    single_instrument=None,                                                 #Plot a single instrument 'instrument_name'             
                    x_unit=u.micron,                                                        #Astropy units for the x-coordinate
                    y_unit=u.photon/u.m**2/u.s/u.micron)                                    #Astropy units for the y-coordinate
                    




# PT envelope code
results.PT_Envelope(skip=1,                                                                 #Skip value for the PT calculation
                    n_processes=2,                                                          #Number of cores to calculate the PT profiles on
                    reevaluate_PT = False,                                                  #Force the recalculation of the PT profiles
                    save=True,                                                              #Save the plot
                    figsize=(6.4, 4.8),                                                     #Size of the figure
                    x_lim =[0,1000],                                                        #Custom limits for the x axis
                    y_lim = [1e-6,1e4],                                                     #Custom limits for the y axis
                    log_x=False,                                                            #x axis in log-scale
                    log_y=False,                                                            #y axis in log-scale
                    plot_residual = False,                                                  #Plot the true flux of the residual relative to the input flux
                    plot_clouds = False,                                                    #Plot the retrieved cloud layer
                    true_cloud_top=[None,None],                                             #Position of the true CT [Temperature,pressure]
                    quantiles=[0.05,0.15,0.25,0.35,0.65,0.75,0.85,0.95],                    #Plotted PT quantiles
                    quantiles_title=[r'$5-95\%$',r'$15-85\%$',r'$25-75\%$',r'$35-65\%$'],   #Titles for the PT quantiles
                    color='#009e73',                                                        #Color for the quantiles
                    inlay_loc='upper right',                                                #Position of the inlay
                    bins_inlay = 20,                                                        #Bins for the inlay 2d posterior
                    x_lim_inlay = None,                                                     #Custom limits for the inlay x axis
                    y_lim_inlay = None,                                                     #Custom limits for the inlay y axis
                    h_cover=0.45,                                                           #Percentage of plot the inlay covers
                    legend_loc = 'lower left',                                              #Position of the legend
                    legend_n_col = 2,                                                       #Number of columns in the legend
                    x_unit=u.K,                                                             #Astropy units for the x-coordinate
                    y_unit=u.bar)                                                           #Astropy units for the y-coordinate

 
