__author__ = "Konrad"
__copyright__ = "Copyright 2022, Konrad"
__maintainer__ = "Björn S. Konrad"
__email__ = "konradb@phys.ethz.ch"
__status__ = "Development"

# Standard Libraries
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

# Import additional external files
from retrieval_plotting_support import retrieval_plotting_colors as rp_col





# Routine for generating corner plots
def Corner(data,titles,units=None,truths=None,dimension=None,quantiles1d = [0.16, 0.5, 0.84], 
            quantiles2d=[0.02,0.15,0.25,0.35,0.65,0.75,0.85,0.98],color='k',color_truth='C3',bins=50,add_table = False,ULU=[],ULU_lim=[-0.15,0.75]):
    
    # Find the dimension of the corner plot.
    if dimension is None:
        dimension=np.shape(data)[1]

    # Generate colorlevels for the different quantiles
    color_levels, level_thresholds, N_levels = rp_col.color_levels(color,quantiles2d)

    if add_table:
        table = []
        columns = ('True', 'Retrieved')
        rows = titles
        colours = ['white','white']

    # Start of plotting routine
    #fig, axs = plt.subplots(dimension, dimension,figsize=(dimension*2.5,dimension*2.5))  
    # Note NL: previous line now needs to be defined outside of function -> moved to posteriors fnc in retrieval_plotting.py
    fig.subplots_adjust(hspace=0.0,wspace=0.0)
    fs = 18
    
    # Iterate over the equal weighted posteriors of all retrieved parameters.
    # Diagonal histogram plots
    for i in range(dimension):
        # Plot the 1d histogram for each retrieved parameter on the diagonal.
        if i in ULU:  
            h = np.histogram(data[:,i],density=True,bins=bins,range = (ULU_lim[0],0))
            h2 = np.histogram(np.log10(1-10**(np.arange(-7,0,0.000001))),density=True,bins=h[1])
            h = (h[0]/h2[0],h[1])
            h = axs[i,i].hist(h[1][: -1],h[1], weights = sp.ndimage.filters.gaussian_filter(h[0], [ULU_lim[1]], mode='constant'),histtype='stepfilled',color=color_levels[1],density=True)
        else:
            h = axs[i,i].hist(data[:,i],histtype='stepfilled',color=color_levels[1],density=True,bins=bins)

        # Define the limits of the plot and remove the yticks
        axs[i,i].set_xlim([h[1][0],h[1][-1]])
        axs[i,i].set_ylim([0,1.1*np.max(h[0])])
        axs[i,i].set_yticks([])

        # Plotting the secified quantiles
        if quantiles1d is not None:
            if i in ULU:
                h_cumulative = np.array([sum(h[0][:i+1])/sum(h[0]) for i in range(len(h[0]))])
                ind = [np.min(np.where(h_cumulative>=quantiles1d[i]))for i in range(len(quantiles1d))]
                q = ((h[1][1:]+h[1][:-1])/2)[ind]
            else:
                q = [np.quantile(data[:,i],ind) for ind in quantiles1d]
            axs[i,i].vlines(q,axs[i,i].get_ylim()[0],axs[i,i].get_ylim()[1],colors='k', ls='--')

            # Round q and print the retrieved value above the histogram plot
            round = min(np.log10(abs(q[2]-q[1])),np.log10(abs(q[0]-q[1])))
            if round>=0.5:
                if add_table:
                    table.append([str(int(truths[i])),str(int(q[1]))+r' $_{\,'+str(int(q[0]-q[1]))+r'}^{\,+'+str(int(q[2]-q[1]))+r'}$'])
                else:
                    axs[i,i].set_title(str(int(q[1]))+r' $_{\,'+str(int(q[0]-q[1]))+r'}^{\,+'+str(int(q[2]-q[1]))+r'}$',fontsize=fs)
            else:
                if add_table:
                    table.append([str(np.round(truths[i],int(-np.floor(round-0.5)))),
                            str(np.round(q[1],int(-np.floor(round-0.5))))+r' $_{\,'+\
                            str(np.round(q[0]-q[1],int(-np.floor(round-0.5))))+r'}^{\,+'+\
                            str(np.round(q[2]-q[1],int(-np.floor(round-0.5))))+r'}$'])
                else:
                    axs[i,i].set_title(str(np.round(q[1],int(-np.floor(round-0.5))))+r' $_{\,'+\
                            str(np.round(q[0]-q[1],int(-np.floor(round-0.5))))+r'}^{\,+'+\
                            str(np.round(q[2]-q[1],int(-np.floor(round-0.5))))+r'}$',fontsize=fs)

        # Plot the ground truth values if known
        if not truths[i] is None:
            axs[i,i].plot([truths[i],truths[i]],axs[i,i].get_ylim(),color=color_truth,linestyle = ':',linewidth = 2)
    
    # 2d histograms plotted below the diagonal
    for i in range(dimension):
        for j in range(dimension):
            # Find the axis boundaries and set the x limits
            ylim = axs[i,i].get_xlim()
            xlim = axs[j,j].get_xlim()

            # Setting 4 even ticks over the range defined by the limits of the subplot
            yticks = [(1-pos)*ylim[0]+pos*ylim[1] for pos in [0.2,0.4,0.6,0.8]]
            xticks = [(1-pos)*xlim[0]+pos*xlim[1] for pos in [0.2,0.4,0.6,0.8]]

            # For all subplots below the Diagonal
            if i > j:
                # Plot the local truth values if provided
                
                if not truths[j] is None:
                    axs[i,j].plot([truths[j],truths[j]],[-10000,10000],color=color_truth,linestyle = ':',linewidth = 2)
                if not truths[i] is None:
                    axs[i,j].plot([-10000,10000],[truths[i],truths[i]],color=color_truth,linestyle = ':',linewidth = 2)
                if not ((truths[j] is None) or (truths[i] is None)):
                    axs[i,j].plot(truths[j],truths[i],color=color_truth,marker='o',markersize=8)
                
                # Plot the 2d histograms between different parameters to show correlations between the parameters                
                Z,X,Y=np.histogram2d(data[:,j],data[:,i],bins=bins,range = [list(xlim),list(ylim)])


                if i in ULU:
                    h = np.histogram(np.log10(1-10**(np.arange(-7,0,0.000001))),density=True,bins=Y)
                    Z = Z/(h[0][None,:])
                    #axs[j,j].hist(X[: -1],X,weights = np.sum(Z,axis = 1),histtype='step',density = True)
                    #axs[j,j].hist(h[1][: -1],h[1], weights = sp.ndimage.filters.gaussian_filter(h[0], [ULU_lim[1]], mode='constant'),histtype='stepfilled',color=color_levels[1],density=True)
                    #Z = sp.ndimage.filters.gaussian_filter(Z, [0.75,0.75], mode='reflect')
                if j in ULU:
                    h = np.histogram(np.log10(1-10**(np.arange(-7,0,0.000001))),density=True,bins=X)
                    Z = Z/(h[0][:,None])
                    #axs[i,i].hist(Y[: -1],Y,weights = np.sum(Z,axis = 0),histtype='step',density = True)
                Z = sp.ndimage.filters.gaussian_filter(Z, [ULU_lim[1],ULU_lim[1]], mode='reflect')


                map, norm, levels = rp_col.color_map(Z,color_levels,level_thresholds)
                axs[i,j].contourf((X[:-1]+X[1:])/2,(Y[:-1]+Y[1:])/2,Z.T,cmap=map,norm=norm,levels=np.array(levels),alpha=0.8)

                # Setting the limit of the x,y-axis
                axs[i,j].set_ylim(ylim)
                axs[i,j].set_xlim(xlim)

            # No Subplots show above the diagonal
            elif i<j:
                axs[i,j].axis('off')
    
            # Add the ticks and the axis labels where necessary on the y axis               
            if j == 0 and i!=0:
                axs[i,j].set_yticks(yticks)

                # Rounding the ticklabels and printing them
                roundy = np.log10(np.abs(yticks[1]-yticks[0]))
                if roundy>=0.5:
                    axs[i,j].set_yticklabels([int(i) for i in yticks],fontsize=fs,rotation=45)
                else:
                    axs[i,j].set_yticklabels(np.round(yticks,int(-np.floor(roundy-0.5))),fontsize=fs,rotation=45)
                    
                # Printing the labels for the axes
                if units is None:
                    axs[i,j].set_ylabel(titles[i],fontsize=fs)
                else:
                    if units[i] == '':
                        axs[i,j].set_ylabel(titles[i],fontsize=fs)
                    else:
                        axs[i,j].set_ylabel(titles[i]+' '+units[i],fontsize=fs)
            else:
                axs[i,j].set_yticks([])

            # Add the ticks and the axis labels where necessary on the y axis on the x axis 
            if i == dimension-1:
                axs[i,j].set_xticks(xticks)

                # Rounding the ticklabels and printing them
                roundx = np.log10(np.abs(xticks[1]-xticks[0]))
                if roundx>=0.5:
                    axs[i,j].set_xticklabels([int(i) for i in xticks],fontsize=fs,rotation=45,ha='right')
                else:
                    axs[i,j].set_xticklabels(np.round(xticks,int(-np.floor(roundx-0.5))),fontsize=fs,rotation=45,ha='right')
                
                # Printing the labels for the axes
                if units is None:
                    axs[i,j].set_xlabel(titles[j],fontsize=fs)
                else:
                    if units[j] == '':
                        axs[i,j].set_xlabel(titles[j],fontsize=fs)
                    else:
                        axs[i,j].set_xlabel(titles[j]+' '+units[j],fontsize=fs)
            else:
                axs[i,j].set_xticks([])

    # If wanted plot the true values in a table
    if add_table:
        ax_table = fig.add_subplot(111, frameon =False)
        ax_table.axes.get_xaxis().set_visible(False)
        ax_table.axes.get_yaxis().set_visible(False)
            
        # Add a table at the bottom of the axes
        the_table = ax_table.table(cellText=table,rowLabels=rows,cellLoc='center',colColours=colours,colLabels=columns,bbox=(0.8, 1-dimension*0.4/17, 0.2, dimension*0.4/17))
        the_table.set_fontsize(1.4*(dimension/17)*fs)
        the_table.scale(1, 3)


    

    # Set all titles at a uniform distance from the subplots
    fig.align_ylabels(axs[:, 0])
    fig.align_xlabels(axs[-1,:])
    return fig, axs





# Routine for plotting the 1d posteriors
def Posterior(data,title,units=None,truth=None,quantiles1d = [0.16, 0.5, 0.84], 
                color='k',color_truth='C3',bins=50,lw = 2,ax=None,histtype='stepfilled',alpha=0.5,hatch=None,ULU = False, ULU_lim=[-0.15,0.75]):

    # Start of plotting routine
    ax_arg = ax
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(2.5,2.5))
    else:
        pass

    # Plot the 1d histogram for each retrieved parameter on the diagonal.
    # Plot the 1d histogram for each retrieved parameter on the diagonal.
    if ULU:
        print(truth)
        h = np.histogram(data,density=True,bins=bins,range = (ULU_lim[0],0))
        h2 = np.histogram(np.log10(1-10**(np.arange(-7,0,0.000001))),density=True,bins=h[1])
        h = (h[0]/h2[0],h[1])
        h = ax.hist(h[1][: -1],h[1], weights = sp.ndimage.filters.gaussian_filter(h[0], [ULU_lim[1]], mode='constant'),histtype=histtype,color=color,density=True,alpha=alpha,hatch=hatch)
    else:
        h = ax.hist(data,histtype=histtype,color=color,alpha=alpha,density=True,bins=bins,hatch=hatch)

    # Define the limits of the plot and remove the yticks
    if ax is None:
        xlim = [h[1][0],h[1][-1]]
        ax.set_xlim(xlim)
        ylim = [0,1.1*np.max(h[0])]
        ax.set_ylim(ylim)
        ax.set_yticks([])

        # Generating the title for the plot
        if units is None or '':
            title = title
        else:
            title = title+' '+units

        # Plotting the secified quantiles
        if quantiles1d is not None:
            if ULU:
                h_cumulative = np.array([sum(h[0][:i+1])/sum(h[0]) for i in range(len(h[0]))])
                ind = [np.min(np.where(h_cumulative>=quantiles1d[i]))for i in range(len(quantiles1d))]
                q = ((h[1][1:]+h[1][:-1])/2)[ind]
            else:
                q = [np.quantile(data,ind) for ind in quantiles1d]
            ax.vlines(q,ax.get_ylim()[0],ax.get_ylim()[1],colors='k', ls='--')

            # Round q and print the retrieved value above the histogram plot
            round = min(np.log10(abs(q[2]-q[1])),np.log10(abs(q[0]-q[1])))
            if round>=0.5:
                ax.set_title(title + ' = ' + str(int(q[1]))+r' $_{\,'+str(int(q[0]-q[1]))+r'}^{\,+'+str(int(q[2]-q[1]))+r'}$')
            else:
                ax.set_title(title + ' = ' + str(np.round(q[1],int(-np.floor(round-0.5))))+r' $_{\,'+\
                        str(np.round(q[0]-q[1],int(-np.floor(round-0.5))))+r'}^{\,+'+\
                        str(np.round(q[2]-q[1],int(-np.floor(round-0.5))))+r'}$')

    # Plot the ground truth values if known
    if not truth is None:
        ax.plot([truth,truth],ax.get_ylim(),color=color_truth,linestyle = ':')

    if ax is None:
        # Setting 4 even ticks over the range defined by the limits of the subplot
        xticks = [(1-pos)*xlim[0]+pos*xlim[1] for pos in [0.2,0.4,0.6,0.8]]
        ax.set_xticks(xticks)

        # Rounding the ticklabels and printing them
        roundx = np.log10(np.abs(xticks[1]-xticks[0]))
        if roundx>=0.5:
            ax.set_xticklabels([int(i) for i in xticks],rotation=45,ha='right')
        else:
            ax.set_xticklabels(np.round(xticks,int(-np.floor(roundx-0.5))),rotation=45,ha='right')
                
        return fig, ax
    else:
        return ax, h



































# Routine for generating corner plots
def Corner_new(data,titles,units=None,truths=None,dimension=None,quantiles1d = [0.16, 0.5, 0.84], 
            quantiles2d=[0.02,0.15,0.25,0.35,0.65,0.75,0.85,0.98],color='k',color_truth='C3',bins=50,add_table = False,ULU=[],ULU_lim=[-0.15,0.75]):
    
    # Find the dimension of the corner plot.
    if dimension is None:
        dimension=np.shape(data)[1]

    # Generate colorlevels for the different quantiles
    color_levels, level_thresholds, N_levels = rp_col.color_levels(color,quantiles2d)

    if add_table:
        table = []
        columns = ('True', 'Retrieved')
        rows = titles
        colours = ['white','white']

    # Start of plotting routine
    fig, axs = plt.subplots(dimension, dimension,figsize=(dimension*2.5,dimension*2.5))
    fig.subplots_adjust(hspace=0.0,wspace=0.0)
    fs = 18
    
    # Iterate over the equal weighted posteriors of all retrieved parameters.
    # Diagonal histogram plots
    for i in range(dimension):
        # Plot the 1d histogram for each retrieved parameter on the diagonal.
        if i in ULU:  
            h = np.histogram(data[:,i],density=True,bins=bins,range = (ULU_lim[0],0))
            h2 = np.histogram(np.log10(1-10**(np.arange(-7,0,0.000001))),density=True,bins=h[1])
            h = (h[0]/h2[0],h[1])
            h = axs[i,i].hist(h[1][: -1],h[1], weights = sp.ndimage.filters.gaussian_filter(h[0], [ULU_lim[1]], mode='constant'),histtype='stepfilled',color=color_levels[1],density=True)
        else:
            h = axs[i,i].hist(data[:,i],histtype='stepfilled',color=color_levels[1],density=True,bins=bins)

        # Define the limits of the plot and remove the yticks
        axs[i,i].set_xlim([h[1][0],h[1][-1]])
        axs[i,i].set_ylim([0,1.1*np.max(h[0])])
        axs[i,i].set_yticks([])

        # Plotting the secified quantiles
        if quantiles1d is not None:
            if i in ULU:
                h_cumulative = np.array([sum(h[0][:i+1])/sum(h[0]) for i in range(len(h[0]))])
                ind = [np.min(np.where(h_cumulative>=quantiles1d[i]))for i in range(len(quantiles1d))]
                q = ((h[1][1:]+h[1][:-1])/2)[ind]
            else:
                q = [np.quantile(data[:,i],ind) for ind in quantiles1d]
            axs[i,i].vlines(q,axs[i,i].get_ylim()[0],axs[i,i].get_ylim()[1],colors='k', ls='--')

            # Round q and print the retrieved value above the histogram plot
            round = min(np.log10(abs(q[2]-q[1])),np.log10(abs(q[0]-q[1])))
            if round>=0.5:
                if add_table:
                    table.append([str(int(truths[i])),str(int(q[1]))+r' $_{\,'+str(int(q[0]-q[1]))+r'}^{\,+'+str(int(q[2]-q[1]))+r'}$'])
                else:
                    axs[i,i].set_title(str(int(q[1]))+r' $_{\,'+str(int(q[0]-q[1]))+r'}^{\,+'+str(int(q[2]-q[1]))+r'}$',fontsize=fs)
            else:
                if add_table:
                    table.append([str(np.round(truths[i],int(-np.floor(round-0.5)))),
                            str(np.round(q[1],int(-np.floor(round-0.5))))+r' $_{\,'+\
                            str(np.round(q[0]-q[1],int(-np.floor(round-0.5))))+r'}^{\,+'+\
                            str(np.round(q[2]-q[1],int(-np.floor(round-0.5))))+r'}$'])
                else:
                    axs[i,i].set_title(str(np.round(q[1],int(-np.floor(round-0.5))))+r' $_{\,'+\
                            str(np.round(q[0]-q[1],int(-np.floor(round-0.5))))+r'}^{\,+'+\
                            str(np.round(q[2]-q[1],int(-np.floor(round-0.5))))+r'}$',fontsize=fs)

        # Plot the ground truth values if known
        if not truths[i] is None:
            axs[i,i].plot([truths[i],truths[i]],axs[i,i].get_ylim(),color=color_truth,linestyle = ':',linewidth = 2)
    
    # 2d histograms plotted below the diagonal
    H_fac = np.shape(data[0,:])[0]*[[]]
    for i in range(dimension):
        for j in range(dimension):
            # Find the axis boundaries and set the x limits
            ylim = axs[i,i].get_xlim()
            xlim = axs[j,j].get_xlim()

            # For all subplots below the Diagonal
            if i > j:               
                # Plot the 2d histograms between different parameters to show correlations between the parameters                
                Z,X,Y=np.histogram2d(data[:,j],data[:,i],bins=bins,range = [list(xlim),list(ylim)])

                if i in ULU:
                    h = np.histogram(np.log10(1-10**(np.arange(-7,0,0.000001))),density=True,bins=Y)
                    H_fac[i]=1/h[0]
                    Z = Z/(h[0][None,:])

                    axs[j,j].clear()
                    h_new = axs[j,j].hist(X[: -1],X,weights = sp.ndimage.filters.gaussian_filter(np.sum(Z,axis = 1),[ULU_lim[1]], mode='reflect'),histtype='stepfilled',color=color_levels[1],density=True)
                    #axs[j,j].hist(data[:,j],bins = X,histtype='step',color='red',density=True)
                    fac = (h_new[0]/sp.ndimage.filters.gaussian_filter(np.histogram(data[:,j],density=True,bins=X)[0],[ULU_lim[1]], mode='reflect'))
                    fac[np.where(fac==np.inf)]=0
                    fac[np.where(fac==np.nan)]=0
                    H_fac[j] = fac

                    axs[j,j].set_xlim([h_new[1][0],h_new[1][-1]])
                    axs[j,j].set_ylim([0,1.1*np.max(h_new[0])])
                    axs[j,j].set_yticks([])

                    h_cumulative = np.array([sum(h_new[0][:i+1])/sum(h_new[0]) for i in range(len(h_new[0]))])
                    ind = [np.min(np.where(h_cumulative>=quantiles1d[i]))for i in range(len(quantiles1d))]
                    q = ((h_new[1][1:]+h_new[1][:-1])/2)[ind]
                    axs[j,j].vlines(q,axs[j,j].get_ylim()[0],axs[j,j].get_ylim()[1],colors='k', ls='--')

                    if not truths[j] is None:
                        axs[j,j].plot([truths[j],truths[j]],axs[j,j].get_ylim(),color=color_truth,linestyle = ':',linewidth = 2)

                    # Round q and print the retrieved value above the histogram plot
                    round = min(np.log10(abs(q[2]-q[1])),np.log10(abs(q[0]-q[1])))
                    if round>=0.5:
                        if add_table:
                            table[j] = [str(int(truths[j])),str(int(q[1]))+r' $_{\,'+str(int(q[0]-q[1]))+r'}^{\,+'+str(int(q[2]-q[1]))+r'}$']
                        else:
                            axs[j,j].set_title(str(int(q[1]))+r' $_{\,'+str(int(q[0]-q[1]))+r'}^{\,+'+str(int(q[2]-q[1]))+r'}$',fontsize=fs)
                    else:
                        if add_table:
                            table[j] = [str(np.round(truths[j],int(-np.floor(round-0.5)))),
                                    str(np.round(q[1],int(-np.floor(round-0.5))))+r' $_{\,'+\
                                    str(np.round(q[0]-q[1],int(-np.floor(round-0.5))))+r'}^{\,+'+\
                                    str(np.round(q[2]-q[1],int(-np.floor(round-0.5))))+r'}$']
                        else:
                            axs[j,j].set_title(str(np.round(q[1],int(-np.floor(round-0.5))))+r' $_{\,'+\
                                    str(np.round(q[0]-q[1],int(-np.floor(round-0.5))))+r'}^{\,+'+\
                                    str(np.round(q[2]-q[1],int(-np.floor(round-0.5))))+r'}$',fontsize=fs)

                if j in ULU:
                    h = np.histogram(np.log10(1-10**(np.arange(-7,0,0.000001))),density=True,bins=X)
                    Z = Z/(h[0][:,None])

                    axs[i,i].clear()
                    h_new = axs[i,i].hist(Y[: -1],Y,weights = sp.ndimage.filters.gaussian_filter(np.sum(Z,axis = 0),[ULU_lim[1]/2], mode='reflect'),histtype='stepfilled',color=color_levels[1],density=True)
                    #axs[i,i].hist(data[:,i],bins = Y,histtype='step',color='red',density=True)

                    fac = (h_new[0]/sp.ndimage.filters.gaussian_filter(np.histogram(data[:,i],density=True,bins=Y)[0],[ULU_lim[1]/2], mode='reflect'))
                    fac[np.where(fac==np.inf)]=0
                    fac[np.where(fac==np.nan)]=0
                    H_fac[i] = fac

                    axs[i,i].set_xlim([h_new[1][0],h_new[1][-1]])
                    axs[i,i].set_ylim([0,1.1*np.max(h_new[0])])
                    axs[i,i].set_yticks([])
                    h_cumulative = np.array([sum(h_new[0][:i+1])/sum(h_new[0]) for i in range(len(h_new[0]))])
                    ind = [np.min(np.where(h_cumulative>=quantiles1d[i]))for i in range(len(quantiles1d))]
                    q = ((h_new[1][1:]+h_new[1][:-1])/2)[ind]
                    axs[i,i].vlines(q,axs[i,i].get_ylim()[0],axs[i,i].get_ylim()[1],colors='k', ls='--')

                    if not truths[i] is None:
                        axs[i,i].plot([truths[i],truths[i]],axs[i,i].get_ylim(),color=color_truth,linestyle = ':',linewidth = 2)

                    # Round q and print the retrieved value above the histogram plot
                    round = min(np.log10(abs(q[2]-q[1])),np.log10(abs(q[0]-q[1])))
                    if round>=0.5:
                        if add_table:
                            table[i] = [str(int(truths[i])),str(int(q[1]))+r' $_{\,'+str(int(q[0]-q[1]))+r'}^{\,+'+str(int(q[2]-q[1]))+r'}$']
                        else:
                            axs[i,i].set_title(str(int(q[1]))+r' $_{\,'+str(int(q[0]-q[1]))+r'}^{\,+'+str(int(q[2]-q[1]))+r'}$',fontsize=fs)
                    else:
                        if add_table:
                            table[i] = [str(np.round(truths[i],int(-np.floor(round-0.5)))),
                                    str(np.round(q[1],int(-np.floor(round-0.5))))+r' $_{\,'+\
                                    str(np.round(q[0]-q[1],int(-np.floor(round-0.5))))+r'}^{\,+'+\
                                    str(np.round(q[2]-q[1],int(-np.floor(round-0.5))))+r'}$']
                        else:
                            axs[i,i].set_title(str(np.round(q[1],int(-np.floor(round-0.5))))+r' $_{\,'+\
                                    str(np.round(q[0]-q[1],int(-np.floor(round-0.5))))+r'}^{\,+'+\
                                    str(np.round(q[2]-q[1],int(-np.floor(round-0.5))))+r'}$',fontsize=fs)

    for i in range(dimension):
        for j in range(dimension):
            # Find the axis boundaries and set the x limits
            ylim = axs[i,i].get_xlim()
            xlim = axs[j,j].get_xlim()

            # Setting 4 even ticks over the range defined by the limits of the subplot
            yticks = [(1-pos)*ylim[0]+pos*ylim[1] for pos in [0.2,0.4,0.6,0.8]]
            xticks = [(1-pos)*xlim[0]+pos*xlim[1] for pos in [0.2,0.4,0.6,0.8]]

            # For all subplots below the Diagonal
            if i > j:
                # Plot the local truth values if provided
                
                if not truths[j] is None:
                    axs[i,j].plot([truths[j],truths[j]],[-10000,10000],color=color_truth,linestyle = ':',linewidth = 2)
                if not truths[i] is None:
                    axs[i,j].plot([-10000,10000],[truths[i],truths[i]],color=color_truth,linestyle = ':',linewidth = 2)
                if not ((truths[j] is None) or (truths[i] is None)):
                    axs[i,j].plot(truths[j],truths[i],color=color_truth,marker='o',markersize=8)
                
                # Plot the 2d histograms between different parameters to show correlations between the parameters                
                Z,X,Y=np.histogram2d(data[:,j],data[:,i],bins=bins,range = [list(xlim),list(ylim)])

                Z = Z*(H_fac[i][None,:])*(H_fac[j][:,None])

                Z = sp.ndimage.filters.gaussian_filter(Z, [1.5*ULU_lim[1],1.5*ULU_lim[1]], mode='reflect')


                map, norm, levels = rp_col.color_map(Z,color_levels,level_thresholds)
                axs[i,j].contourf((X[:-1]+X[1:])/2,(Y[:-1]+Y[1:])/2,Z.T,cmap=map,norm=norm,levels=np.array(levels))

                # Setting the limit of the x,y-axis
                axs[i,j].set_ylim(ylim)
                axs[i,j].set_xlim(xlim)

            # No Subplots show above the diagonal
            elif i<j:
                axs[i,j].axis('off')
    
            # Add the ticks and the axis labels where necessary on the y axis               
            if j == 0 and i!=0:
                axs[i,j].set_yticks(yticks)

                # Rounding the ticklabels and printing them
                roundy = np.log10(np.abs(yticks[1]-yticks[0]))
                if roundy>=0.5:
                    axs[i,j].set_yticklabels([int(i) for i in yticks],fontsize=fs,rotation=45)
                else:
                    axs[i,j].set_yticklabels(np.round(yticks,int(-np.floor(roundy-0.5))),fontsize=fs,rotation=45)
                    
                # Printing the labels for the axes
                if units is None:
                    axs[i,j].set_ylabel(titles[i],fontsize=fs)
                else:
                    if units[i] == '':
                        axs[i,j].set_ylabel(titles[i],fontsize=fs)
                    else:
                        axs[i,j].set_ylabel(titles[i]+' '+units[i],fontsize=fs)
            else:
                axs[i,j].set_yticks([])

            # Add the ticks and the axis labels where necessary on the y axis on the x axis 
            if i == dimension-1:
                axs[i,j].set_xticks(xticks)

                # Rounding the ticklabels and printing them
                roundx = np.log10(np.abs(xticks[1]-xticks[0]))
                if roundx>=0.5:
                    axs[i,j].set_xticklabels([int(i) for i in xticks],fontsize=fs,rotation=45,ha='right')
                else:
                    axs[i,j].set_xticklabels(np.round(xticks,int(-np.floor(roundx-0.5))),fontsize=fs,rotation=45,ha='right')
                
                # Printing the labels for the axes
                if units is None:
                    axs[i,j].set_xlabel(titles[j],fontsize=fs)
                else:
                    if units[j] == '':
                        axs[i,j].set_xlabel(titles[j],fontsize=fs)
                    else:
                        axs[i,j].set_xlabel(titles[j]+' '+units[j],fontsize=fs)
            else:
                axs[i,j].set_xticks([])

    # If wanted plot the true values in a table
    if add_table:
        ax_table = fig.add_subplot(111, frameon =False)
        ax_table.axes.get_xaxis().set_visible(False)
        ax_table.axes.get_yaxis().set_visible(False)
            
        # Add a table at the bottom of the axes
        the_table = ax_table.table(cellText=table,rowLabels=rows,cellLoc='center',colColours=colours,colLabels=columns,bbox=(0.8, 1-dimension*0.4/17, 0.2, dimension*0.4/17))
        the_table.set_fontsize(1.4*(dimension/17)*fs)
        the_table.scale(1, 3)


    

    # Set all titles at a uniform distance from the subplots
    fig.align_ylabels(axs[:, 0])
    fig.align_xlabels(axs[-1,:])
    return fig, axs
