__author__ = "Konrad"
__copyright__ = "Copyright 2022, Konrad"
__maintainer__ = "Björn S. Konrad"
__email__ = "konradb@phys.ethz.ch"
__status__ = "Development"

# Standard Libraries
import matplotlib.colors as col
import matplotlib.pyplot as plt
import numpy as np





def position_inlay(loc_surface,figure,ax_arg,ax):
    if ax_arg is None:
        if loc_surface == 'bottom left':
            ax2 = figure.add_axes([0.11, 0.12, 0.2, 0.4])
        elif loc_surface == 'bottom right':
            ax2 = figure.add_axes([1-0.11-0.2, 0.12, 0.2, 0.4])
        elif loc_surface == 'top left':
            ax2 = figure.add_axes([0.11, 1-0.12-0.4, 0.2, 0.4])
        elif loc_surface == 'top right':
            ax2 = figure.add_axes([1-0.12833-0.2, 1-0.12-0.2833, 0.212, 0.2833])
    else:
        if loc_surface == 'bottom left':
            ax2 = ax.inset_axes([0.02*2/3, 0.02, 0.5*2/3, 0.5])
        elif loc_surface == 'bottom right':
            ax2 = ax.inset_axes([1-0.5*2/3, 0.02, 1-0.02*2/3, 0.5])
        elif loc_surface == 'top left':
            ax2 = ax.inset_axes([0.02*2/3, 1-0.5, 0.5*2/3, 1-0.02])
        elif loc_surface == 'top right':
            ax2 = ax.inset_axes([1-0.5*2/3-0.02*2/3, 1-0.5-0.02,0.5*2/3, 0.5])
    return ax2



def axesticks_inlay(ax2,ax2_xlabel,ax2_ylabel,loc_surface):
    if loc_surface == 'bottom left':
        ax2.set_ylabel(ax2_ylabel,va='top',rotation = 90)
        ax2.set_xlabel(ax2_xlabel,va='bottom')
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        ax2.xaxis.set_label_position("top")
        ax2.xaxis.tick_top()
    elif loc_surface == 'bottom right':
        ax2.set_ylabel(ax2_ylabel,rotation = 90)
        ax2.set_xlabel(ax2_xlabel,va='bottom')
        ax2.xaxis.set_label_position("top")
        ax2.xaxis.tick_top()
    elif loc_surface == 'top left':
        ax2.set_ylabel(ax2_ylabel,va='top',rotation = 90)
        ax2.set_xlabel(ax2_xlabel)
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
    elif loc_surface == 'top right':
        ax2.set_ylabel(ax2_ylabel,rotation = 90)
        ax2.set_xlabel(ax2_xlabel)
        