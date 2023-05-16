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

# Standard Libraries
import matplotlib.colors as col
import matplotlib.pyplot as plt
import numpy as np
import sys

# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

def add_inlay_plot(loc_surface,figure,ax_arg,ax,h_cover=0.45,Off_h=0.02):
    bbox = ax.get_window_extent()
    width, height = bbox.width, bbox.height
    w_cover = h_cover*height/width
    Off_w = Off_h*height/width
    if loc_surface == 'lower left':
        ax2 = ax.inset_axes([Off_w, Off_h, w_cover, h_cover])
    elif loc_surface == 'lower right':
        ax2 = ax.inset_axes([1-Off_w-w_cover, Off_h,w_cover, h_cover])
    elif loc_surface == 'upper left':
        ax2 = ax.inset_axes([Off_w,1-Off_h-h_cover,w_cover, h_cover])
    elif loc_surface == 'upper right':
        ax2 = ax.inset_axes([1-Off_w-w_cover,1-Off_h-h_cover,w_cover, h_cover])
    else:
        sys.exit('ERROR: "'+str(loc_surface)+'" is invalid for the inlay position. Must be: [lower left, lower right, upper left, upper right]')
    return ax2



def add_inlay_plot_labels(ax2,ax2_xlabel,ax2_ylabel,loc_surface):
    if loc_surface == 'lower left':
        ax2.set_ylabel(ax2_ylabel,va='top',rotation = 90)
        ax2.set_xlabel(ax2_xlabel,va='bottom')
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        ax2.xaxis.set_label_position("top")
        ax2.xaxis.tick_top()
    elif loc_surface == 'lower right':
        ax2.set_ylabel(ax2_ylabel,rotation = 90)
        ax2.set_xlabel(ax2_xlabel,va='bottom')
        ax2.xaxis.set_label_position("top")
        ax2.xaxis.tick_top()
    elif loc_surface == 'upper left':
        ax2.set_ylabel(ax2_ylabel,va='top',rotation = 90)
        ax2.set_xlabel(ax2_xlabel)
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
    elif loc_surface == 'upper right':
        ax2.set_ylabel(ax2_ylabel,rotation = 90)
        ax2.set_xlabel(ax2_xlabel)
        