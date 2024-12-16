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

def add_inlay_plot(loc_surface,
                   ax,
                   h_cover=0.45,
                   Off_h=0.02):
    """
    Add an inset plot to a given axis at a specified location.

    This function creates an inset plot (an inlay plot) within a larger plot and positions it according to 
    the specified location. The inset plot is a smaller plot placed inside the original plot, and its size 
    and position can be customized based on the given parameters.

    :param loc_surface: The location of the inset plot relative to the main plot. Possible values are 
                        'lower left', 'lower right', 'upper left', and 'upper right'.
    :type loc_surface: str
    :param ax: The main matplotlib axis where the inset plot will be added.
    :type ax: matplotlib.axes.Axes
    :param h_cover: The height coverage of the inset plot as a fraction of the main plot height. Default is 0.45.
    :type h_cover: float, optional
    :param Off_h: The offset for the inset plot's vertical position, as a fraction of the main plot height. Default is 0.02.
    :type Off_h: float, optional

    :return: The axis object of the created inset plot.
    :rtype: matplotlib.axes.Axes
    """

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



def add_inlay_plot_labels(ax2,
                          ax2_xlabel,
                          ax2_ylabel,
                          loc_surface):
    """
    Add labels to the axes of an inset plot at a specified location.

    This function adds the x and y axis labels to the inset plot and adjusts their positions based on 
    the specified location of the inset plot within the main plot. It allows for customized positioning 
    of axis labels in different corners of the plot.

    :param ax2: The inset axis to which the labels will be applied.
    :type ax2: matplotlib.axes.Axes
    :param ax2_xlabel: The label for the x-axis of the inset plot.
    :type ax2_xlabel: str
    :param ax2_ylabel: The label for the y-axis of the inset plot.
    :type ax2_ylabel: str
    :param loc_surface: The location of the inset plot relative to the main plot. Possible values are 
                        'lower left', 'lower right', 'upper left', and 'upper right'.
    :type loc_surface: str

    :return: None
    :rtype: None
    """

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
        