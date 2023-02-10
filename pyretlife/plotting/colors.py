__author__ = "Konrad"
__copyright__ = "Copyright 2022, Konrad"
__maintainer__ = "Björn S. Konrad"
__email__ = "konradb@phys.ethz.ch"
__status__ = "Development"


# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import matplotlib.colors as col
import numpy as np


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------


def color_levels(color, quantiles):
    """
    Generate colorlevels for a set of provided quantiles and a base color.
    """

    rgba_color = col.to_rgba(color)

    # Percentage of points in quantile range quantiles[i] to quantiles[-i-1]
    level_thresholds = list(
        2 * np.array(quantiles[: int(len(quantiles) / 2)])
    ) + [1]
    N_levels = len(level_thresholds) - 1

    # Generate the color levels
    color_levels = np.ones((N_levels, 4))
    for rgb_ind in range(3):
        color_levels[:, rgb_ind] = np.array(
            [
                (i + 1) / N_levels * rgba_color[rgb_ind]
                + (N_levels - i - 1) / N_levels
                for i in range(N_levels)
            ]
        )

    # Return information
    return color_levels, level_thresholds, N_levels


def color_map(Z, color_levels, level_thresholds):
    """
    Generate colormap for a 2d histogram based on percentage levels.
    """

    # Choose all bins with non-zero entries and sort them in ascending order
    level_counts = np.sort(Z.flatten())[np.where(np.sort(Z.flatten()) > 0)]

    # Identify the bin where the level counts exceeds the threshold values
    sum_level_counts = 0
    threshold_ind = 0
    levels = []
    for i in range(len(level_counts)):
        sum_level_counts += level_counts[i]
        if (
            sum_level_counts / np.sum(level_counts)
            >= level_thresholds[threshold_ind]
        ):
            threshold_ind += 1
            levels += [level_counts[i]]

    # Boundary case where the maximum is reached before the last level
    while len(levels) <= np.shape(color_levels)[0]:
        levels += [levels[-1] * 10]

    # Generate the colormap
    cmap, norm = col.from_levels_and_colors(levels, color_levels)

    # Return the colormap
    return cmap, norm, levels


def uniform_color_map(color_map):
    """
    Generate uniform colormap.
    """

    # Convert to colors to RGB
    c1 = col.to_rgba(color_map[0])
    c2 = col.to_rgba(color_map[1])

    # Generate the levels
    levels = np.linspace(0, 1, 1000)

    # Generate the color levels
    color_levels = np.ones((999, 4))
    for rgb_ind in range(3):
        color_levels[:, rgb_ind] = (
            c1[rgb_ind] * (1 - levels[1:]) + c2[rgb_ind] * levels[:-1]
        )

    # Return the colormap
    return col.from_levels_and_colors(levels, color_levels)[0]
