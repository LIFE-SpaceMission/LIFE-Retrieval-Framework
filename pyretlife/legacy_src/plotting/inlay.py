__author__ = "Konrad"
__copyright__ = "Copyright 2022, Konrad"
__maintainer__ = "Björn S. Konrad"
__email__ = "konradb@phys.ethz.ch"
__status__ = "Development"


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

def position_inlay(loc_surface, figure, ax_arg, ax, h_cover=0.45, Off_h=0.02):
    bbox = ax.get_window_extent()
    width, height = bbox.width, bbox.height
    w_cover = h_cover * height / width
    Off_w = Off_h * height / width
    if loc_surface == "lower left":
        ax2 = ax.inset_axes([Off_w, Off_h, w_cover, h_cover])
    elif loc_surface == "lower right":
        ax2 = ax.inset_axes([1 - Off_w - w_cover, Off_h, w_cover, h_cover])
    elif loc_surface == "upper left":
        ax2 = ax.inset_axes([Off_w, 1 - Off_h - h_cover, w_cover, h_cover])
    elif loc_surface == "upper right":
        ax2 = ax.inset_axes(
            [1 - Off_w - w_cover, 1 - Off_h - h_cover, w_cover, h_cover]
        )
    else:
        raise ValueError(
            f'Invalid value "{loc_surface}" for loc_surface. Must be one of: '
            f"[lower left, lower right, upper left, upper right]"
        )
    return ax2


def axesticks_inlay(ax2, ax2_xlabel, ax2_ylabel, loc_surface):
    if loc_surface == "lower left":
        ax2.set_ylabel(ax2_ylabel, va="top", rotation=90)
        ax2.set_xlabel(ax2_xlabel, va="bottom")
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        ax2.xaxis.set_label_position("top")
        ax2.xaxis.tick_top()
    elif loc_surface == "lower right":
        ax2.set_ylabel(ax2_ylabel, rotation=90)
        ax2.set_xlabel(ax2_xlabel, va="bottom")
        ax2.xaxis.set_label_position("top")
        ax2.xaxis.tick_top()
    elif loc_surface == "upper left":
        ax2.set_ylabel(ax2_ylabel, va="top", rotation=90)
        ax2.set_xlabel(ax2_xlabel)
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
    elif loc_surface == "upper right":
        ax2.set_ylabel(ax2_ylabel, rotation=90)
        ax2.set_xlabel(ax2_xlabel)
