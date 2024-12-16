# Standard Libraries
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
import matplotlib.text as mtext
from matplotlib.collections import PatchCollection



class MulticolorPatch(object):
    """
    Initialize a MulticolorPatch object with specified colors and transparency.

    This class represents a multicolor gradient patch that can be used in legends to display a 
    gradient of colors with a specified level of transparency.

    Parameters
    ----------
    colors : list of str or tuple
        A list of colors to be used in the patch. Each color can be specified as a string 
        (e.g., 'red', '#FF0000') or as an RGBA tuple (e.g., (1, 0, 0, 1)).

    alpha : float
        The transparency level of the patch, ranging from 0 (completely transparent) to 
        1 (completely opaque).

    Attributes
    ----------
    colors : list of str or tuple
        The list of colors used for the gradient in the patch.
    
    alpha : float
        The transparency level of the patch.

    Notes
    -----
    - This class allows the creation of a color gradient with specified transparency, which can be used in plotting libraries
      like Matplotlib for creating customized legends or other visual representations where a gradient is needed.
    """
    
    def __init__(self,
                 colors,
                 alpha):
        self.colors = colors
        self.alpha = alpha



# define a handler for the MulticolorPatch object
class MulticolorPatchHandler(object):
    """
    Handler for creating a legend artist for the MulticolorPatch object.

    This class defines a custom handler for the `MulticolorPatch` object, which can be used to 
    create a legend entry with a gradient of colors and transparency levels.

    Methods
    -------
    legend_artist(legend, orig_handle, fontsize, handlebox):
        Creates the legend artist, representing a gradient of colors with varying transparency, 
        based on the original `MulticolorPatch` object.
    """

    def legend_artist(self,
                      legend,
                      orig_handle,
                      fontsize,
                      handlebox):
        """
        Create the legend artist for the MulticolorPatch.

        This method creates a custom legend entry, which consists of a gradient of color patches 
        representing the `MulticolorPatch` object in the legend.

        :param legend: The `matplotlib.legend.Legend` instance that holds the legend information.
        :type legend: `matplotlib.legend.Legend`

        :param orig_handle: The original handle, which is a `MulticolorPatch` object containing the colors 
                             and transparency levels for the gradient.
        :type orig_handle: `MulticolorPatch`

        :param fontsize: The font size for the legend text. It is currently unused in this method.
        :type fontsize: int

        :param handlebox: The `matplotlib.legend.HandlerBase.HandleBox` object, which provides the 
                          dimensions for the legend entry.
        :type handlebox: `matplotlib.legend.HandlerBase.HandleBox`

        :return: A collection of patch objects that form the custom legend entry.
        :rtype: `matplotlib.collections.PatchCollection`

        :notes:
            - This method creates a series of rectangles to represent the gradient of colors from the `MulticolorPatch` 
              object. The transparency levels are also taken into account.
            - The method uses `matplotlib.patches.Rectangle` to create each color patch and `matplotlib.collections.PatchCollection` 
              to group them together for the legend.
            - The returned `PatchCollection` can then be added to the legend in a Matplotlib plot.
        """

        width, height = handlebox.width, handlebox.height
        patches = []
        patches.append(plt.Rectangle([ - handlebox.xdescent, 
                          -handlebox.ydescent],
                           width,
                           height, 
                           facecolor='white',
                           alpha =  1,
                           edgecolor=(1,1,1,1)))
        for i, c in enumerate(orig_handle.colors):
            patches.append(plt.Rectangle([width/len(orig_handle.colors) * i - handlebox.xdescent, 
                                          -handlebox.ydescent],
                           width / len(orig_handle.colors),
                           height, 
                           facecolor=c,
                           alpha =  orig_handle.alpha[i]))
        patches.append(plt.Rectangle([ - handlebox.xdescent, 
                -handlebox.ydescent],
                width,
                height, 
                facecolor='none',
                alpha =  1,
                edgecolor=(0,0,0,1)))

        patch = PatchCollection(patches,match_original=True)

        handlebox.add_artist(patch)
        return patch



class Handles(HandlerBase):
    """
    Handler class to create custom legend artists for various types of data representations.

    This class defines how different data representations should appear in a legend, using custom 
    markers and line styles. It supports multiple types of handles, including CloudTopPressure, 
    SurfacePressure, ErrorBar, SNR, and others, each with their own visual representation in the legend.

    Parameters
    ----------
    text_props : dict, optional
        A dictionary of text properties (e.g., font size, style) that can be applied to text in the legend.
        If not provided, defaults to an empty dictionary.

    Methods
    -------
    create_artists(legend, orig_handle, x0, y0, width, height, fontsize, trans)
        Creates custom legend artists (lines, markers, etc.) based on the original handle type.
        This method handles the visual representation of the data in the legend.
    """

    def __init__(self,
                 text_props=None):
        """
        Initialize a Handles object to create custom legend artists.

        This constructor allows setting optional text properties for legend labels, which can be applied 
        during the creation of artists for custom legend entries.

        :param text_props: A dictionary of text properties to customize text appearance in the legend.
                           Defaults to an empty dictionary if not provided.
        :type text_props: dict, optional
        """

        self.text_props = text_props or {}
        super(Handles, self).__init__()

    def create_artists(self,
                       legend,
                       orig_handle,
                       x0,
                       y0,
                       width,
                       height,
                       fontsize,
                       trans):
        """
        Create custom legend artists (lines, markers, etc.) based on the provided `orig_handle`.

        This method generates the visual elements to represent various types of data in the legend, such as 
        markers, lines, and patterns, for specific handle types like 'CloudTopPressure', 'SurfacePressure', 
        'ErrorBar', and others. Each type is customized with specific line styles, markers, and colors.

        :param legend: The legend instance to which the artists are being added.
        :param orig_handle: The original handle (identifier) for the data type being represented.
                            For example, it could be a string like 'CloudTopPressure', 'SNR5', etc.
        :param x0: The x-coordinate for the legend position.
        :param y0: The y-coordinate for the legend position.
        :param width: The width of the legend entry.
        :param height: The height of the legend entry.
        :param fontsize: The font size for the legend text.
        :param trans: The transformation applied to the legend (e.g., scale or rotation).
        
        :return: A list of matplotlib Line2D objects or other artists that represent the custom legend entry.
        :rtype: list of matplotlib.artist.Artist
        """

        if orig_handle == 'CloudTopPressure':
            l1 = plt.Line2D([0.5*width], [0.5*height], marker = 's', ms=2.5,color='C4')
            l2 = plt.Line2D([x0+0.5*width-0.4*height,x0+0.5*width+0.4*height], [0.5*height,0.5*height], linestyle='-', color='C4',lw=1)
            l3 = plt.Line2D([0.5*width,0.5*width], [0.1*height,0.9*height], linestyle='-', color='C4',lw=1)
            return [l1, l2 ,l3]
        if orig_handle == 'SurfacePressure':
            l1 = plt.Line2D([0.5*width], [0.5*height], marker = 'o', ms=2.5,color='C3')
            l2 = plt.Line2D([x0+0.5*width-0.4*height,x0+0.5*width+0.4*height], [0.5*height,0.5*height], linestyle='-', color='C3',lw=1)
            l3 = plt.Line2D([0.5*width,0.5*width], [0.1*height,0.9*height], linestyle='-', color='C3',lw=1)
            return [l1, l2 ,l3]
        if orig_handle == 'ErrorBar':
            l1 = plt.Line2D([width/2], [0.5*height], marker = 'o', ms=10,color='w',markeredgecolor='k',alpha=0.2)
            l2 = plt.Line2D([x0,x0], [0.2*height,0.8*height], linestyle='-', color='k',lw=2)
            l3 = plt.Line2D([y0+width,y0+width], [0.2*height,0.8*height], linestyle='-', color='k',lw=2)
            l4 = plt.Line2D([x0,y0+width], [0.5*height,0.5*height], color='k',lw=2)
            return [l1, l2 ,l3, l4]
        elif orig_handle == 'Limit':
            l1 = plt.Line2D([2*width/3], [0.5*height], marker = 'o', ms=10,color='w',markeredgecolor='k',alpha=0.2)
            l2 = plt.Line2D([x0+width/3,x0+width/3], [0.2*height,0.8*height], linestyle='-', color='k',lw=2)
            l3 = plt.Line2D([y0+width,y0+width], [0.2*height,0.8*height], linestyle='-', color='k',lw=2)
            l4 = plt.Line2D([x0+width/3,y0+width], [0.5*height,0.5*height], color='k',lw=2)
            l5 = plt.Line2D([x0,y0+width/3], [0.5*height,0.5*height], color='k',ls=':',lw=2)
            l6 = plt.Line2D([x0+3*0.2*height,x0,x0+3*0.2*height], [0.2*height,0.5*height,0.8*height], color='k',ls='-',lw=2)
            return [l1, l2 ,l3, l4, l5, l6]
        elif orig_handle == 'UpperLimit':
            #l1 = plt.Line2D([width/2], [0.5*height], marker = 'o', ms=10,color='w',markeredgecolor='k',alpha=0.2)
            l2 = plt.Line2D([x0+width,x0+width], [0.2*height,0.8*height], linestyle='-', color='k',lw=2)
            l3 = plt.Line2D([x0,y0+width], [0.5*height,0.5*height], color='k',lw=2)
            l4 = plt.Line2D([x0+3*0.2*height,x0,x0+3*0.2*height], [0.2*height,0.5*height,0.8*height], color='k',ls='-',lw=2)
            return [l1, l2 ,l3, l4]
        elif orig_handle == 'LowerLimit':
            l1 = plt.Line2D([width/2], [0.5*height], marker = 'o', ms=10,color='w',markeredgecolor='k',alpha=0.2)
            l2 = plt.Line2D([x0+width,x0+width], [0.2*height,0.8*height], linestyle='-', color='k',lw=2)
            l3 = plt.Line2D([x0,y0+width], [0.5*height,0.5*height], color='k',lw=2)
            l4 = plt.Line2D([x0+3*0.2*height,x0,x0+3*0.2*height], [0.2*height,0.5*height,0.8*height], color='k',ls='-',lw=2)
            return [l1, l2 ,l3, l4]
        elif orig_handle == 'Unconstrained':
            l1 = plt.Line2D([width/2], [0.5*height],marker = 'o',ms=10,color='w',markeredgecolor='k',alpha=0.2)
            l2 = plt.Line2D([x0+width-3*0.2*height,x0+width,x0+width-3*0.2*height], [0.2*height,0.5*height,0.8*height], linestyle='-', color='k',lw=2)
            l3 = plt.Line2D([x0,y0+width], [0.5*height,0.5*height], color='k',lw=2,ls=':')
            l4 = plt.Line2D([x0+3*0.2*height,x0,x0+3*0.2*height], [0.2*height,0.5*height,0.8*height], color='k',ls='-',lw=2)
            return [l1, l2 ,l3, l4]
        elif orig_handle == 'SNR5':
            l1 = plt.Line2D([x0,x0+width], [0.5*height,0.5*height],lw=2,color='k',ls='-',alpha=0.2)
            l2 = plt.Line2D([width/2], [0.5*height],marker = 'o',ms=10,color='w',markeredgecolor='k')
            return [l1,l2]
        elif orig_handle == 'SNR10':
            l1 = plt.Line2D([x0,x0+width], [0.5*height,0.5*height],lw=2,color='k',ls='-',alpha=0.2)
            l2 = plt.Line2D([width/2], [0.5*height],marker = 's',ms=10,color='w',markeredgecolor='k')
            return [l1,l2]
        elif orig_handle == 'SNR15':
            l1 = plt.Line2D([x0,x0+width], [0.5*height,0.5*height],lw=2,color='k',ls='-',alpha=0.2)
            l2 = plt.Line2D([width/2], [0.5*height],marker = 'D',ms=10,color='w',markeredgecolor='k')
            return [l1,l2]
        elif orig_handle == 'SNR20':
            l1 = plt.Line2D([x0,x0+width], [0.5*height,0.5*height],lw=2,color='k',ls='-',alpha=0.2)
            l2 = plt.Line2D([width/2], [0.5*height],marker = 'v',ms=10,color='w',markeredgecolor='k')
            return [l1,l2]
        elif orig_handle == 'R20':
            l1 = plt.Line2D([width/2], [0.5*height],marker = 'o',ms=10,color='w',markeredgecolor='k',alpha=0.2)
            l2 = plt.Line2D([x0,x0+width], [0.5*height,0.5*height],lw=2,color='C3',ls='-')
            return [l1,l2]
        elif orig_handle == 'R35':
            l1 = plt.Line2D([width/2], [0.5*height],marker = 'o',ms=10,color='w',markeredgecolor='k',alpha=0.2)
            l2 = plt.Line2D([x0,x0+width], [0.5*height,0.5*height],lw=2,color='C2',ls='-')
            return [l1,l2]
        elif orig_handle == 'R50':
            l1 = plt.Line2D([width/2], [0.5*height],marker = 'o',ms=10,color='w',markeredgecolor='k',alpha=0.2)
            l2 = plt.Line2D([x0,x0+width], [0.5*height,0.5*height],lw=2,color='C0',ls='-')
            return [l1,l2]
        elif orig_handle == 'R100':
            l1 = plt.Line2D([width/2], [0.5*height],marker = 'o',ms=10,color='w',markeredgecolor='k',alpha=0.2)
            l2 = plt.Line2D([x0,x0+width], [0.5*height,0.5*height],lw=2,color='C1',ls='-')
            return [l1,l2]
        elif orig_handle == 'Prior':
            l1 = plt.Line2D([width/2], [0.5*height],marker = 'o',ms=10,color='w',markeredgecolor='k',alpha=0.2)
            l2 = plt.Line2D([x0,x0+width], [0.5*height,0.5*height],lw=2,color='Black',ls='-')
            return [l1,l2]
        elif orig_handle == 'Multiline':
            l1 = plt.Line2D([x0,x0+0.95*width], [0.0*height,0.0*height],lw=1,color='gray',ls=':')
            l2 = plt.Line2D([x0,x0+0.95*width], [1./3*height,1./3*height],lw=1,color='gray',ls='-.')
            l3 = plt.Line2D([x0,x0+0.95*width], [2./3*height,2./3*height],lw=1,color='gray',ls='--')
            l4 = plt.Line2D([x0+0.04*width,x0+0.91*width], [1*height,1*height],lw=1,color='gray',ls='-')
            return [l1,l2,l3,l4]
        else:
            title = mtext.Text(x0, y0, orig_handle + '',weight='bold', usetex=False, **self.text_props)
            return [title]
