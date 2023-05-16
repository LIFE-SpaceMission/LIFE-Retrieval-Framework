# Standard Libraries
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
import matplotlib.text as mtext
from matplotlib.collections import PatchCollection





# define an object that will be used by the legend
class MulticolorPatch(object):
    def __init__(self, colors,alpha):
        self.colors = colors
        self.alpha = alpha





# define a handler for the MulticolorPatch object
class MulticolorPatchHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
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
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(Handles, self).__init__()
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
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
