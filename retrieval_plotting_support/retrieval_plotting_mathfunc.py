__author__ = "Konrad"
__copyright__ = "Copyright 2022, Konrad"
__maintainer__ = "Björn S. Konrad"
__email__ = "konradb@phys.ethz.ch"
__status__ = "Development"

# Standard Libraries
import numpy as np






def Logistic_Function(self,x,L,k,hm):
    return L/(1+np.exp(-k*(x-hm)))



def Inverse_Logistic_Function(self,y,L,k,hm):
    return hm-np.log(L/y-1)/k
