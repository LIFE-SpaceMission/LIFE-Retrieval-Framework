__author__ = "Konrad"
__copyright__ = "Copyright 2022, Konrad"
__maintainer__ = "Bjorn Konrad"
__email__ = "konradb@phys.ethz.ch"
__status__ = "Development"

import numpy as np

# Definitions of the posterior models
def Model_Flat(x,c):
    return np.ones_like(x)*c

def Model_SoftStep(x,a,b,c):
    return c / (1+np.exp(a*x+b))

def Inv_Model_SoftStep(y,a,b,c):
    return (np.log(c/y-1)-b)/a

def Model_upper_SoftStep(x,a,b,c):
    return c / (1+np.exp(-a*x+b))

def Inv_Model_upper_SoftStep(y,a,b,c):
    return (np.log(c/y-1)-b)/a
    
def Model_Gauss(x,h,m,s):
    return h/(np.sqrt(2*np.pi)*s)*np.exp(-1/2*((x-m)/s)**2)

def Model_SoftStepG(x,a,b,c,s,e):
    return (c+(e/(np.sqrt(2*np.pi)*s)*np.exp(-(x+b)**2/s**2)))/ (1+np.exp(a*(x+b)))

# Definition of the likelihood for comparison of different posterior models
def log_likelihood(theta,x,y,Model):
    model = Model(x,*theta)
    if np.sum(model) == 0:
        return -10**100
    return -0.5*np.sum((y-model)**2/0.01**2)
