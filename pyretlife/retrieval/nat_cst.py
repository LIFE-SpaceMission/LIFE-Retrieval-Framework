from __future__ import print_function
import astropy.constants as anc
import numpy as np
import scipy.constants as snc

import os as os
import sys

# Natural constants
# Everything is in cgs!
# Defined constants
c = snc.c * 1e2
h = snc.h * 1e7
kB = snc.k * 1e7
nA = snc.N_A
e = snc.e * np.sqrt(1e9 / (4 * snc.pi * snc.epsilon_0))

# Measured constants
G = snc.G * 1e3
m_elec = snc.m_e * 1e3

# Derived exact constants
sigma = snc.sigma * 1e3
L0 = snc.physical_constants['Loschmidt constant (273.15 K, 101.325 kPa)'][0] * 1e-6
R = snc.R

# Units definitions
bar = 1e6
atm = snc.atm * 1e1
AU = snc.au * 1e2
pc = snc.parsec * 1e2
light_year = snc.light_year * 1e2
amu = snc.physical_constants['atomic mass constant'][0] * 1e3

# Astronomical constants
r_sun = anc.R_sun.cgs.value
r_jup = anc.R_jup.cgs.value
r_earth = anc.R_earth.cgs.value
m_sun = anc.M_sun.cgs.value
m_jup = anc.M_jup.cgs.value
m_earth = anc.M_earth.cgs.value
l_sun = anc.L_sun.cgs.value

r_jup_mean = 6.9911e9
s_earth = 1.3654e6  # erg.s-1.cm-2, source: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2010GL045777

# Molecular weights in amu
molecular_weight = {
    'H2O': 18.,
    'O2': 32.,
    'N2': 28.,
    'CH4': 16.,
    'CO2': 44.,
    'CO': 28.,
    'H2': 2.,
    'He': 4.
}

