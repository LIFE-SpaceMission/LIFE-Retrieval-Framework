"""
Read in configuration files.
"""
import os
import sys
import glob
import numpy as np
import subprocess
import astropy.units as u

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Union, Tuple
import warnings
import hashlib
import yaml

from deepdiff import DeepDiff

from pyretlife.retrieval.units import UnitsUtil


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

def __init__(self,input_file = 'config.ini',retrieval = True):
    '''
    This function reads the config.ini file and initializes all
    the variables. It also ensures that the run is not rewritten
    unintentionally.
    '''

    config = configparser.ConfigParser(inline_comment_prefixes=('#',))
    config.optionxform = str
    config.read(input_file, encoding=None)

    self.path_prt = config.get('PATHS', 'settings_pRT')
    self.path_opacity = config.get('PATHS', 'settings_opacity_database')
    self.path_multinest = config.get('PATHS', 'settings_multinest')
    sys.path.append(self.path_multinest)

    # If we are running a retrieval create the run directory and check that
    # we are not overwriting an existing run
    if retrieval:
        self.prefix = config.get('PREFIX', 'settings_prefix')+'/'
        if os.path.isdir(self.prefix):
            if 'input.ini' in  os.listdir(self.prefix):
                st = os.system('diff ' + input_file + ' ' +
                                self.prefix + 'input.ini')

                # Protecting from unintentional rewriting
                if st == 256:
                        sys.exit('ERROR! same prefix but different input files. ')
        else:
            try:
                os.mkdir(self.prefix)
            except OSError:
                pass
            os.system('cp '+ input_file + ' ' + self.prefix + 'input.ini')

    f = open(self.path_prt + '/petitRADTRANS/path.txt', "r")
    orig_path = f.read()

    if not orig_path == "#\n" + self.path_opacity:
        with open(self.path_prt + '/petitRADTRANS/path.txt', 'w+') as input_data:
            input_data.write("#\n" + self.path_opacity)

    self.log_top_pressure=-5 #-6
    self.config_file = config
    self.params = OrderedDict()
    self.knowns = OrderedDict()
        
    # Initialization of the standard settings
    self.settings = {}
    self.settings['directlight']=False
    self.settings['CIA']=False
    self.settings['clouds']='transparent'
    self.settings['moon']=False
    self.settings['scattering']=False
    self.settings['extra_lines'] =''
    # Import PRt
    sys.path.append(self.path_prt)
    os.environ['pRT_input_data_path'] = self.path_opacity
    self.rt = __import__('petitRADTRANS')
    self.nc = self.rt.nat_cst

    # Create a units object to enable unit conversions
    self.units = units.units_util(self.rt.nat_cst)



def read_var(self):
    '''
    This function reads the config.ini file and fills up the three
    dictionaries: settings, params, knowns.
    '''

    # Check if there are units defined by the user and add these to the unis
    if 'USER DEFINED UNITS' in self.config_file.sections():
        for (key, val) in self.config_file.items('USER DEFINED UNITS'):
            self.units.custom_unit(key,u.Quantity(val))

    # Read the sections of the config file
    for section in self.config_file.sections():
        if section != 'USER DEFINED UNITS':
            for (key, val) in self.config_file.items(section):
                if 'settings' in key:
                    self.settings[key[9:]] = val

                # Check if the first element is a letter (the priors)
                elif val[0].upper().isupper():
                    # Extract the units from the input string. If none are
                    # provided the standard input units are assumed.
                    input_unit, val = self.units.unit_extract(key,val)

                    # Extract the input values
                    input_prior = [u.Quantity(val[i]).value for i in range(1,3)]
                    #try:
                    input_truth = u.Quantity(val[4]).value if (len(val)>=5 and val[3] == 'T') else None
                    #except:
                    #    input_truth = None

                    # Convert the input to retrieval units
                    conv_unit = self.units.return_units(key,self.units.retrieval_units)
                    conv_truth, conv_prior = self.units.unit_conv(key,input_unit,conv_unit,input_truth,prior_type=val[0],input_prior=input_prior)
                        
                    # Store the retrieved parameters
                    param = {'prior_type': val[0],
                            'prior': conv_prior,
                            'input_prior': input_prior,
                            'truth': conv_truth,
                            'input_truth': input_truth,
                            'unit': conv_unit,
                            'input_unit': input_unit,
                            'type': section}
                    self.params[key] = param
                    
                # The parameters that are fixed during the retrieval
                else:
                    # Extract the units from the input string. If none are
                    # provided the standard input units are assumed.
                    input_unit, val = self.units.unit_extract(key,val)

                    # Extract the input values
                    input_truth = u.Quantity(val[0]).value

                    # Convert the input to retrieval units
                    conv_unit = self.units.return_units(key,self.units.retrieval_units)
                    conv_truth = self.units.unit_conv(key,input_unit,conv_unit,input_truth)

                    # Store the knowns
                    known = {'value': conv_truth,
                            'input_value': input_truth,
                            'unit': conv_unit,
                            'input_unit': input_unit,
                            'type': section}
                    self.knowns[key] = known

















def read_data(self,retrieval=True,result_dir=None):
    """
    Reads the input data, trims to the wavelength range of interest
    and converts the units to CGS.
    """

    self.instrument = {}

    for name in self.config_file['INPUT FILES'].keys():

        input_string = self.config_file.get('INPUT FILES', name)
            
        # Case handling for the retrieval plotting
        if not retrieval:
            if os.path.isfile(result_dir+'/input_'+input_string.split('/')[-1].split(' ')[0]):
                input_string = result_dir+'/input_'+input_string.split('/')[-1]
            else:
                input_string = result_dir+ '/input_spectrum.txt '+ ' '.join(input_string.split('/')[-1].split(' ')[1:])

        # Extract the Units from the config file and load the data. If non are provided the standard units are assumed.
        input_unit_wl, input_unit_flux, spectrum_dir = self.units.unit_spectrum_extract(input_string)
        input_data = np.genfromtxt(spectrum_dir)

        # Trim the spectra to the desired wl
        input_data = input_data[input_data[:, 0] >= (self.knowns['WMIN']['value']*self.knowns['WMIN']['unit']).to(input_unit_wl).value]
        input_data = input_data[input_data[:, 0] <= (self.knowns['WMAX']['value']*self.knowns['WMAX']['unit']).to(input_unit_wl).value]

        # Convert the Units to retrieval units
        conv_unit_wl = self.units.return_units('wavelength',self.units.retrieval_units)
        conv_unit_flux = self.units.return_units('flux',self.units.retrieval_units)
        conv_data = self.units.unit_spectrum_conv(name,[input_unit_wl,input_unit_flux],[conv_unit_wl,conv_unit_flux],input_data)

        # Store the spectra for each instrument
        self.instrument[name] = {'wl':conv_data[:, 0],
                                'flux':conv_data[:, 1],
                                'error':conv_data[:, 2],
                                'input_wl':input_data[:, 0],
                                'input_flux':input_data[:, 1],
                                'input_error':input_data[:, 2],
                                'unit_wl':conv_unit_wl,
                                'unit_flux':conv_unit_flux,
                                'input_unit_wl':input_unit_wl,
                                'input_unit_flux':input_unit_flux,}

        # If we are running retrievals copy the input spectra
        if retrieval:
            os.system('cp '+ spectrum_dir + ' ' + self.prefix + '/input_'+spectrum_dir.split('/')[-1])