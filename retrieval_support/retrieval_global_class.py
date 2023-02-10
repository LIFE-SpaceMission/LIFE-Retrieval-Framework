from __future__ import absolute_import, unicode_literals, print_function
import spectres as spectres
from collections import OrderedDict
import warnings
import json
import sys
import os
import numpy as np
import configparser
from retrieval_support import retrieval_priors as priors
from retrieval_support import retrieval_units as units
import scipy.ndimage as sci
import astropy.units as u

__author__ = "Alei, Konrad, Molliere, Quanz"
__copyright__ = "Copyright 2022, Alei, Konrad, Molliere, Quanz"
__maintainer__ = ",Björn S. Konrad, Eleonora Alei"
__email__ = "konradb@ethz.ch, elalei@phys.ethz.ch"
__status__ = "Development"

import os

os.environ["OMP_NUM_THREADS"] = "1"
warnings.simplefilter("ignore")


class globals:
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
                    st = os.system('diff config.ini ' +
                                self.prefix + 'input.ini')

                    # Protecting from unintentional rewriting
                    if st == 256:
                        sys.exit('ERROR! same prefix but different input files. ')
            else:
                try:
                    os.mkdir(self.prefix)
                except OSError:
                    pass
                os.system('cp config.ini ' + self.prefix + 'input.ini')

        f = open(self.path_prt + '/petitRADTRANS/path.txt', "r")
        orig_path = f.read()

        if not orig_path == "#\n" + self.path_opacity:
           with open(self.path_prt + '/petitRADTRANS/path.txt', 'w+') as input_data:
                input_data.write("#\n" + self.path_opacity)

        self.log_top_pressure=-6
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
                        input_truth = u.Quantity(val[4]).value if val[3] == 'T' else None
                        
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
        

    def check_temp_params(self):
        '''
        This function checks if all temperature variables necessary
        for the given parametrization are provided by the user. If not,
        it stops the run.
        '''

        input_pt = list(self.config_file['TEMPERATURE PARAMETERS'].keys())

        # check if all parameters are there:
        if self.settings['parametrization'] == 'polynomial':
            pt_params = ['a_'+str(i) for i in range(len(input_pt)-1)]
        elif self.settings['parametrization'] == 'vae_pt':
            pt_params = ['z_'+str(i+1) for i in range(len([input_pt[i] for i in range(len(input_pt)) if not 'settings' in input_pt[i]])-2)]
        elif self.settings['parametrization'] == 'guillot':
            pt_params = ['log_delta', 'log_gamma','t_int', 't_equ', 'log_p_trans', 'alpha']
        elif self.settings['parametrization'] == 'madhuseager':
            pt_params = ['T0','log_P1','log_P2','log_P3','alpha1','alpha2']
        elif self.settings['parametrization'] == 'mod_madhuseager':
            pt_params = ['T0','log_P1','log_P2','alpha1','alpha2']
        elif self.settings['parametrization'] == 'isothermal':
            pt_params = ['T_eq']
        elif self.settings['parametrization'] == 'input':
            pt_params = ['input_path']
        else:
            sys.exit('Unknown PT parametrization.')

        if not all(elem in input_pt for elem in pt_params):
            sys.exit('Missing one or more PT parameters/knowns. Make sure these exist:' + str(pt_params))


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



    def init_rt(self):
        """
        Initializes the rt_object given the wavelength range.
        """
        # add resolutions
        string = ''
        if self.settings['resolution'] != '1000':
            string = '_R_' + str(self.settings['resolution'])
        speciesres = []
        # add extra linelists (only if you need to read multiple files for
        # the same linelist (e.g. UV O3 linelists)
        for extra in self.settings['extra_lines'].split(','):
            speciesres.append(extra.strip() + string)
        # Line absorbers
        for key in self.params.keys():
            if self.params[key]['type'] == 'CHEMICAL COMPOSITION PARAMETERS':
                speciesres.append(key+string)
        for key in self.knowns.keys():
            if self.knowns[key]['type'] == 'CHEMICAL COMPOSITION PARAMETERS':
                speciesres.append(key+string)

        # check if lines are there (trick with resolution)
        species = list(set([s.split('_')[0] for s in speciesres]))
        line_species = os.listdir(self.path_opacity + '/opacities/lines/corr_k/')
        used_line_species = (list(set(speciesres) & set(line_species)))
        tot_mols =[s.split('_')[0] for s in used_line_species]

        # rayleigh_species
        rayleigh_species = ['H2', 'He', 'H2O',
                            'CO2', 'O2', 'N2', 'CO', 'CH4', 'N2']
        used_rayleigh_species = list(set(species) & set(rayleigh_species))
        tot_mols.extend(used_rayleigh_species)

        # CIA species
        used_cia_species = []
        if self.settings['CIA']=='True':
            continuum_opacities = os.listdir(
                self.path_opacity + '/opacities/continuum/CIA/')

            for cia in continuum_opacities:
                cia_components = cia.split('-')
                if len(cia_components) > 1:
                    if species.count(cia_components[0]) + species.count(cia_components[1]) == 2:
                        used_cia_species.append(cia)
                        tot_mols.append(cia_components[0])
                        tot_mols.append(cia_components[1])

        # Cloud species: Check if the opacity files are present and store them
        cloud_dict = {'a' : '/amorphous','c' : '/crystalline','m' : '/mie', 'd' : '/DHS'}
        used_cloud_species = []
        for key in self.params.keys():
            if self.params[key]['type'] == 'CLOUD PARAMETERS' and len(key.split("_"))==2:
                cloud_dir = str(self.path_opacity + '/opacities/continuum/clouds/' + key[:-6] +
                        '_c'+cloud_dict[key[-2]]+cloud_dict[key[-1]])
                if not os.path.exists(cloud_dir):
                    sys.exit('ERROR: No opacities found for the cloud species '+str(key)+'.')
                tot_mols.append(key.split("_", 1)[0])
                used_cloud_species.append(key)
        for key in self.knowns.keys():
            if self.knowns[key]['type'] == 'CLOUD PARAMETERS' and len(key.split("_"))==2:
                cloud_dir = str(self.path_opacity + '/opacities/continuum/clouds/' + key[:-6] +
                        '_c'+cloud_dict[key[-2]]+cloud_dict[key[-1]])
                if not os.path.exists(cloud_dir):
                    sys.exit('ERROR: No opacities found for the cloud species '+str(key)+'.')
                tot_mols.append(key.split("_", 1)[0])
                used_cloud_species.append(key)

        # if the vae_pt is selected initialize the pt profile model
        if self.settings['parametrization'] == 'vae_pt':
            from retrieval_support import retrieval_pt_vae as vae
            #try:
            self.vae_pt = vae.VAE_PT_Model(file_path=os.path.dirname(os.path.realpath(__file__))+'/vae_pt_models/'+self.settings['vae_net'],
                                                flow_file_path=os.path.dirname(os.path.realpath(__file__))+'/vae_pt_models/'+self.settings['flow_net'])
            #except:
            #    self.vae_pt = vae.VAE_PT_Model(file_path=os.path.dirname(os.path.realpath(__file__))+'/vae_pt_models/'+self.settings['vae_net'])                

        # PROTECTION FROM BAD INPUTS
        self.tot_mols = list(set(tot_mols))

        # update due to the new spectres version
        WLEN = [self.knowns['WMIN']['value'], self.knowns['WMAX']['value']]

        print('Used line species:\t\t' + str(used_line_species))
        print('Used rayleigh species:\t\t' + str(used_rayleigh_species))
        print('Used continuum opacities:\t' + str(used_cia_species))
        print('Used cloud species:\t\t' + str(used_cloud_species))
        print('Used species *in general*:\t' + str(self.tot_mols))
        print()
        # Read in the molecular weights database
        self.MMW_Storage = {}
        reader = np.loadtxt(self.path_opacity + "/opa_input_files/Molecular_Weights.txt",dtype='str')
        for i in range(len(reader[:,0])):
            self.MMW_Storage[reader[i,0]]=float(reader[i,1])

        # Initialize th pRT object and prevent printing of reading
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        ls = sorted(used_line_species)[::-1]
        self.rt_object = self.rt.Radtrans(line_species=ls, #sorted(used_line_species),
                                  rayleigh_species=sorted(used_rayleigh_species),
                                  continuum_opacities=sorted(used_cia_species),
                                  cloud_species=sorted(used_cloud_species),
                                  wlen_bords_micron=WLEN,
                                  mode='c-k',
                                  do_scat_emis=self.settings['scattering'])
        sys.stdout = old_stdout

        self.rt_object.setup_opa_structure(np.logspace(-6, 0, 100, base=10))


    def print_params(self):
        with open('%sparams.json' % self.prefix, 'w') as f:
            json.dump(list(self.params.keys()), f, indent=2)


    def Priors(self, cube):
        ccube = cube.copy()
        """
        Converts the unity cube to prior cube by recognizing the names
        of the Parameters and giving them identificators.
        """

        for par in self.params.keys():
            prior = self.params[par]['prior']
            key = list(self.params.keys()).index(par)

            switcher = {
                'U': priors.UniformPrior(ccube[key], prior[0], prior[1]),
                'LU': np.power(10., priors.UniformPrior(ccube[key], prior[0], prior[1])),
                'ULU': 1 - np.power(10., priors.UniformPrior(ccube[key], prior[0], prior[1])),
                'FU': np.power(priors.UniformPrior(ccube[key], prior[0], prior[1]), 4),
                'G': priors.GaussianPrior(ccube[key], prior[0], prior[1]).astype('float64'),
                'LG': np.power(10., priors.GaussianPrior(ccube[key], prior[0], prior[1])),
            }

            pr = switcher.get(self.params[par]['prior_type'])
            if pr is None:
                priors.InvalidPrior(par)
            ccube[key] = pr
        return ccube


    def LogLike(self, cube):
        """
        Calculates the log(likelihood) of the forward model generated
        with parameters and known variables.
        """

        # Generate dictionaries for the different classes of parameters
        # and add the known parameters as well as a sample of the
        # retrieved parameters to them
        self.get_param_sample(cube)

        # Test if all the length scales are provided
        self.Length_scales_test()

        # Test and Potential conversion of P0
        self.P0_test()

        # Test/derivation of surface gravity
        self.g_test()

        # Check: Return -inf if pressures for madhuseager models are not nonotonically increasing.
        if self.settings['parametrization'] == 'madhuseager':
            if not self.log_top_pressure<self.temp_vars['log_P1']<self.temp_vars['log_P2']<self.temp_vars['log_P3']<self.phys_vars['log_P0']:
                return -1e32
        if self.settings['parametrization'] == 'mod_madhuseager':
            if not self.log_top_pressure<self.temp_vars['log_P1']<self.temp_vars['log_P2']<self.phys_vars['log_P0']:
                return -1e32

        # Check: Return -inf if parameters for the Guillot model are bad
        if self.settings['parametrization'] == 'guillot':
            if cube[self.params.index('alpha')] < -1:
                return -1e32

        # Check: Return -inf if the cube is bad
        log_prior = cube.sum()
        if (log_prior == -np.inf):
            return -1e32

        # Calculate the P-T profiles
        self.make_press_temp_terr()  # pressures from low to high
        self.rt_object.setup_opa_structure(self.press)

        # Check: Return -inf if there are negative temperatures
        if any((self.temp < 0).tolist()):
            return -1e32

        # Check: Return -inf if the abundances exceed 1
        metal_sum = sum(self.chem_vars.values())
        if metal_sum > 1:
            return -1e32
        else:
            self.inert = (1-metal_sum) * np.ones_like(self.press)

        # Calculate the forward model; this returns the wavelengths in cm
        # and the flux F_nu in erg/cm^2/s/Hz.
        self.retrieval_model_plain()

        # Check: Return -inf if forward model returns NaN values.
        if np.sum(np.isnan(self.rt_object.flux)) > 0:
            print("NaN spectrum encountered")
            return -1e32
                     
        # Calculate total log-likelihood (sum over instruments)
        log_likelihood = 0.
        for inst in self.instrument.keys():
            # Rebin the spectrum according to the input spectrum if wavelenths differ strongly
            if not np.array([(np.round(self.instrument[inst]['wl'],10)==np.round(self.nc.c/self.rt_object.freq*1e4,10))]).all():
                flux_temp = spectres.spectres(self.instrument[inst]['wl'],
                                                self.nc.c/self.rt_object.freq*1e4,
                                                self.rt_object.flux)
                
                # Rebin and add the moon Flux if present
                if self.settings['moon'] == 'True':
                    flux_temp = flux_temp + spectres.spectres(self.instrument[inst]['wl'],
                                                    self.nc.c/self.rt_object.freq*1e4,
                                                    self.moon_flux)

            # If no rebin is required
            else:
                flux_temp = self.rt_object.flux
                
                # Rebin and add the moon Flux if present
                if self.settings['moon'] == 'True':
                    flux_temp = flux_temp + self.moon_flux

            # Calculate log-likelihood
            log_likelihood += -0.5 * np.sum(((flux_temp -
                                            self.instrument[inst]['flux']) /
                                            self.instrument[inst]['error'])**2.)

        return log_likelihood

    
    def get_param_sample(self,cube):
        '''
        Function to generate dictionaries and
        containing a sample of parameters that
        is used to generate the spectra
        '''
        
        # Generate the gictionaries for the parameters 
        self.temp_vars = {}
        self.chem_vars = {}
        self.phys_vars = {}
        self.cloud_vars = {}
        self.scat_vars = {}
        self.moon_vars = {}

        # Add the known parameters to the dictionary
        for par in self.knowns.keys():
            if self.knowns[par]['type'] == 'TEMPERATURE PARAMETERS':
                self.temp_vars[par] = self.knowns[par]['value']
            elif self.knowns[par]['type'] == 'CHEMICAL COMPOSITION PARAMETERS':
                self.chem_vars[par] = self.knowns[par]['value']
            elif self.knowns[par]['type'] == 'PHYSICAL PARAMETERS':
                self.phys_vars[par] = self.knowns[par]['value']
            elif self.knowns[par]['type'] == 'CLOUD PARAMETERS':
                if not '_'.join(par.split('_',2)[:2]) in self.cloud_vars.keys():
                    self.cloud_vars['_'.join(par.split('_',2)[:2])] = {}
                try:
                    self.cloud_vars['_'.join(par.split('_',2)[:2])][par.split('_',2)[2]] = self.knowns[par]['value']
                except:
                    self.cloud_vars['_'.join(par.split('_',2)[:2])]['abundance'] = self.knowns[par]['value']
                    self.chem_vars[par.split('_',1)[0]] = self.knowns[par]['value']
            elif self.knowns[par]['type'] == 'SCATTERING PARAMETERS':
                self.scat_vars[par] = self.knowns[par]['value']
            elif self.knowns[par]['type'] == 'MOON PARAMETERS':
                self.moon_vars[par] = self.knowns[par]['value']

        # Add samples of the retrieved parameters to the dictionary
        r_params = list(self.params.keys())
        for par in range(len(r_params)):
            if self.params[r_params[par]]['type'] == 'TEMPERATURE PARAMETERS':
                self.temp_vars[r_params[par]] = cube[par]
            elif self.params[r_params[par]]['type'] == 'CHEMICAL COMPOSITION PARAMETERS':
                self.chem_vars[r_params[par]] = cube[par]
            elif self.params[r_params[par]]['type'] == 'PHYSICAL PARAMETERS':
                self.phys_vars[r_params[par]] = cube[par]
            elif self.params[r_params[par]]['type'] == 'CLOUD PARAMETERS':
                if not '_'.join(r_params[par].split('_',2)[:2]) in self.cloud_vars.keys():
                    self.cloud_vars['_'.join(r_params[par].split('_',2)[:2])] = {}
                try:
                    self.cloud_vars['_'.join(r_params[par].split('_',2)[:2])][r_params[par].split('_',2)[2]] = cube[par]
                except:
                    self.cloud_vars['_'.join(r_params[par].split('_',2)[:2])]['abundance'] = cube[par]
                    self.chem_vars[r_params[par].split('_',1)[0]] = cube[par]
            elif self.params[r_params[par]]['type'] == 'SCATTERING PARAMETERS':
                self.scat_vars[r_params[par]] = cube[par]
            elif self.params[r_params[par]]['type'] == 'MOON PARAMETERS':
                self.moon_vars[r_params[par]] = cube[par]


    def Length_scales_test(self):
        '''
        Function to scale lengths check form pc
        and earth radii to cm and m
        '''

        # Check if distance to system is provided
        if not 'd_syst' in self.phys_vars:
            print("ERROR! Distance from the star is missing!")
            sys.exit()
        
        # Check if radius of the planet is provided
        if not 'R_pl' in self.phys_vars:
            print("ERROR! Planetary radius is missing!")
            sys.exit()

        # Check if radius of the moon is provided
        if self.settings['moon'] == 'True':
            if not 'R_m' in self.phys_vars:
                print("ERROR! Moon radius is missing!")
                sys.exit()


    def P0_test(self,ind=None):
        '''
        Function to check if the surface pressure is provided or can
        be calculated from the provided parameters and brings it to
        the correct format for petit radtrans
        '''

        # Case dependant setting of the surface pressure
        if self.settings['clouds'] == 'opaque':
            # Choose a surface pressure below the lower cloud deck
            if not (('log_P0' in self.phys_vars) or ('P0' in self.phys_vars)):
                self.phys_vars['log_P0'] = 4
            else:
                if (('log_P0' in self.knowns) or ('P0' in self.knowns)):
                    if ind is not None:
                        if ind == 0:
                            if 'P0' in self.knowns:
                                self.phys_vars['log_P0'] = np.log10(self.knowns['P0']['value'])
                            else:
                                self.phys_vars['log_P0'] = self.knowns['log_P0']['value']
                        else:
                            self.phys_vars['log_P0'] = 4
                    else:
                        self.phys_vars['log_P0'] = 4
                else:
                    print("ERROR! For opaque cloud models the surface pressure P0 is not retrievable!")
                    sys.exit()
        else:
            if not 'log_P0' in self.phys_vars:
                if 'P0' in self.phys_vars:
                    self.phys_vars['log_P0'] = np.log10(self.phys_vars['P0'])
                else:
                    print("ERROR! Either log_P0 or P0 is needed!")
                    sys.exit()


    def g_test(self):
        """
        Function to check if the surface gravity is provided or can
        be calculated from the provided parameters and brings it to
        the correct format for petit radtrans
        """

        # Calculate the surface gravity g given M_Pl and R_pl or log_g. If in knowns already, skip
        if 'g' not in self.phys_vars.keys():
            if 'log_g' in self.phys_vars.keys():
                self.phys_vars['g'] = 10**self.phys_vars['log_g']
            else:
                if not 'M_pl' in self.phys_vars:
                    sys.exit()
                self.phys_vars['g'] = self.nc.G*self.phys_vars['M_pl'] / (self.phys_vars['R_pl'])**2


    def retrieval_model_plain(self,em_contr=True):
        """
        Creates the pressure-temperature profile for the current atmosphere
        and calculates the corresponding emitted flux.
        """

        add_cloud_scat_as_abs = False
        self.abundances = {}
        self.cloud_radii = {}
        self.cloud_lnorm = 0
        for name in self.chem_vars.keys():
            if '(c)' in name:
                add_cloud_scat_as_abs = True
                self.abundances[name] = np.zeros_like(self.press)
                for cloud in self.cloud_vars.keys():
                    # Calculate the bottom pressure from the thickness parameter
                    self.cloud_vars[cloud]['bottom_pressure'] = self.cloud_vars[cloud]['top_pressure']+self.cloud_vars[cloud]['thickness']
                    if name in cloud:
                        self.abundances[name][np.where((self.press<self.cloud_vars[cloud]['bottom_pressure'])
                                                      &(self.press>self.cloud_vars[cloud]['top_pressure']))] = self.chem_vars[name]
                        self.cloud_radii[name]=self.cloud_vars[cloud]['particle_radius']
                        self.cloud_lnorm=self.cloud_vars[cloud]['sigma_lnorm']
            else:
                self.abundances[name.split('_')[0]] = np.ones_like(
                    self.press) * self.chem_vars[name]
        self.calc_MMW()

        # Calculate the moon flux
        if self.settings['moon'] == 'True':
            nu = self.rt_object.freq
            exponent = self.nc.h*nu/(self.nc.kB*self.moon_vars['T_m'])
            B_nu = 2*self.nc.h*nu**3/self.nc.c**2 / (np.exp(exponent)-1)  # in erg/cm^2/s/Hz/sr
            self.moon_flux = np.pi*B_nu  # in erg/cm^2/s/Hz 

        # Setting the scattering parameters for the surface
        if self.settings['scattering'] == 'True':
            # Try adding the reflectance
            try:
                self.rt_object.reflectance = self.scat_vars['reflectance'] * np.ones_like(
                    self.rt_object.freq)
            except:
                pass
            # Try adding the emissivity
            try:
                self.rt_object.emissivity = self.scat_vars['emissivity'] * \
                    np.ones_like(self.rt_object.freq)
            except:
                pass

        # Calculate the Spectrum of the planet, prevent unnecessary printing
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        if not self.settings['directlight']:
            self.rt_object.calc_flux(self.temp, self.abundances, self.phys_vars['g'],
                            self.MMW,radius=self.cloud_radii,sigma_lnorm=self.cloud_lnorm,
                            add_cloud_scat_as_abs = add_cloud_scat_as_abs,contribution = em_contr)
        else:
            self.rt_object.calc_flux(self.temp, self.abundances, self.phys_vars['g'],
                            self.MMW,radius=self.cloud_radii,sigma_lnorm=self.cloud_lnorm,
                            geometry=self.settings['geometry'],Tstar= self.scat_vars['stellar_temp'],
                            Rstar=self.scat_vars['stellar_radius'], semimajoraxis=self.scat_vars['semimajoraxis'],   #*self.nc.r_sun*self.nc.AU
                            add_cloud_scat_as_abs = add_cloud_scat_as_abs,contribution = em_contr)
        sys.stdout = old_stdout

        # Scale the fluxes to the desired separation
        if self.phys_vars['d_syst'] is not None:
            self.rt_object.flux *= self.phys_vars['R_pl']**2/self.phys_vars['d_syst']**2
            if self.settings['moon'] == 'True':
                self.moon_flux *= self.moon_vars['R_m']**2/self.phys_vars['d_syst']**2


    def make_press_temp_terr(self,log_ground_pressure=None,layers=100,log_top_pressure=None):
        """
        Creates the pressure-temperature profile from the temperature
        parameters and the pressure.
        """

        if log_top_pressure is None:
            log_top_pressure = self.log_top_pressure
    
        if self.settings['parametrization'] == 'polynomial':
            if log_ground_pressure is None:
                self.press = np.logspace(log_top_pressure,self.phys_vars['log_P0'], layers, base=10)
            else:
                self.press = np.logspace(log_top_pressure,log_ground_pressure, layers, base=10)
            self.temp = np.polyval(np.array([self.temp_vars['a_'+str(len(self.temp_vars)-1-i)]
                                            for i in range(len(self.temp_vars))]), np.log10(self.press))

        elif self.settings['parametrization'] == 'vae_pt':
            if log_ground_pressure is None:
                self.press = np.logspace(log_top_pressure,self.phys_vars['log_P0'], layers, base=10)
            else:
                self.press = np.logspace(log_top_pressure,log_ground_pressure, layers, base=10)
            self.temp = self.vae_pt.get_temperatures(z=np.array([self.temp_vars['z_'+str(i+1)]
                                            for i in range(len(self.temp_vars))]), log_p=np.log10(self.press))

        elif self.settings['parametrization'] == 'guillot':
            if log_ground_pressure is None:
                self.press=np.logspace(log_top_pressure,self.phys_vars['log_P0'], layers, base=10)
            else:
                self.press=np.logspace(log_top_pressure,log_ground_pressure, layers, base=10)
                
            self.temp = self.nc.guillot_modif(self.press,
                                         1e1**self.temp_vars['log_delta'], 1e1**self.temp_vars['log_gamma'],
                                         self.temp_vars['t_int'], self.temp_vars['t_equ'],
                                         1e1**self.temp_vars['log_p_trans'], self.temp_vars['alpha'])

        elif self.settings['parametrization'] == 'isothermal':
            if log_ground_pressure is None:
                self.press=np.logspace(log_top_pressure,self.phys_vars['log_P0'], layers, base=10)
            else:
                self.press=np.logspace(log_top_pressure,log_ground_pressure, layers, base=10)
            self.temp = self.temp_vars['T_eq'] * np.ones_like(self.press)

        elif self.settings['parametrization'] == 'madhuseager':
            beta1=0.5
            beta2=0.5
            
            def T_P(P_m,P_i,T_i,alpha,beta):
                return (np.log(P_m/P_i)/alpha)**(1/beta)+T_i
            
            P0,P1,P2,P3 = 10**log_top_pressure,10**self.temp_vars['log_P1'],10**self.temp_vars['log_P2'],10**self.temp_vars['log_P3'] 

            if log_ground_pressure is None:
                self.press=np.logspace(log_top_pressure,self.phys_vars['log_P0'], layers, base=10)
            else:
                self.press=np.logspace(log_top_pressure,log_ground_pressure, layers, base=10)
            self.temp = np.zeros_like(self.press)
            
            T2 = self.temp_vars['T0'] + (np.log(P1/P0)/self.temp_vars['alpha1'])**(1/beta1) - (np.log(P1/P2)/self.temp_vars['alpha2'])**(1/beta2)
            T3 = T_P(P3,P2,T2,self.temp_vars['alpha2'],beta2)

            for i in range(np.size(self.press)):
                if self.press[i] < P1:
                    self.temp[i] = T_P(self.press[i],P0,self.temp_vars['T0'],self.temp_vars['alpha1'],beta1)
                elif P1 < self.press[i] < P3:
                    self.temp[i] = T_P(self.press[i],P2,T2,self.temp_vars['alpha2'],beta2)
                elif self.press[i] > P3:
                    self.temp[i] = T3
            
            self.temp = sci.gaussian_filter1d(self.temp, 20.0, mode = 'nearest')

        elif self.settings['parametrization'] == 'mod_madhuseager':
            beta1=0.5
            beta2=0.5
            
            def T_P(P_m,P_i,T_i,alpha,beta):
                return (np.log(P_m/P_i)/alpha)**(1/beta)+T_i
            
            P0,P1,P2 = 10**log_top_pressure,10**self.temp_vars['log_P1'],10**self.temp_vars['log_P2'] 

            if log_ground_pressure is None:
                self.press=np.logspace(log_top_pressure,self.phys_vars['log_P0'], layers, base=10)
            else:
                self.press=np.logspace(log_top_pressure,log_ground_pressure, layers, base=10)
            self.temp = np.zeros_like(self.press)
            
            T2 = self.temp_vars['T0'] + (np.log(P1/P0)/self.temp_vars['alpha1'])**(1/beta1) - (np.log(P1/P2)/self.temp_vars['alpha2'])**(1/beta2)

            for i in range(np.size(self.press)):
                if self.press[i] < P1:
                    self.temp[i] = T_P(self.press[i],P0,self.temp_vars['T0'],self.temp_vars['alpha1'],beta1)
                elif P1 < self.press[i]:
                    self.temp[i] = T_P(self.press[i],P2,T2,self.temp_vars['alpha2'],beta2)
            
            #self.temp = sci.gaussian_filter1d(self.temp, 20.0, mode = 'nearest')

        elif self.settings['parametrization'] == 'input':
            self.press, self.temp = np.loadtxt(
                self.temp_vars['input_path'], unpack=True)

        else:
            sys.exit('Unknown pt setting.')

        return


    def calc_MMW(self):
        """
        Calculates the mean molecular weight for the modeled atmosphere, by summing
        the molecular weights for each gas weighted by their abundance.
        """

        self.MMW = 0.
        for key in self.abundances.keys():
            if key in self.MMW_Storage.keys():
                self.MMW += self.abundances[key] * self.MMW_Storage[key]
            else:
                print('WARNING! Missing MMW for ', key)

        if 'mmw_inert' in self.settings.keys():
            self.MMW +=self.inert * float(self.settings['mmw_inert'])
        return
