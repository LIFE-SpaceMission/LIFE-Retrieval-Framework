"""
This module contains the `RetrievalObject` class, which is the main
class of the pyretlife package.
"""

__author__ = "Alei, Konrad, Molliere, Quanz"
__copyright__ = "Copyright 2022, Alei, Konrad, Molliere, Quanz"
__maintainer__ = "Björn S. Konrad, Eleonora Alei"
__email__ = "konradb@ethz.ch, elalei@phys.ethz.ch"
__status__ = "Development"


# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import importlib
import json
import os
import sys
from pathlib import Path
import numpy as np

from pyretlife.retrieval.atmospheric_variables import (
    calculate_gravity,
    calculate_polynomial_profile,
    calculate_vae_profile,
    calculate_guillot_profile,
    calculate_isothermal_profile,
    calculate_madhuseager_profile,
    calculate_mod_madhuseager_profile,
    calculate_abundances,
    set_log_ground_pressure,
    assign_cloud_parameters,
    calc_mmw,
)
from pyretlife.retrieval.configuration_ingestion import (
    read_config_file,
    check_if_configs_match,
    populate_dictionaries,
    make_output_folder,
    load_data,
    get_check_opacity_path,
    get_check_prt_path,
    get_retrieval_path,
    set_prt_opacity,
)
from pyretlife.retrieval.likelihood_validation import (
    validate_pt_profile,
    validate_cube_finite,
    validate_positive_temperatures,
    validate_sum_of_abundances,
    validate_spectrum_goodness,
)
from pyretlife.retrieval.priors import assign_priors
from pyretlife.retrieval.radiative_transfer import (
    define_linelists,
    calculate_moon_flux,
    assign_reflectance_emissivity,
    calculate_emission_flux,
    scale_flux_to_distance,
    rebin_spectrum,
)
from pyretlife.retrieval.units import (UnitsUtil,
                                       convert_spectrum,
                                       convert_knowns_and_parameters,
                                       )


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------


class RetrievalObject:
    """
    This class binds together all the different parts of the retrieval.

    Args:
        run_retrieval:

    Attributes:
        config: The configuration (i.e., the contents of the YAML or
            INI file for a given retrieval) as a dictionary.
        input_prt_path: Path to the petitRADTRANS installation.
        input_opacity_path: Path to the opacity data.
        ...

    TODO: Keep adding attributes here to document them.
    TODO: Maybe also add an `__repr__` method to this class?
    """

    def __init__(
        self,
        run_retrieval: bool = True,
        ):
        """
        This function reads the config.ini file and initializes all
        the variables. It also ensures that the run is not rewritten
        unintentionally.
        """

        # Store constructor arguments
        self.vae_pt = None
        self.moon_flux = None
        self.MMW = None
        self.inert = None
        self.temp = None
        self.press = None
        self.moon_vars = None
        self.scat_vars = None
        self.cloud_vars = None
        self.phys_vars = None
        self.chem_vars = None
        self.temp_vars = None
        self.config = None
        self.config_default = None
        self.rt_object = None
        self.run_retrieval = run_retrieval

        self.knowns = {}
        self.parameters = {}
        self.settings = {}
        self.instrument = {}

        # # Get and check the goodness of the environmental variables
        self.input_opacity_path = get_check_opacity_path()
        self.input_prt_path = get_check_prt_path()
        self.input_retrieval_path = get_retrieval_path()
        sys.path.append(str(self.input_prt_path))
        set_prt_opacity(self.input_prt_path, self.input_opacity_path)

        self.petitRADTRANS = importlib.import_module("petitRADTRANS")

        # Create a units object to enable unit conversions
        self.units = UnitsUtil(self.petitRADTRANS.nat_cst)

    def load_configuration(self, config_file: str):
        # Load standard configurations (hard-coded)
        self.config_default = read_config_file(
            file_path=Path("configs/config_default.yaml")
        )
        # Read in the configuration and check if there is already one in the file
        self.config = read_config_file(file_path=Path(config_file))

        # Check if configuration file exists and if it matches
        if not check_if_configs_match(config=self.config):
            raise RuntimeError("Config exists and does not match!")

        # Save config into the four dictionaries
        (
            self.knowns,
            self.parameters,
            self.settings,
            self.units,
        ) = populate_dictionaries(
            self.config_default,
            self.knowns,
            self.parameters,
            self.settings,
            self.units,
        )
        (
            self.knowns,
            self.parameters,
            self.settings,
            self.units,
        ) = populate_dictionaries(
            self.config,
            self.knowns,
            self.parameters,
            self.settings,
            self.units
        )
        self.instrument = load_data(self.settings, self.units,retrieval=self.run_retrieval)

        # IF CLOUDS, ASSIGN P0 WHEN NOT PROVIDED
        # TODO P0_test()
        # TODO implement validation
        # validate_config(self.config)

    def unit_conversion(self):
        self.instrument = convert_spectrum(self.instrument, self.units)
        self.knowns = convert_knowns_and_parameters(self.knowns, self.units)
        self.parameters = convert_knowns_and_parameters(
            self.parameters, self.units
        )

    def assign_knowns(self):
        self.temp_vars = {}
        self.chem_vars = {}
        self.phys_vars = {}
        self.cloud_vars = {}
        self.scat_vars = {}
        self.moon_vars = {}

        # Add the known parameters to the dictionary
        for par in self.knowns.keys():
            if self.knowns[par]["type"] == "TEMPERATURE PARAMETERS":
                self.temp_vars[par] = self.knowns[par]["truth"]
            elif self.knowns[par]["type"] == "CHEMICAL COMPOSITION PARAMETERS":
                self.chem_vars[par] = self.knowns[par]["truth"]
            elif self.knowns[par]["type"] == "PHYSICAL PARAMETERS":
                self.phys_vars[par] = self.knowns[par]["truth"]
            elif self.knowns[par]["type"] == "CLOUD PARAMETERS":
                # TODO review this snippet
                if (
                    not "_".join(par.split("_", 2)[:2])
                    in self.cloud_vars.keys()
                ):
                    self.cloud_vars["_".join(par.split("_", 2)[:2])] = {}
                try:
                    self.cloud_vars["_".join(par.split("_", 2)[:2])][
                        par.split("_", 2)[2]
                    ] = self.knowns[par]["truth"]
                except:
                    self.cloud_vars["_".join(par.split("_", 2)[:2])][
                        "abundance"
                    ] = self.knowns[par]["truth"]
                    self.chem_vars[par.split("_", 1)[0]] = self.knowns[par][
                        "truth"
                    ]
            elif self.knowns[par]["type"] == "SCATTERING PARAMETERS":
                self.scat_vars[par] = self.knowns[par]["truth"]
            elif self.knowns[par]["type"] == "MOON PARAMETERS":
                self.moon_vars[par] = self.knowns[par]["truth"]

        # in case the PT profile is known, assign it already
        if self.settings["parameterization"] == "input":
            self.press, self.temp = np.loadtxt(
                self.temp_vars["input_path"], unpack=True
            )

    def assign_prior_functions(self):
        self.parameters = read_input_prior(self.parameters)
        self.parameters = assign_priors(self.parameters)
        # TODO Check that all priors are valid (invalid_prior function in priors.py otherwise)

    def petitRADTRANS_initialization(self):
        """
        Initializes the rt_object given the wavelength range.
        """
        (
            used_line_species,
            used_rayleigh_species,
            used_cia_species,
            used_cloud_species,
        ) = define_linelists(
            self.config, self.settings, self.input_opacity_path
        )

        # TODO implement verbose output
        # old_stdout = sys.stdout
        # sys.stdout = open(os.devnull, "w")
        ls = sorted(used_line_species)[::-1]
        self.rt_object = self.petitRADTRANS.Radtrans(
            line_species=ls,
            rayleigh_species=sorted(used_rayleigh_species),
            continuum_opacities=sorted(used_cia_species),
            cloud_species=sorted(used_cloud_species),
            wlen_bords_micron=self.settings["wavelength_range"],
            mode="c-k",
            do_scat_emis=True in self.settings["include_scattering"].values(),
        )
        # sys.stdout = old_stdout
        self.rt_object.setup_opa_structure(
            np.logspace(
                self.settings["log_top_pressure"],
                0,
                self.settings["n_layers"],
                base=10,
            )
        )

    def unity_cube_to_prior_space(self, cube):
        cube_copy = cube.copy()
        for par in self.parameters.keys():
            prior = self.parameters[par]["prior"]
            idx = list(self.parameters.keys()).index(par)
            cube_copy[idx] = prior["function"](cube_copy[idx], prior["prior_specs"])
        return cube_copy

    def assign_cube_to_parameters(self, cube):
        for par in self.parameters.keys():
            idx = list(self.parameters.keys()).index(par)
            if self.parameters[par]["type"] == "TEMPERATURE PARAMETERS":
                self.temp_vars[par] = cube[idx]
            elif (
                self.parameters[par]["type"]
                == "CHEMICAL COMPOSITION PARAMETERS"
            ):
                self.chem_vars[par] = cube[idx]
            elif self.parameters[par]["type"] == "PHYSICAL PARAMETERS":
                self.phys_vars[par] = cube[idx]
            elif self.parameters[par]["type"] == "CLOUD PARAMETERS":
                if (
                    not "_".join(par.split("_", 2)[:2])
                    in self.cloud_vars.keys()
                ):
                    self.cloud_vars["_".join(par.split("_", 2)[:2])] = {}
                try:
                    self.cloud_vars["_".join(par.split("_", 2)[:2])][
                        par.split("_", 2)[2]
                    ] = cube[idx]
                except:
                    self.cloud_vars["_".join(par.split("_", 2)[:2])][
                        "abundance"
                    ] = cube[idx]
                    self.chem_vars[par.split("_", 1)[0]] = cube[idx]
            elif self.parameters[par]["type"] == "SCATTERING PARAMETERS":
                self.scat_vars[par] = cube[idx]
            elif self.parameters[par]["type"] == "MOON PARAMETERS":
                self.moon_vars[par] = cube[idx]

    def calculate_pt_profile(
        self, parameterization, log_ground_pressure, log_top_pressure, layers
    ):
        """
        Creates the pressure-temperature profile from the temperature
        parameters and the pressure.
        """
        self.press = np.array(
            np.logspace(log_top_pressure, log_ground_pressure, layers, base=10)
        )

        if parameterization == "polynomial":
            self.temp = calculate_polynomial_profile(self.press, self.temp_vars)

        # TODO understand what is going on here
        elif parameterization == "vae_pt":
            self.temp = calculate_vae_profile(self.press, self.vae_pt, self.temp_vars)

        elif parameterization == "guillot":
            self.temp = calculate_guillot_profile(self.press, self.petitRADTRANS, self.temp_vars)

        elif self.settings["parameterization"] == "isothermal":
            self.temp = calculate_isothermal_profile(self.press, self.temp_vars)

        elif self.settings["parameterization"] == "madhuseager":
            self.temp = calculate_madhuseager_profile(self.press, self.temp_vars)

        elif self.settings["parameterization"] == "mod_madhuseager":
            self.temp = calculate_mod_madhuseager_profile(
                self.press, self.temp_vars
            )

        else:
            raise ValueError("Unknown PT setting!")

        return
    
    def calculate_spectrum(self):
        self.inert = (1 - sum(self.chem_vars.values())) * np.ones_like(
            self.press
        )

        self.abundances = calculate_abundances(self.chem_vars, self.press)
        (
            self.abundances,
            self.cloud_vars,
            self.cloud_radii,
            self.cloud_lnorm,
        ) = assign_cloud_parameters(
            self.abundances, self.cloud_vars, self.press
        )

        self.MMW = calc_mmw(self.abundances, self.settings, self.inert)

        # initialize calculated pressure
        self.rt_object.setup_opa_structure(self.press)

        if self.settings["include_moon"]:
            self.moon_flux = calculate_moon_flux(self.rt_object.freq, self.petitRADTRANS, self.moon_vars)

        if (
            self.settings["include_scattering"]["direct_light"]
            or self.settings["include_scattering"]["thermal"]
        ):
            (
                self.rt_object.reflectance,
                self.rt_object.emissivity,
            ) = assign_reflectance_emissivity(
                self.scat_vars, self.rt_object.freq
            )

        # Calculate the forward model; this returns the frequency
        # and the flux F_nu in erg/cm^2/s/Hz.
        self.rt_object.freq, self.rt_object.flux = calculate_emission_flux(self.rt_object, self.settings, self.temp,
                                                                           self.abundances, self.phys_vars["g"],
                                                                           self.MMW, self.cloud_radii, self.cloud_lnorm,
                                                                           self.scat_vars, em_contr=False)

        self.rt_object.wavelength = (
            self.petitRADTRANS.nat_cst.c / self.rt_object.freq * 1e4
        )
    
    def distance_scale_spectrum(self):
        if self.phys_vars["d_syst"] is not None:
            # WARNING! THIS CONVERTS UNITS OF PRT SPECTRUM from cm-2 to m-2
            self.rt_object.flux = scale_flux_to_distance(
                self.rt_object.flux,
                self.phys_vars["R_pl"],
                self.phys_vars["d_syst"],
            )
            if self.settings["include_moon"]:
                self.moon_flux = scale_flux_to_distance(
                    self.moon_flux,
                    self.moon_vars["R_m"],
                    self.phys_vars["d_syst"],
                )

    def calculate_log_likelihood(self, cube):
        """
        Calculates the log(likelihood) of the forward model generated
        with parameters and known variables.
        """

        # Generate dictionaries for the different classes of parameters
        # and add the known parameters as well as a sample of the
        # retrieved parameters to them
        self.assign_cube_to_parameters(cube)
        # TODO expand on tests here

        # test goodness of random draw
        if validate_pt_profile(self.settings, self.temp_vars, self.phys_vars):
            return -1e99
        if validate_cube_finite(cube):
            return -1e99
        
        self.phys_vars = calculate_gravity(self.phys_vars,self.config)
        self.phys_vars = set_log_ground_pressure(self.phys_vars, self.config, self.knowns)

        if self.settings["parameterization"] != "input":
            self.calculate_pt_profile(
                parameterization=self.settings["parameterization"],
                log_ground_pressure=self.phys_vars["log_P0"],
                log_top_pressure=self.settings["log_top_pressure"],
                layers=self.settings["n_layers"],
            )

        if validate_positive_temperatures(self.temp):
            return -1e99
        if validate_sum_of_abundances(self.chem_vars):
            return -1e99

        self.calculate_spectrum()
        if validate_spectrum_goodness(self.rt_object.flux):
            return -1e99

        self.distance_scale_spectrum()

        # Calculate total log-likelihood (sum over instruments)
        log_likelihood = 0.0
        for inst in self.instrument.keys():
            # Rebin the spectrum according to the input spectrum if wavelengths
            # differ strongly
            rebinned_flux = rebin_spectrum(
                self.instrument[inst],
                self.rt_object.wavelength,
                self.rt_object.flux,
            )
            if self.settings["include_moon"] == "True":
                rebinned_flux += rebin_spectrum(
                    self.instrument[inst],
                    self.rt_object.wavelength,
                    self.moon_flux,
                )

            # Calculate log-likelihood
            log_likelihood += -0.5 * np.sum(
                (
                    (rebinned_flux - self.instrument[inst]["flux"])
                    / self.instrument[inst]["error"]
                )
                ** 2.0
            )
        return log_likelihood

    def vae_initialization(self):
        # TODO see if it can be improved

        # if the vae_pt is selected initialize the pt profile model
        if self.settings["parameterization"] == "vae_pt":
            from pyretlife.retrieval import pt_vae as vae

            self.vae_pt = vae.VAE_PT_Model_Flow(
                os.path.dirname(os.path.realpath(__file__))
                + "/vae_pt_models/Flow/"
                + self.settings["vae_net"],
            )
        if self.settings["parameterization"] == "vae_pt_flow":
            from pyretlife.retrieval import pt_vae as vae

            print("flow")
            self.vae_pt = vae.VAE_PT_Model_Flow(
                os.path.dirname(os.path.realpath(__file__))
                + "/vae_pt_models/Flow/"
                + self.settings["vae_net"],
                flow_path=os.path.dirname(os.path.realpath(__file__))
                + "/vae_pt_models/Flow/flow-state-dict.pt",
            )

    def saving_inputs_to_folder(self):
        make_output_folder(self.settings["output_folder"])
        for data_file in self.settings["data_files"].keys():
            input_string = self.settings["data_files"][data_file]["path"]
            os.system(
                "cp "
                + input_string
                + " "
                + self.settings["output_folder"]
                + "/input_"
                + input_string.split("/")[-1]
            )

        # SAVE GITHUB COMMIT STRING
        if self.input_retrieval_path != "":
            os.system(
                "git -C "
                + self.input_retrieval_path
                + " show --name-status >"
                + self.settings["output_folder"]
                + "/git_commit.txt"
            )

        with open("%s/params.json" % self.settings["output_folder"], "w") as f:
            json.dump(list(self.parameters.keys()), f, indent=2)
        # save_config_file()
        # save_instrument_data()
        # save_converted_dictionaries()
        # save_environment_variables()
        # save_github_commit_string()
