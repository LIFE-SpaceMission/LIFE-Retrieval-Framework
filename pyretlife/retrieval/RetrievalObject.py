"""
This module contains the `RetrievalObject` class, which is the main
class of the pyretlife package.
"""

__author__ = "Alei, Konrad, Molliere, Quanz"
__copyright__ = "Copyright 2022, Alei, Konrad, Molliere, Quanz"
__maintainer__ = "Björn S. Konrad, Eleonora Alei"
__email__ = "konradb@ethz.ch, elalei@phys.ethz.ch"
__status__ = "Development"

import importlib

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
import os,sys
import numpy as np
from pyretlife.retrieval import UnitsUtil as units
from pyretlife.retrieval.config import (
    read_config_file,
    check_if_configs_match,
    populate_dictionaries,
    make_output_folder,
    load_data,
    get_check_opacity_path,
    get_check_prt_path,
    set_prt_opacity
)
from pyretlife.retrieval.petitRADTRANS_initialization import define_linelists
from pyretlife.retrieval.config_validation import validate_config
from pyretlife.retrieval.unit_conversions import (
    convert_spectrum,
    convert_knowns_and_parameters,
)
from pyretlife.retrieval.priors import assign_priors
import pyretlife.retrieval.nat_cst as nc


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------


class RetrievalObject:
    """
    This class binds together all the different parts of the retrieval.

    Args:
        config_file_path: Path to the config file.
        paths_file_path: Path to the paths file.
        run_retrieval:

    Attributes:
        config: The configuration (i.e., the contents of the YAML or
            INI file for a given retrieval) as a dictionary.
        path_prt: Path to the petitRADTRANS installation.
        path_opacity: Path to the opacity data.
        path_multinest: Path to the MultiNest installation.
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
        self.config = None
        self.config_default = None
        self.rt_object = None
        self.run_retrieval = run_retrieval

        self.petitRADTRANS = importlib.import_module("petitRADTRANS")

        self.knowns = {}
        self.parameters = {}
        self.settings = {}
        self.instrument = {}
        # Create a units object to enable unit conversions
        self.units = units.UnitsUtil(nc)

        # TODO uncomment when final
        # # Get and check the goodness of the environmental variables
        self.input_opacity_path=get_check_opacity_path()
        self.input_prt_path = get_check_prt_path()
        sys.path.append(str(self.input_prt_path))
        set_prt_opacity(self.input_prt_path,self.input_opacity_path)


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
        ) = populate_dictionaries(self.config_default, self.knowns, self.parameters, self.settings, self.units)
        (
            self.knowns,
            self.parameters,
            self.settings,
            self.units,
        ) = populate_dictionaries(self.config, self.knowns, self.parameters, self.settings, self.units)
        self.instrument = load_data(self.settings, self.units)

        # TODO implement validation
        # validate_config(self.config)

    def unit_conversion(self):
        self.instrument = convert_spectrum(self.instrument, self.units)
        self.knowns = convert_knowns_and_parameters(self.knowns, self.units)
        self.parameters = convert_knowns_and_parameters(
            self.parameters, self.units
        )

    def assign_prior_functions(self):
        self.parameters= assign_priors(self.parameters)

    def petitRADTRANS_initialization(self):
        """
        Initializes the rt_object given the wavelength range.
        """
        used_line_species, used_rayleigh_species, used_cia_species, used_cloud_species = define_linelists(self.config,self.settings, self.input_opacity_path)

        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        ls = sorted(used_line_species)[::-1]
        self.rt_object = self.petitRADTRANS.Radtrans(
            line_species=ls,
            rayleigh_species=sorted(used_rayleigh_species),
            continuum_opacities=sorted(used_cia_species),
            cloud_species=sorted(used_cloud_species),
            wlen_bords_micron=self.settings['wavelength_range'],
            mode="c-k",
            do_scat_emis=self.settings["scattering"],
        )
        sys.stdout = old_stdout
        self.rt_object.setup_opa_structure(np.logspace(self.settings['top_log_pressure'], 0, self.settings['n_layers'], base=10))



    def read_MMW_Storage(self):

        # Read in the molecular weights database
        self.MMW_Storage = {}
        reader = np.loadtxt(
            self.input_path_opacity + "/opa_input_files/Molecular_Weights.txt",
            dtype="str",
        )
        for i in range(len(reader[:, 0])):
            self.MMW_Storage[reader[i, 0]] = float(reader[i, 1])




    # def vae_initialization(self):
        # # if the vae_pt is selected initialize the pt profile model
        # if self.settings["parametrization"] == "vae_pt":
        #     from pyretlife.retrieval import pt_vae as vae
        #
        #     self.vae_pt = vae.VAE_PT_Model_Flow(
        #         os.path.dirname(os.path.realpath(__file__))
        #         + "/vae_pt_models/Flow/"
        #         + self.settings["vae_net"],
        #     )
        # if self.settings["parametrization"] == "vae_pt_flow":
        #     from pyretlife.retrieval import pt_vae as vae
        #
        #     print("flow")
        #     self.vae_pt = vae.VAE_PT_Model_Flow(
        #         os.path.dirname(os.path.realpath(__file__))
        #         + "/vae_pt_models/Flow/"
        #         + self.settings["vae_net"],
        #         flow_path=os.path.dirname(os.path.realpath(__file__))
        #         + "/vae_pt_models/Flow/flow-state-dict.pt",
        #     )
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

        # save_config_file()
        # save_instrument_data()
        # save_converted_dictionaries()
        # save_environment_variables()
        # save_github_commit_string()
