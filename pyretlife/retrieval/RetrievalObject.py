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

from pyretlife.retrieval import UnitsUtil as units
from pyretlife.retrieval.config import read_config_file, check_if_configs_match,populate_dictionaries,make_output_folder,load_data
from pyretlife.retrieval.config_validation import validate_config
from pyretlife.retrieval.unit_conversions import convert_spectrum
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
        self.run_retrieval = run_retrieval

        self.petitRADTRANS = importlib.import_module("petitRADTRANS")

        self.knowns = {}
        self.parameters = {}
        self.settings = {}
        self.instrument={}
        # Create a units object to enable unit conversions
        self.units = units.UnitsUtil(nc)

    def load_configuration(self, config_file:str):

        # Load standard configurations (hard-coded)
        self.config_default = read_config_file(file_path=Path("configs/config_default.yaml"))
        # Read in the configuration and check if there is already one in the file
        self.config = read_config_file(file_path=Path(config_file))

        # Check if configuration file exists and if it matches
        if not check_if_configs_match(config=self.config):
            raise RuntimeError("Config exists and does not match!")

        # Save config into the four dictionaries
        self.knowns,self.parameters,self.settings,self.units = populate_dictionaries(self.config_default, self.knowns,self.parameters,self.settings,self.units)
        self.knowns, self.parameters, self.settings, self.units = populate_dictionaries(self.config,  self.knowns,self.parameters,self.settings,self.units)
        self.instrument=load_data(self.settings,self.units)

        # TODO implement validation
        validate_config(self.config)

    def unit_conversion(self):

        self.instrument=convert_spectrum(self.instrument, self.units)
        pass
        # TODO Bjoern?
        # for par in self.parameters:
        #     if 'unit' in par.keys():
        #         # custom or specified unit
        #         input_unit = u.Unit(par['unit'].value)
        #     else:
        #         input_unit = u.return_units(par, u.std_input_units)


    def saving_inputs_to_folder(self):
        make_output_folder(self.settings['output_folder'])
        # save_config_file()
        # save_instrument_data()
        # save_converted_dictionaries()
        # save_environment_variables()
        # save_github_commit_string()