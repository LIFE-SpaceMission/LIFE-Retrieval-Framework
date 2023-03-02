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

from collections import OrderedDict

from pathlib import Path

from pyretlife.retrieval import units as units
from config import read_config_file, check_if_configs_match


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
        config_file,
        run_retrieval: bool = True,
    ):
        """
        This function reads the config.ini file and initializes all
        the variables. It also ensures that the run is not rewritten
        unintentionally.
        """

        # Store constructor arguments
        self.run_retrieval = run_retrieval

        # Store the config_file
        self.config = config_file

        # TODO: I don't even know what you are doing here, but calling
        #   `__import__` is not a good idea. If anything, you should use
        #   `importlib.import_module`.
        #   Also, `rt` and `nc` are not very descriptive names.
        self.rt = importlib.import_module("petitRADTRANS")
        self.nc = self.rt.nat_cst

        # TODO: These things need documentation.
        #   Maybe it even makes sense to use a `dataclass` here?
        self.log_top_pressure = -5  # -6 # why did we change this?
        self.params = OrderedDict()
        self.knowns = OrderedDict()

        # Create a units object to enable unit conversions
        self.units = units.UnitsUtil(self.rt.nat_cst)

    def populate_dictionaries(self):
        # old read_var function
        self.Knowns={}
        self.Parameters={}
        self.Settings={}

        for section in self.config.keys():
            for subsection in self.config[section].keys():
                if type(self.config[section][subsection]) is dict:
                    if "prior" in self.config[section][subsection].keys():
                        self.Parameters[subsection] = self.config[section][subsection]
                    elif "truth" in self.config[section][subsection].keys():
                        self.Knowns[subsection] = self.config[section][subsection]
                    else:
                        self.Settings[subsection] = self.config[section][subsection]

