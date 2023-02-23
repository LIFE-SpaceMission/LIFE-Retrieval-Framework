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

from collections import OrderedDict

from pathlib import Path

from pyretlife.retrieval import units as units
from config import read_config_file, read_paths, check_if_configs_match


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
        config_file_path: Path = Path("config.ini"),
        paths_file_path: Path = Path("paths.yaml"),
        run_retrieval: bool = True,
    ):
        """
        This function reads the config.ini file and initializes all
        the variables. It also ensures that the run is not rewritten
        unintentionally.
        """

        # Store constructor arguments
        self.run_retrieval = run_retrieval

        # Read the config file and check if the run is not overwritten
        self.config = read_config_file(config_file_path)
        check_if_configs_match(self.config)

        # Read the paths file
        # TODO: You should decide if this function only reads in the paths
        #   file or if it also does things like setting environment variables
        #   or appending to the `sys.path`. In this case, you should rename
        #   the function to something like `read_paths_and_set_environment`.
        self.path_prt, self.path_opacity, self.path_multinest = read_paths(
            paths_file_path
        )

        # TODO: See above. General rule: "explicit is better than implicit",
        #   meaning, for example, that if you have a function "read_paths",
        #   it should only read in the paths and not do anything else, like
        #   modifying the `sys.path` or setting environment variables.
        # should I do it again? It has been done already in read_paths
        # sys.path.append(self.path_prt)
        # os.environ["pRT_input_data_path"] = self.path_opacity

        # TODO: I don't even know what you are doing here, but calling
        #   `__import__` is not a good idea. If anything, you should use
        #   `importlib.import_module`.
        #   Also, `rt` and `nc` are not very descriptive names.
        self.rt = __import__("petitRADTRANS")
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
        pass
