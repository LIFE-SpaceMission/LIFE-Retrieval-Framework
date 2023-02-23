__author__ = "Alei, Konrad, Molliere, Quanz"
__copyright__ = "Copyright 2022, Alei, Konrad, Molliere, Quanz"
__maintainer__ = ",Björn S. Konrad, Eleonora Alei"
__email__ = "konradb@ethz.ch, elalei@phys.ethz.ch"
__status__ = "Development"

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from collections import OrderedDict

import configparser
import json
import os
import sys
import warnings
from pathlib import Path

import astropy.units as u
import numpy as np
import scipy.ndimage as sci
import spectres as spectres

from pyretlife.retrieval import priors as priors, units as units
from config import read_config_file, read_paths, check_if_configs_match


class RetrievalObject:
    def __init__(
        self,
        input_file=Path("config.ini"),
        path_file=Path("paths.yaml"),
        retrieval=True,
    ):
        """
        This function reads the config.ini file and initializes all
        the variables. It also ensures that the run is not rewritten
        unintentionally.
        """
        self.config_file = read_config_file(input_file)
        check_if_configs_match(self.config_file)
        self.path_prt, self.path_opacity, self.path_multinest = read_paths(
            path_file
        )

        # should I do it again? It has been done already in read_paths
        #sys.path.append(self.path_prt)
        #os.environ["pRT_input_data_path"] = self.path_opacity
        self.rt = __import__("petitRADTRANS")
        self.nc = self.rt.nat_cst

        self.log_top_pressure = -5  # -6 # why did we change this?
        self.params = OrderedDict()
        self.knowns = OrderedDict()

        # Create a units object to enable unit conversions
        self.units = units.UnitsUtil(self.rt.nat_cst)

    def populate_dictionaries(self):
        # old read_var function
        pass
