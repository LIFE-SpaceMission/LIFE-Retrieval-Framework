"""
Script to generate a new spectrum (which can later be fed into a retrieval).
The config file passed to this script must:
    1. have the output folder as a file where the spectrum will be written to
    2. all variables must be knowns (i.e. must have truth values and no prior dict)
    3. not have a data_file specified
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from argparse import ArgumentParser, Namespace

from pyretlife.retrieval.run import RetrievalObject

import sys


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------


def get_cli_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        required=False,
        help="Path to the configuration file.",
    )
    args = parser.parse_args()
    return args


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Read the command line arguments (config file path)
    args = get_cli_arguments()
    args.config = '/home/ipa/quanz/user_accounts/zaburr/sampling/config/simple_spec.yaml'

    # Initializes a RetrievalObject (the pyret_ship)
    pyret_ship = RetrievalObject(run_retrieval=True)
    pyret_ship.load_configuration(config_file=args.config)
    pyret_ship.unit_conversion()
    pyret_ship.assign_knowns()
    # pyret_ship.assign_prior_functions()
    pyret_ship.vae_initialization()
    pyret_ship.petitRADTRANS_initialization()
    
    
    # Run forward model to calculate spectrum
    pyret_ship.generate_new_spectrum()
