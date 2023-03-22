"""
Read in configuration files.
"""
import os
import sys
import glob
# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from configparser import ConfigParser
from pathlib import Path
from typing import Union, Tuple

import hashlib
import yaml

from deepdiff import DeepDiff


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

def read_config_file(file_path: Union[Path, str]) -> dict:
    """
    Read a configuration from a YAML file.

    Args:
        file_path: Path to a file with the configuration.

    Returns:
        The configuration as a dictionary.
    """

    file_path = Path(file_path)

    if file_path.suffix == ".yaml":
        with open(file_path, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
        return config
    else:
        raise ValueError(f"Unknown file extension: {file_path.suffix}. Please convert it to a .yaml file.")


def check_if_configs_match(config: dict) -> bool:
    """
    The check_if_configs_match function checks if the config file in the retrieval directory
    matches the input.yaml file that was used to run a previous simulation. If they match,
    the function returns True; otherwise, it returns False.

    Args:
        config [dict]: The configuration dictionary

    Returns
        True if the files are the same, False if they are different.
    """
    # Expected location of a config file; check if it exists
    retrieval_dir = Path(config["RUN SETTINGS"]["output_folder"])
    if not retrieval_dir.exists():
        return True

    # Find all the config files in the retrieval directory
    try:
        config_file = next(retrieval_dir.glob("input.yaml"))
    except StopIteration:
        return True

    return not DeepDiff(
        config,
        read_config_file(config_file),
        ignore_order=True,
    )


def compute_hash_of_config_file(file_path: Union[Path, str]) -> str:
    """
    Compute the hash of a configuration file.

    Args:
        file_path: Path to a file with the configuration.

    Returns:
        The hash of the configuration file.
    """

    file_path = Path(file_path)
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def convert_ini_to_yaml(file_path: Union[Path, str]) -> None:
    pass



def validate_config(config: dict) -> None:
    check_temperature_parameters(config)
    pass


def get_check_opacity_path() -> Path:
    """
    The get_check_opacity_path function checks that the PYRETLIFE_OPACITY_PATH environment variable is set.
    If it is not, an error message is printed and the program exits. If it is set, then a Path object pointing to
    the opacity folder in this directory will be returned.

    Args:
        None
    Returns:
        The path to the opacity folder
    """

    input_opacity_path = os.environ.get("PYRETLIFE_OPACITY_PATH")
    if input_opacity_path is None:
        raise RuntimeError("PYRETLIFE_OPACITY_PATH not set!")
    if not Path(input_opacity_path).exists():
        raise RuntimeError("PYRETLIFE_OPACITY_PATH set, but folder does not exist!")
    if len(glob.glob(input_opacity_path +"/opacities/*"))==0:
        raise RuntimeError("PYRETLIFE_OPACITY_PATH set, but folder is not valid.")
    return Path(input_opacity_path)

def get_check_prt_path() -> Path:
    """
    The get_check_pRT_path function checks that the PYRETLIFE_PRT_PATH environment variable is set, and if so,
    checks that it points to a valid folder. If all these conditions are met, then the function returns a Path object
    pointing to this folder.

    Args:
        None
    Returns:
        The path to the petitRADTRANS folder

    """

    input_pRT_path = os.environ.get("PYRETLIFE_PRT_PATH")
    if input_pRT_path is None:
        raise RuntimeError("PYRETLIFE_PRT_PATH not set!")
    if not Path(input_pRT_path).exists():
        raise RuntimeError("PYRETLIFE_PRT_PATH set, but folder does not exist!")
    if len(glob.glob(input_pRT_path +"/petitRADTRANS/*"))==0:
        raise RuntimeError("PYRETLIFE_PRT_PATH set, but folder is not valid.")
    return Path(input_pRT_path)


def set_prt_opacity(input_prt_path,input_opacity_path) -> None:

    file_path = Path(input_prt_path) / "petitRADTRANS" / "path.txt"
    with open(file_path, "r") as path_file:
        orig_path = path_file.read()

    #LEGACY: for older versions of petitRADTRANS
    if orig_path != "#\n" + str(input_opacity_path):
        with open(file_path, "w+") as input_data:
            input_data.write("#\n" + input_opacity_path)

    # For new versions of pRT
    os.environ["pRT_input_data_path"] = input_opacity_path


def populate_dictionaries(config: dict) -> Tuple[dict, dict, dict]:
    # old read_var function
    Knowns={}
    Parameters={}
    Settings={}

    for section in config.keys():
        for subsection in config[section].keys():
            if type(config[section][subsection]) is dict:
                if "prior" in config[section][subsection].keys():
                    Parameters[subsection] = config[section][subsection]
                elif "truth" in self.config[section][subsection].keys():
                    Knowns[subsection] = config[section][subsection]
                else:
                    Settings[subsection] = config[section][subsection]
    return Knowns, Parameters, Settings

def check_temperature_parameters(config: dict) -> None:
    """
    This function checks if all temperature variables necessary
    for the given parametrization are provided by the user. If not,
    it stops the run.
    """

    # TODO: I would recommend to restructure the configuration of the
    #   temperature parameters in a more general and structured way,
    #   for example something like this:
    #   ```
    #   TEMPERATURE PARAMETERS:
    #     parametrization: polynomial
    #     parameters:
    #       a_4 = U 2 5 T 3.67756393
    #       a_3 = U 0 100 T 40.08733884
    #       a_2 = U 0 300 T 136.42147966
    #       a_1 = U 0 500 T 182.6557084
    #       a_0 = U 0 600 T 292.92802205
    #     extra_parameters:
    #       dim_z: dimensionality of the latent space
    #       file_path: /path/to/learned/PT/model
    #       ...
    #       (other parameters that are specific to the parametrization)
    #  ```
    #  This would make it easier to check if all the necessary parameters
    #  for the given parametrization are provided, and it would also make
    #  it easier to add new parametrizations in the future.

    input_pt = list(config["TEMPERATURE PARAMETERS"].keys())

    # check if all parameters are there:
    if (
        config["TEMPERATURE PARAMETERS"]["settings_parametrization"]
        == "polynomial"
    ):
        required_params = ["a_" + str(i) for i in range(len(input_pt) - 1)]
    elif "vae_pt" in config["TEMPERATURE PARAMETERS"]["parametrization"]:
        required_params = [
            "z_" + str(i + 1)
            for i in range(
                len(
                    [
                        input_pt[i]
                        for i in range(len(input_pt))
                        if "settings" not in input_pt[i]
                    ]
                )
                - 2
            )
        ]
    elif config["TEMPERATURE PARAMETERS"]["parametrization"] == "guillot":
        required_params = [
            "log_delta",
            "log_gamma",
            "t_int",
            "t_equ",
            "log_p_trans",
            "alpha",
        ]
    elif config["TEMPERATURE PARAMETERS"]["parametrization"] == "madhuseager":
        required_params = [
            "T0",
            "log_P1",
            "log_P2",
            "log_P3",
            "alpha1",
            "alpha2",
        ]
    elif (
        config["TEMPERATURE PARAMETERS"]["parametrization"]
        == "mod_madhuseager"
    ):
        required_params = ["T0", "log_P1", "log_P2", "alpha1", "alpha2"]
    elif config["TEMPERATURE PARAMETERS"]["parametrization"] == "isothermal":
        required_params = ["T_eq"]
    elif config["TEMPERATURE PARAMETERS"]["parametrization"] == "input":
        required_params = ["input_path"]
    else:
        raise RuntimeError("Unknown PT parametrization.")

    if not all(elem in input_pt for elem in required_params):
        missing_params = [_ for _ in required_params if _ not in input_pt]
        raise RuntimeError(
            "Missing one or more PT parameters/knowns. "
            "Make sure these exist:" + str(missing_params)
        )
