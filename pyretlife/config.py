"""
Read in configuration files.
"""
import os
import sys

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


def read_config_from_ini(file_path: Union[Path, str]) -> dict:
    """
    Read a configuration from an *.ini file.

    Args:
        file_path: Path to a *.ini file with the configuration.

    Returns:
        The configuration as a dictionary.
    """

    config_parser = ConfigParser(inline_comment_prefixes=("#",))
    config_parser.optionxform = str

    config_parser.read(filenames=file_path, encoding=None)

    config = {}
    for section in config_parser.sections():
        config[section] = {k: v for k, v in config_parser[section].items()}

    return config


def read_config_from_yaml(file_path: Union[Path, str]) -> dict:
    """
    Read a configuration from a *.yaml file.

    Args:
        file_path: Path to a *.yaml file with the configuration.

    Returns:
        The configuration as a dictionary.
    """

    with open(file_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def read_config_file(file_path: Union[Path, str]) -> dict:
    """
    Read a configuration from a file.

    Args:
        file_path: Path to a file with the configuration.

    Returns:
        The configuration as a dictionary.
    """

    file_path = Path(file_path)

    if file_path.suffix == ".ini":
        return read_config_from_ini(file_path)
    elif file_path.suffix == ".yaml":
        return read_config_from_yaml(file_path)
    else:
        raise ValueError(f"Unknown file extension: {file_path.suffix}")


def check_if_configs_match(config: dict) -> bool:
    # Expected location of a config file; check if it exists
    retrieval_dir = Path(config["PREFIX"]["settings_prefix"])
    if not retrieval_dir.exists():
        return True

    # Find all the config files in the retrieval directory
    try:
        config_file = next(retrieval_dir.glob("input.yaml"))
    except StopIteration:
        return True

    return not DeepDiff(
        config,
        read_config_from_yaml(config_file),
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


def fix_types_for_ini_config(config: dict) -> dict:
    pass


def validate_config(config: dict) -> None:
    check_temperature_parameters(config)
    pass


def read_paths(file_path: Union[Path, str]) -> Tuple[Path, Path, Path]:
    """
    Read the contents of a paths file (in *.txt format) that contains
    the paths to the petitRADTRANS package, the opacity data, and the
    path to the MultiNest installation.

    Args:
        file_path: The path to the paths file.

    Returns:
        A 3-tuple with the the paths to: (1) petitRADTRANS, (2) the
        opacity data, and (3) the MultiNest installation.
    """

    # TODO: You are calling this function twice, once directly in the
    #   `__init__` of the `RetrievalObject` class, and now also in the
    #   `read_paths` function, which you call from the `__init__` of
    #   the `RetrievalObject` class.
    #   Maybe try to refactor it in a way that you only call it once?
    #   I assumed that the input of the `read_path` function is the
    #   global configuration file which does not change between the
    #   retrievals? You could even consider using environment variables
    #   for the paths, and then you don't need to read the paths file.
    dict_paths = read_config_file(file_path)

    # TODO: Please don't ever use `open()` without a context manager (i.e.,
    #   the `with` part), because then you might not close the file properly.
    file_path = Path(dict_paths["path_prt"]) / "petitRADTRANS" / "path.txt"
    with open(file_path, "r") as path_file:
        orig_path = path_file.read()

    # TODO: Document what this is doing (and why).
    # LEGACY (for older versions of pRT)
    if orig_path != "#\n" + dict_paths["path_opacity"]:
        with open(file_path, "w+") as input_data:
            input_data.write("#\n" + dict_paths["path_opacity"])

    # TODO: As mentioned in the main class, don't change the state of the
    #   system in a function that is only called "read ...".
    # For new versions of pRT
    os.environ["pRT_input_data_path"] = dict_paths["path_opacity"]
    sys.path.append(dict_paths["path_prt"])

    # FIXME: You are returning the path to the petitRADTRANS package twice.
    #   The last return value should be the path to the MultiNest installation.
    return (
        Path(dict_paths["path_prt"]),
        Path(dict_paths["path_opacity"]),
        Path(dict_paths["path_prt"]),
    )


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
