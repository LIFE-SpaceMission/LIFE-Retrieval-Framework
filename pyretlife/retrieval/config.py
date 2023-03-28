"""
Read in configuration files.
"""
import os
import glob
import numpy as np

import astropy.units as u

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Union, Tuple

import hashlib
import yaml

from deepdiff import DeepDiff

from pyretlife.retrieval.UnitsUtil import UnitsUtil


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
        raise ValueError(
            f"Unknown file extension: {file_path.suffix}. Please convert it to a .yaml file."
        )


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


# def compute_hash_of_config_file(file_path: Union[Path, str]) -> str:
#     """
#     Compute the hash of a configuration file.
#
#     Args:
#         file_path: Path to a file with the configuration.
#
#     Returns:
#         The hash of the configuration file.
#     """
#
#     file_path = Path(file_path)
#     with open(file_path, "rb") as f:
#         return hashlib.sha256(f.read()).hexdigest()

#
# def convert_ini_to_yaml(file_path: Union[Path, str]) -> None:
#     pass


def make_output_folder(folder_path: Union[Path, str]) -> None:
    folder_path = Path(folder_path)
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)


def populate_dictionaries(
    config: dict,
    knowns: dict,
    parameters: dict,
    settings: dict,
    units: UnitsUtil,
) -> Tuple[dict, dict, dict, UnitsUtil]:

    if "USER-DEFINED UNITS" in config.keys():
        for key in config["USER-DEFINED UNITS"]:
            units.custom_unit(
                key, u.Quantity(config["USER-DEFINED UNITS"][key])
            )
    linelist=[]
    for section in config.keys():
        if section != "USER-DEFINED UNITS":
            for subsection in config[section].keys():
                if (
                    type(config[section][subsection]) is dict
                    and "prior" in config[section][subsection].keys()
                ):
                    parameters[subsection] = config[section][
                        subsection
                    ]
                    if "unit" in config[section][subsection].keys():
                        input_unit = u.Unit(config[section][subsection]["unit"])
                    else:
                        input_unit = units.return_units(
                            subsection, units.std_input_units
                        )
                    parameters[subsection]["unit"] = input_unit
                    parameters[subsection]['type'] = section

                elif (
                    type(config[section][subsection]) is dict
                    and "truth" in config[section][subsection].keys()
                ):
                    knowns[subsection] = config[section][subsection]
                    if "unit" in config[section][subsection].keys():
                        input_unit = u.Unit(config[section][subsection]["unit"])
                    else:
                        input_unit = units.return_units(
                            subsection, units.std_input_units
                        )
                    knowns[subsection]["unit"] = input_unit
                    knowns[subsection]['type'] = section

                else:
                    settings[subsection] = config[section][subsection]

                # read lists if available. Can be a str or list.
                if (
                    type(config[section][subsection]) is dict
                    and "lines" in config[section][subsection].keys()
                ):
                    if isinstance(config[section][subsection]['lines'],str):
                        linelist.append(config[section][subsection]['lines'])
                    else:
                        linelist.extend(config[section][subsection]['lines'])
        settings['opacity_linelist']=linelist
    return knowns, parameters, settings, units


def load_data(settings: dict, units: UnitsUtil, retrieval: bool = True) -> dict:
    result_dir = settings["output_folder"]
    instrument = {}
    for data_file in settings["data_files"].keys():
        input_string = settings["data_files"][data_file]["path"]
        # Case handling for the retrieval plotting
        if not retrieval:
            if os.path.isfile(
                result_dir
                + "/input_"
                + input_string.split("/")[-1].split(" ")[0]
            ):
                input_string = (
                    result_dir + "/input_" + input_string.split("/")[-1]
                )
            else:
                input_string = (
                    result_dir
                    + "/input_spectrum.txt "
                    + " ".join(input_string.split("/")[-1].split(" ")[1:])
                )

        input_data = np.genfromtxt(input_string)

        # retrieve units
        if "unit" in settings["data_files"][data_file].keys():
            input_unit_wavelength = u.Unit(
                settings["data_files"][data_file]["unit"].split(",")[0]
            )
            input_unit_flux = u.Unit(
                settings["data_files"][data_file]["unit"].split(",")[1]
            )
        else:
            input_unit_wavelength = units.return_units(
                "wavelength", units.std_input_units
            )
            input_unit_flux = units.return_units("flux", units.std_input_units)

        # trim spectrum
        input_data = input_data[
            input_data[:, 0]
            >= (
                    settings["wavelength_range"][0]
                    * units.return_units("WMIN", units.std_input_units)
            )
            .to(input_unit_wavelength)
            .value
        ]
        input_data = input_data[
            input_data[:, 0]
            <= (
                    settings["wavelength_range"][1]
                    * units.return_units("WMAX", units.std_input_units)
            )
            .to(input_unit_wavelength)
            .value
        ]
        instrument[data_file] = {
            "input_data": input_data,
            "input_unit_wavelength": input_unit_wavelength,
            "input_unit_flux": input_unit_flux,
        }
    return instrument


def get_check_opacity_path() -> Path:
    """
    The get_check_opacity_path function checks that the PYRETLIFE_OPACITY_PATH environment variable is set.
    If it is not, an error message is printed and the program exits. If it is set, then a Path object pointing to
    the opacity folder in this directory will be returned.
    Returns:
        The path to the opacity folder
    """

    input_opacity_path = os.environ.get("PYRETLIFE_OPACITY_PATH")
    if input_opacity_path is None:
        raise RuntimeError("PYRETLIFE_OPACITY_PATH not set!")
    if not Path(input_opacity_path).exists():
        raise RuntimeError(
            "PYRETLIFE_OPACITY_PATH set, but folder does not exist!"
        )
    if len(glob.glob(input_opacity_path + "/opacities/*")) == 0:
        raise RuntimeError(
            "PYRETLIFE_OPACITY_PATH set, but folder is not valid."
        )
    return Path(input_opacity_path)


def get_check_prt_path() -> Path:
    """
    The get_check_pRT_path function checks that the PYRETLIFE_PRT_PATH environment variable is set, and if so,
    checks that it points to a valid folder. If all these conditions are met, then the function returns a Path object
    pointing to this folder.
    Returns:
        The path to the petitRADTRANS folder

    """

    input_pRT_path = os.environ.get("PYRETLIFE_PRT_PATH")
    if input_pRT_path is None:
        raise RuntimeError("PYRETLIFE_PRT_PATH not set!")
    if not Path(input_pRT_path).exists():
        raise RuntimeError("PYRETLIFE_PRT_PATH set, but folder does not exist!")
    if len(glob.glob(input_pRT_path + "/petitRADTRANS/*")) == 0:
        raise RuntimeError("PYRETLIFE_PRT_PATH set, but folder is not valid.")
    return Path(input_pRT_path)


def set_prt_opacity(input_prt_path: Union[Path,str], input_opacity_path:Union[Path,str]) -> None:
    file_path = Path(input_prt_path) / "petitRADTRANS" / "path.txt"
    with open(file_path, "r") as path_file:
        orig_path = path_file.read()

    # LEGACY: for older versions of petitRADTRANS
    # if orig_path != "#\n" + str(input_opacity_path):
    #     with open(file_path, "w+") as input_data:
    #         input_data.write("#\n" + input_opacity_path)

    # For new versions of pRT
    os.environ["pRT_input_data_path"] = str(input_opacity_path)
