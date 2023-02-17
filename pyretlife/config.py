"""
Read in configuration files.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from configparser import ConfigParser
from pathlib import Path
from typing import Union

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
    pass
