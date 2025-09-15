"""
Unit tests for config.ini.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

from configparser import ConfigParser

import pytest
import yaml

# from pyretlife.retrieval.configuration_ingestion import (
#    read_config_from_ini,
#    read_config_from_yaml,
# )

# from pyretlife.retrieval.configuration_ingestion import read_config_from_ini, read_config_from_yaml


# -----------------------------------------------------------------------------
# UNIT TESTS
# -----------------------------------------------------------------------------


@pytest.fixture
def config() -> dict:
    config = {
        "section_1": {"a": 1, "b": "test"},
        "section_2": {"c": 3.141, "d": False},
    }

    return config


@pytest.fixture
def path_to_ini(tmp_path: Path, config: dict) -> Path:
    config_parser = ConfigParser()
    for section, values in config.items():
        config_parser[section] = values

    file_path = tmp_path / "config.ini"
    with open(file_path, "w") as f:
        config_parser.write(f)

    return file_path


@pytest.fixture
def path_to_yaml(tmp_path: Path, config: dict) -> Path:
    file_path = tmp_path / "config.yaml"
    with open(file_path, "w") as yaml_file:
        yaml.dump(config, yaml_file)

    return file_path


# def test__read_config_from_ini(path_to_ini: Path) -> None:
#     config = read_config_from_ini(path_to_ini)
#
#     assert "section_1" in config
#     assert "section_2" in config
#     assert config["section_1"]["a"] == "1"
#     assert config["section_1"]["b"] == "test"
#     assert config["section_2"]["c"] == "3.141"
#     assert config["section_2"]["d"] == "False"


# def test__read_config_from_yaml(path_to_yaml: Path, config: dict) -> None:
# read_config = read_config_from_yaml(path_to_yaml)
# assert read_config == config
