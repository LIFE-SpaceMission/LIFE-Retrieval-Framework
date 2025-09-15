import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("../../pyretlife"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pyRetLIFE"
copyright = "2022, Alei, Konrad, et al."
author = "Alei, Konrad, et al."
release = "2022"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
]

autosummary_generate = True
add_module_names = False


source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}
templates_path = ["_templates"]
exclude_patterns = []

autodoc_mock_imports = ["petitRADTRANS", "normflows", "spectres", "deepdiff"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_theme_options = {
    "show_toc_level": 2,
    "repository_url": "https://github.com/LIFE-SpaceMission/LIFE-Retrieval-Framework",
    "path_to_docs": "docs/source",
    "use_issues_button": True,
    "use_repository_button": True,
}

html_logo = "logo.png"
