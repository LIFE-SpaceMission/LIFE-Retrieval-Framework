# PyRetLIFE 🏴‍☠ ️🪐 
 
Python-based retrievals for LIFE (© by Eleonora)

---

## ⚡ Quickstart

### 📦 Installation

The code in this repository is organized as a Python package named `pyretlife`.
Clone this repository and install `pyretlife` using pip as follows:

```bash
git clone https://github.com/konradbjorn/Terrestrial_Retrieval ;
cd Terrestrial_Retrieval ;
pip install -e .
```

The `-e` option installs the package in "editable" mode, which means that any changes you make to the code will be immediately available to you without having to reinstall the package.

If you need developer tools (e.g., for running tests, linting, or type-checking), you can install the package with the optional `[develop]` option:

```bash
pip install -e ".[develop]"
```


### 🐭 Running unit tests

To run the unit tests, use the following command (requires `pytest`):

```bash
pytest tests
```

If you also want to get a coverage report, use this (requires `coverage` and `pytest-cov`):

```bash
pytest --cov-report term-missing --cov=pyretlife
```
