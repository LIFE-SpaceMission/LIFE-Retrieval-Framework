"""
Setup script to install PyRetLIFE as a Python package.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from os.path import join, dirname
from setuptools import find_packages, setup


# -----------------------------------------------------------------------------
# RUN setup() FUNCTION
# -----------------------------------------------------------------------------

# Get version from VERSION file
with open(join(dirname(__file__), 'pyretlife/VERSION')) as version_file:
    version = version_file.read().strip()

# Run setup()
setup(
    name='pyretlife',
    version=version,
    description='PyRetLIFE: Python-based retrievals for LIFE',
    author='Bjoern Konrad, Eleonora Alei, and others',
    install_requires=[
        'astropy',
        'matplotlib',
        'pymultinest',
        'numpy',
        'scipy',
    ],
    extras_require={
        'develop': [
            'coverage',
            'flake8',
            'mypy',
            'pylint',
            'pytest',
            'pytest-cov',
        ]
    },
    packages=find_packages(),
    zip_safe=False,
)
