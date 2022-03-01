# -*- coding: UTF-8 -*-
#! /usr/bin/python

from pathlib import Path
from setuptools import setup, find_packages

# ...
# Read library version into '__version__' variable
path = Path(__file__).parent / 'manapy' / 'version.py'
exec(path.read_text())
# ...

NAME    = 'manapy'
VERSION = __version__
AUTHOR  = 'Imad Kissami'
EMAIL   = 'imad.kissami@gmail.ma'
URL     = 'https://github.com/pyccel/manapy'
DESCR   = 'TODO.'
KEYWORDS = ['math']
LICENSE = "LICENSE"

setup_args = dict(
    name                 = NAME,
    version              = VERSION,
    description          = DESCR,
    #long_description     = open('README.rst').read(),
    author               = AUTHOR,
    author_email         = EMAIL,
    license              = LICENSE,
    keywords             = KEYWORDS,
#    url                  = URL,
)

# ...
packages = find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"])
# ...

# Dependencies
#install_requires = [
#    'numpy',
#    'meshio<4',
#    'mpi4py',
#    'mgmetis',
#    'numba==0.53.0',
#    'scipy',
#    ]

install_requires = [
    'wheel',
    'mpi4py',
    'Arpeggio==1.10.2',
    'cycler==0.11.0',
    'Cython==0.29.24',
    'filelock==3.3.2',
    'h5py==3.5.0',
    'kiwisolver==1.3.2',
    'llvmlite==0.36.0',
    'lxml==4.6.4',
    'matplotlib==3.4.3',
    'meshio==3.1.1',
    'mpmath==1.2.1',
    'numba==0.53.0',
    'numpy==1.21.4',
    'Pillow==8.4.0',
    'pyccel==1.4.1',
    'pyparsing==3.0.4',
    'python-dateutil==2.8.2',
    'scipy==1.7.2',
    'six==1.16.0',
    'sympy==1.9',
    'termcolor==1.1.0',
    'textX==2.3.0',
    'mgmetis==0.1.1',
    #'pymumps==0.3.2'
    ]

def setup_package():
    setup(packages=packages, \
          include_package_data=True, \
          install_requires=install_requires, \
         **setup_args)

if __name__ == "__main__":
    setup_package()
