"""
pymsz is a package written primarily in Python. It is designed to mimic
SZ observations from hydro-dynamical simulations.

More details of the functions and models.
"""

__author__ = 'Weiguang Cui'
__email__ = 'cuiweiguang@gmail.com'
__ver__ = 'beta'

# import numpy as np  # For modern purposes
from pymgal.SSP_models import SSP_models
from pymgal.load_data import load_data
from pymgal.filters import filters
from pymgal import utils
from pymgal import dusts
