"""
pymsz is a package written primarily in Python. It is designed to mimic
SZ observations from hydro-dynamical simulations.

More details of the functions and models.
"""

# import numpy as np  # For modern purposes
from pymsz.Theoretical_models import TT_model, TK_model
from pymsz.load_data import load_data
from pymsz.SZpack_models import SZpack_model
from pymsz.SZpacklib import SZpack
from pymsz import version

__author__ = 'Weiguang Cui'
__email__ = 'cuiweiguang@gmail.com'
__verion__ = version.version
__all__ = ["pymsz", "SZpacklib"]
