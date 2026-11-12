"""
pymsz is a package written primarily in Python. It is designed to mimic
SZ observations from hydro-dynamical simulations.

More details of the functions and models.
"""

# import numpy as np  # For modern purposes
from pymsz.Theoretical_models import TT_model, TK_model
from pymsz.load_data import load_data
from pymsz.SZpack_models import SZpack_model  # needs the compiled SZpack extension at use time
from pymsz.telescope import Telescope, Band, SkyMaps, MockObservation
from pymsz import sz_spectra, bandpasses
from pymsz import pipeline  # mock observations from a parameter file: pymsz.pipeline.run('params.yaml')
from pymsz import version

__author__ = 'Weiguang Cui'
__email__ = 'cuiweiguang@gmail.com'
__verion__ = version.version
__all__ = ["pymsz"]
