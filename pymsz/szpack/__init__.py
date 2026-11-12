"""
Python access to SZpack v2 (https://github.com/CMBSPEC/SZpack).

SZpack computes the thermal and kinematic SZ effect including relativistic corrections, either by
explicit numerical integration of the Boltzmann collision term ("3D", "5D") or with fast analytic
approximations ("asymptotic", "CNSN", "CNSN_opt", "combo", and the moment method "means").

The extension module ``pymsz.szpack.SZpack`` is compiled from the upstream sources in the
top-level ``SZpack/`` folder (git submodule) by ``setup.py`` / ``build_szpack.sh``. It is the full
upstream pybind11 wrapper, so everything documented in ``SZpack/python/PySZpack.ipynb`` is available::

    >>> from pymsz.szpack import SZpack
    >>> SZpack.Integral3D.compute_from_variables(xcmb, The, betac, muc, 1e-4, "all", DI=True)

If you use it, please cite the SZpack papers listed in ``SZpack/README.md``
(Chluba, Nagai, Sazonov, Nelson 2012; Chluba, Switzer, Nagai, Nelson 2012; Lee & Chluba ...).
"""

import importlib

try:
    SZpack = importlib.import_module('pymsz.szpack.SZpack')  # compiled: pymsz/szpack/SZpack.*.so
    available = True
    _import_error = None
except ImportError as _err:  # extension not built (missing GSL / pybind11 / submodule)
    SZpack = None
    available = False
    _import_error = _err


def require():
    """Return the compiled SZpack module or raise an ImportError explaining how to build it."""
    if not available:
        raise ImportError(
            "The SZpack extension (pymsz.szpack.SZpack) is not built: %s\n"
            "Build it from the pymsz source directory with\n"
            "    git submodule update --init        # fetch ./SZpack\n"
            "    pip install -e .   (or: python setup.py build_ext --inplace)\n"
            "GSL is required: install libgsl-dev / `conda install gsl`, set GSL_DIR, or run\n"
            "    ./build_szpack.sh --build-gsl" % _import_error)
    return SZpack


__all__ = ['SZpack', 'available', 'require']
