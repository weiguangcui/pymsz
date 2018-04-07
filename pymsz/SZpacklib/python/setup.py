#===================================================================================================
#! /usr/bin/env python
#===================================================================================================

# System imports
from distutils.core import *
from distutils      import sysconfig

# Third-party modules - we depend on numpy for everything
import numpy

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# sz_wrap extension module
_SZpack = Extension("_SZpack",
                   ["SZpack.i","SZpack.python.cpp"],
                   include_dirs = [numpy_include, "./.", "../.", "../include",
                                   "/usr/include/gsl/","/usr/include/"],
                   libraries = ['SZpack', 'gsl', 'gslcblas'],
                   library_dirs=['./.','../.',"/usr/lib/x86_64-linux-gnu/","/usr/lib/"],
                   extra_compile_args = ['-fPIC'],
                   swig_opts=['-modern', '-I../include', '-c++']
                   )

# NumyTypemapTests setup
setup(  name        = "SZpack",
        description = "This package provides the main SZpack functions for Python",

        author      = "E. Switzer & J. Chluba",
        version     = "1.0",
        ext_modules = [_SZpack],
        py_modules  = ["SZpack"],
        )

#===================================================================================================
#===================================================================================================
