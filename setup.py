#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages
import subprocess

MAJOR = 1
MINOR = 0
VERSION = '%d.%d' % (MAJOR, MINOR)

# Return the git revision as a string
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION


def write_version_py(filename='pymsz/version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM SETUP.PY
version = '%(version)s'
git_revision = '%(git_revision)s'
"""
    GIT_REVISION = git_version()

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'git_revision': GIT_REVISION})
    finally:
        a.close()
    return VERSION


# ----------------------------------------------------------------------------------------------
# SZpack v2 (https://github.com/CMBSPEC/SZpack) python extension.
# The upstream sources live in ./SZpack (git submodule). They are compiled together with the
# upstream pybind11 wrappers (SZpack/python) into the extension module pymsz.szpack.SZpack.
# GSL is needed. It is searched for in (first match wins):
#   $GSL_DIR (prefix) or $GSL_INC/$GSL_LIB, `gsl-config`, ./third_party/gsl (made by
#   build_szpack.sh --build-gsl), $CONDA_PREFIX, /usr/local, /opt/local, /opt/homebrew, /usr.
# ----------------------------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
SZPACK_DIR = 'SZpack'


def find_gsl():
    """Return (include_dir, lib_dir) for GSL, or (None, None) if it cannot be found."""
    def ok(inc):
        return inc is not None and os.path.isfile(os.path.join(inc, 'gsl', 'gsl_math.h'))

    if os.environ.get('GSL_INC') or os.environ.get('GSL_LIB'):
        inc, lib = os.environ.get('GSL_INC'), os.environ.get('GSL_LIB')
        if ok(inc):
            return inc, lib
    candidates = []
    if os.environ.get('GSL_DIR'):
        candidates.append(os.environ['GSL_DIR'])
    try:
        out = subprocess.run(['gsl-config', '--prefix'], capture_output=True, text=True)
        if out.returncode == 0 and out.stdout.strip():
            candidates.append(out.stdout.strip())
    except OSError:
        pass
    candidates.append(os.path.join(HERE, 'third_party', 'gsl'))
    if os.environ.get('CONDA_PREFIX'):
        candidates.append(os.environ['CONDA_PREFIX'])
    candidates += ['/usr/local', '/opt/local', '/opt/homebrew', '/usr']
    for prefix in candidates:
        inc = os.path.join(prefix, 'include')
        if ok(inc):
            for libdir in ('lib', 'lib64', os.path.join('lib', 'x86_64-linux-gnu'),
                           os.path.join('lib', 'aarch64-linux-gnu')):
                lib = os.path.join(prefix, libdir)
                if os.path.exists(os.path.join(lib, 'libgsl.a')) or \
                        any(f.startswith('libgsl.') for f in os.listdir(lib) if os.path.isdir(lib)):
                    return inc, lib
            return inc, None
    return None, None


def szpack_extensions():
    """Build description of the SZpack extension; empty list if it cannot be built."""
    from glob import glob

    if not os.path.isfile(os.path.join(HERE, SZPACK_DIR, 'SZpack.cpp')):
        print("pymsz: ./SZpack not found -> SZpack models are NOT built.\n"
              "       Run `git submodule update --init` (or ./build_szpack.sh) first.")
        return [], {}
    try:
        from pybind11.setup_helpers import Pybind11Extension, build_ext, ParallelCompile
    except ImportError:
        print("pymsz: pybind11 not installed -> SZpack models are NOT built. `pip install pybind11`.")
        return [], {}
    gsl_inc, gsl_lib = find_gsl()
    if gsl_inc is None:
        print("pymsz: GSL not found -> SZpack models are NOT built.\n"
              "       Install GSL (e.g. `apt install libgsl-dev`, `conda install gsl`), set GSL_DIR,\n"
              "       or run `./build_szpack.sh --build-gsl` to build a local copy.")
        return [], {}
    ParallelCompile("NPY_NUM_BUILD_JOBS", default=0).install()

    sp = SZPACK_DIR
    # same source list as the upstream SZpack/python/setup.py
    sources = sorted(glob(os.path.join(sp, 'python', '*.cpp')) +
                     glob(os.path.join(sp, 'python', 'MultipoleKernel', '*.cpp')) +
                     glob(os.path.join(sp, 'python', 'VariedBackgroundAsymptotic', '*.cpp')) +
                     glob(os.path.join(sp, 'StartUp', 'global_variables.cpp')) +
                     glob(os.path.join(sp, 'Tools', 'Cosmology', '*.cpp')) +
                     glob(os.path.join(sp, 'Tools', 'Simple_routines', '*.cpp')) +
                     glob(os.path.join(sp, 'Tools', 'Integration', '*.cpp')) +
                     glob(os.path.join(sp, 'src', '*.cpp')) +
                     [os.path.join(sp, 'SZpack.cpp')])
    include_dirs = [sp] + [os.path.join(sp, d) for d in
                           ('include', 'src', 'StartUp', 'Tools/Definitions', 'Tools/Cosmology',
                            'Tools/Integration', 'Tools/Simple_routines')] + [gsl_inc]
    ext = Pybind11Extension(
        'pymsz.szpack.SZpack', sources,
        include_dirs=include_dirs,
        library_dirs=[gsl_lib] if gsl_lib else [],
        runtime_library_dirs=[gsl_lib] if gsl_lib and not gsl_lib.startswith('/usr/lib') else [],
        libraries=['gsl', 'gslcblas', 'm'],
        define_macros=[('UsePybind11', None)],
        extra_compile_args=['-O2', '-w'],  # upstream code is noisy; silence warnings
        language='c++',
    )
    print("pymsz: building SZpack extension with GSL from %s" % os.path.dirname(gsl_inc))
    return [ext], {'build_ext': build_ext}


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


EXT_MODULES, CMDCLASS = szpack_extensions()

setup(
    name="pymsz",
    ext_modules=EXT_MODULES,
    cmdclass=CMDCLASS,
    version=write_version_py(),
    author="Weiguang Cui",
    author_email="cuiweiguang@gmail.com",
    description="Mock Sunyaev-Zel'dovich observations of hydrodynamical simulations",
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    url="https://github.com/weiguangcui/pymsz",
    project_urls={
        'Documentation': 'https://pymsz.readthedocs.io',
        'Source': 'https://github.com/weiguangcui/pymsz',
        'Issues': 'https://github.com/weiguangcui/pymsz/issues',
        'Changelog': 'https://pymsz.readthedocs.io/en/latest/changelog.html',
    },
    python_requires='>=3.10',
    packages=find_packages(exclude=['SZpack*', 'third_party*', 'test*']),
    install_requires=['numpy', 'scipy', 'astropy', 'h5py', 'psutil'],
    extras_require={
        'yaml': ['pyyaml'],      # YAML parameter files of pymsz-mock (TOML and JSON need nothing extra)
        'cmb': ['camb'],         # primary CMB power spectrum computed on the fly (observation.cmb: camb)
        'yt': ['yt'],            # load_data(..., yt_load=True): experimental, only TT_model
        'test': ['pytest', 'pyyaml'],   # pip install -e ".[test]"; pytest
    },
    package_data={
        # measured telescope bandpasses (pymsz.bandpasses), parameter template of pymsz-mock
        'pymsz': ['data/bandpasses/*.md', 'data/bandpasses/*/*', 'data/*.yaml'],
        '': ['*.fits',
             '*README*',
             #  'models/*.model',
             #  'filters/*',
             # 'SZpacklib/__init__.py',
             # 'SZpacklib/SZpack.py',
            #  'SZpacklib/_SZpack*.so'
             ]},
    license="GPL-3.0-or-later",
    include_package_data=True,
    # mock observations from a parameter file (pymsz.pipeline); also: python -m pymsz
    entry_points={'console_scripts': ['pymsz-mock = pymsz.pipeline:main']},
    keywords='astronomy astrophysics cosmology Sunyaev-Zeldovich SZ galaxy clusters hydrodynamical '
             'simulation mock observation CMB',
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
    ]
)
