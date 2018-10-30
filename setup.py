#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess

MAJOR = 0
MINOR = 7
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


class InstallSZpack(install, object):
    """Custom handler for the 'install' command."""

    def run(self):
        self._compile_and_install_software()
        super(InstallSZpack, self).run()

    def _compile_and_install_software(self):
        """Used the subprocess module to compile the C software."""
        src_path = './pymsz/SZpacklib/'

        print("making SZpack models...")
        subprocess.check_call('make all; make SZpack.py; ', cwd=src_path, shell=True)


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="pymsz",
    version=write_version_py(),
    author="Weiguang Cui",
    author_email="cuiweiguang@gmail.com",
    description="A Package for Mock SZ Observations",
    long_description=read('README.md'),
    packages=find_packages(),
    cmdclass={'install': InstallSZpack},
    requires=['numpy', 'pyfits', 'scipy', 'astropy'],
    package_data={
        '': ['*.fits',
             '*README*',
             #  'models/*.model',
             #  'filters/*',
             # 'SZpacklib/__init__.py',
             # 'SZpacklib/SZpack.py',
             'SZpacklib/_SZpack*.so'
             ]},
    license="BSD",
    include_package_data=True,
    keywords='astronomy astrophysics hydro-dynamical simulation mock observation',
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3"
    ]
)
