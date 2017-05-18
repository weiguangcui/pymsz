#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess


class InstallSZpack(install, object):
    """Custom handler for the 'install' command."""

    def run(self):
        self._compile_and_install_software()
        super(InstallSZpack, self).run()

    def _compile_and_install_software(self):
        """Used the subprocess module to compile the C software."""
        src_path = './pymsz/SZpack.v1.1.1/'

        print("making SZpack models...")
        subprocess.check_call('make all', cwd=src_path, shell=True)


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="pymsz",
    version="1.0.0",
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
             #  'refs/*',
             'SZpack.v1.1.1/SZpack.py',
             'SZpack.v1.1.1/libSZpack.a']},
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
