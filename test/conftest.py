"""Shared settings and fixtures of the pymsz test suite.

Run from the repository root:   pytest            (all tests, a few minutes)
                                pytest -k telescope   (a subset)
The tests that need the compiled SZpack extension are skipped if it is not built, unless the
environment variable PYMSZ_REQUIRE_SZPACK=1 is set (as in the GitHub workflow), which turns the skip
into a failure. The tests on simulation data use the test cluster test/snap_CL288.hdf5.
"""
import copy
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT = os.path.join(HERE, 'snap_CL288.hdf5')
SNAPSHOT_BINARY = os.path.join(HERE, 'snap_CL288')
# the test cluster, as in test/mock_CL288.yaml (comoving kpc/h)
CL288 = dict(center=[500095.89011853, 500286.04465527, 500481.30095660], radius=1446.17,
             restrict_r=False, cut_sfr=0.1)
NCPU = int(os.environ.get('PYMSZ_TEST_NCPU', min(4, os.cpu_count() or 1)))
# image geometry shared by the model fixtures and the pipeline tests (small, for speed)
GEO = dict(npixel=48, AR=40., redshift=0.1, axis='z', Ncpu=NCPU)


def pytest_report_header(config):
    import pymsz
    from pymsz import szpack
    return ["pymsz %s imported from %s" % (pymsz.version.version, os.path.dirname(pymsz.__file__)),
            "SZpack extension: %s" % ('built' if szpack.available else 'NOT built (SZpack tests skipped)')]


@pytest.fixture(scope='session')
def szpack():
    """The compiled SZpack module."""
    from pymsz import szpack as sz
    if not sz.available:
        if os.environ.get('PYMSZ_REQUIRE_SZPACK') == '1':
            pytest.fail("The SZpack extension is not built: %s" % sz._import_error)
        pytest.skip("SZpack extension not built")
    return sz.SZpack


def load_cluster(fname=SNAPSHOT):
    from pymsz import load_data
    if not os.path.exists(fname):
        pytest.skip("test snapshot %s not found" % fname)
    c = CL288
    return load_data(filename=fname, snapshot=True, center=np.array(c['center']), radius=c['radius'],
                     restrict_r=c['restrict_r'], cut_sfr=c['cut_sfr'])


@pytest.fixture(scope='session')
def cluster():
    """Gas of the test cluster (load_data)."""
    return load_cluster()


@pytest.fixture(scope='session')
def _tt(cluster):
    from pymsz import TT_model
    return TT_model(cluster, **GEO)


@pytest.fixture(scope='session')
def _tk(cluster):
    from pymsz import TK_model
    return TK_model(cluster, **GEO)


@pytest.fixture
def tt(_tt):
    """TT_model (Compton y) of the test cluster with GEO; a fresh copy for every test."""
    return copy.deepcopy(_tt)


@pytest.fixture
def tk(_tk):
    """TK_model (b = tau v/c) of the test cluster with GEO; a fresh copy for every test."""
    return copy.deepcopy(_tk)
