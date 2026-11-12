"""SZ models of the test cluster (load_data, TT_model, TK_model, SZpack_model): the independent code
paths must agree with each other and with the analytic limits."""
import numpy as np
import pytest

from conftest import GEO, SNAPSHOT_BINARY, load_cluster
from pymsz import sz_spectra as sp

F150 = np.array([150.])
G150 = sp.G(sp.freq_to_x(150.))
TSZ150 = sp.tsz_nonrel(sp.freq_to_x(150.))


@pytest.fixture(scope='module')
def sz_nonrel(cluster, szpack):
    """SZpack_model, non-relativistic tSZ + kSZ, particle-by-particle line-of-sight sum."""
    from pymsz import SZpack_model
    return SZpack_model(cluster, F150, method='nonrel', kinetic=True, unit='Dn', **GEO)


@pytest.fixture(scope='module')
def sz_combo(cluster, szpack):
    """SZpack_model, relativistic tSZ only."""
    from pymsz import SZpack_model
    return SZpack_model(cluster, F150, method='combo', kinetic=False, unit='Dn', **GEO)


def test_load_data(cluster):
    n = np.size(cluster.temp)
    assert n > 300000
    for k in ('pos', 'vel'):
        assert np.shape(getattr(cluster, k)) == (n, 3)
    assert np.all(cluster.temp > 0) and np.all(cluster.hsml > 0) and np.all(cluster.mass > 0)
    assert np.all(np.abs(cluster.pos) <= 1446.17 * 1.0001)  # restrict_r false: a cube of half side radius
    assert cluster.cosmology['h'] == pytest.approx(0.6777)


def test_gadget_binary_snapshot(cluster):
    # the same cluster in Gadget binary format. The temperatures differ by ~2 % because the binary file
    # has only the total metallicity (another mean molecular weight); the rest must agree.
    b = load_cluster(SNAPSHOT_BINARY)
    assert np.size(b.temp) == pytest.approx(np.size(cluster.temp), abs=10)
    for k in ('mass', 'ne', 'hsml'):
        assert np.sum(getattr(b, k)) == pytest.approx(np.sum(getattr(cluster, k)), rel=1e-3), k
    assert np.median(b.temp) == pytest.approx(np.median(cluster.temp), rel=0.03)


def test_y_map(tt):
    assert tt.ydata.shape == (48, 48) and tt.ar == 40.
    assert np.all(tt.ydata >= 0)
    assert 1e-5 < tt.ydata.max() < 1e-3
    peak = np.unravel_index(np.argmax(tt.ydata), tt.ydata.shape)
    assert np.hypot(peak[0] - 23.5, peak[1] - 23.5) < 4  # the cluster is centred
    Ty = tt.Tydata[tt.ydata > 0.1 * tt.ydata.max()]
    assert 1. < np.median(Ty) < 15.  # keV, y-weighted temperature in the core


def test_integrated_y_does_not_depend_on_the_pixels(cluster, tt):
    from pymsz import TT_model
    coarse = TT_model(cluster, **dict(GEO, npixel=24, AR=80.))
    assert coarse.ydata.sum() * 80.**2 == pytest.approx(tt.ydata.sum() * 40.**2, rel=0.01)


def test_tt_spectrum(tt):
    out = tt.to_sz([150., 220., 353.], unit='Dn')
    x = sp.freq_to_x([150., 220., 353.])
    assert np.allclose(out, sp.tsz_nonrel(x)[:, None, None] * tt.ydata[None], rtol=1e-12, atol=0)
    uk = tt.to_sz(F150, unit='uK')[0]
    assert np.allclose(uk, sp.TCMB * 1e6 * TSZ150 / G150 * tt.ydata, rtol=1e-12)


def test_tk_map(tk):
    assert tk.bdata.shape == (48, 48)
    assert 0 < np.abs(tk.bdata).max() < 1e-4
    out = tk.to_sz(F150, unit='uK')[0]
    assert np.allclose(out, sp.TCMB * 1e6 * tk.bdata, rtol=1e-12)


def test_szpack_nonrel_matches_tt_tk(sz_nonrel, tt, tk):
    # summing Delta n particle by particle along the line of sight with the non-relativistic spectrum
    # gives exactly y g(x) + b G(x) of the y and b maps
    expect = tt.ydata * TSZ150 + tk.bdata * G150
    assert np.allclose(sz_nonrel.szdata[0], expect, rtol=0, atol=1e-6 * np.abs(expect).max())
    assert np.allclose(sz_nonrel.ydata, tt.ydata, rtol=1e-10, atol=0)


def test_szpack_relativistic(sz_combo, tt, szpack):
    total_nonrel = (tt.ydata * TSZ150).sum()
    ratio = sz_combo.szdata[0].sum() / total_nonrel
    assert 0.9 < ratio < 0.99  # relativistic reduction of the decrement at 150 GHz for a few keV
    # first-order approximation: TT_model with the relativistic spectrum at the y-weighted Te
    rel = tt.to_sz(F150, unit='Dn', relativistic=True)[0]
    assert rel.sum() == pytest.approx(sz_combo.szdata[0].sum(), rel=1e-3)
    assert np.allclose(rel, sz_combo.szdata[0], rtol=0, atol=5e-3 * np.abs(rel).max())


def test_szpack_fits_roundtrip(sz_combo, tmp_path):
    from pymsz import SkyMaps
    f = str(tmp_path / 'sz.fits')
    sz_combo.write_fits_image(f)
    sky = SkyMaps.from_fits(f)
    assert sky.ar == pytest.approx(40.) and np.allclose(sky.freqs, F150)
    assert np.allclose(sky.sz_maps(F150, 'Dn'), sz_combo.szdata, rtol=1e-5, atol=1e-6 * np.abs(sz_combo.szdata).max())
