"""SZ spectra and unit conversions (pymsz.sz_spectra): analytic limits, no simulation data."""
import numpy as np
import pytest
from scipy.optimize import brentq

from pymsz import sz_spectra as sp


def test_frequency_x_roundtrip():
    f = np.array([30., 150., 857.])
    assert np.allclose(sp.x_to_freq(sp.freq_to_x(f)), f, rtol=1e-14)
    assert sp.x_to_freq(1.) == pytest.approx(56.79, abs=0.01)  # x = 1 for Tcmb = 2.7255 K


def test_intensity_unit():
    assert sp.I0() == pytest.approx(270.07, rel=1e-4)  # 2 (k Tcmb)^3 / (h c)^2 in MJy/sr


def test_tsz_null_at_217GHz():
    x0 = brentq(sp.tsz_nonrel, 3., 4.5)
    assert x0 == pytest.approx(3.8300, abs=1e-4)
    assert sp.x_to_freq(x0) == pytest.approx(217.5, abs=0.05)


def test_tsz_rayleigh_jeans_limit():
    # Delta T / Tcmb = -2 y at low frequencies
    assert sp.tsz_nonrel(1e-3) / sp.G(1e-3) == pytest.approx(-2., rel=1e-6)


def test_ksz_is_a_temperature_change():
    # Delta n of a CMB temperature change dT is G(x) dT/Tcmb: the kSZ gives Delta T/Tcmb = tau v/c
    f = np.array([90., 150., 220., 353.])
    b = 1.0e-5
    dT = sp.dn_to_unit(b * sp.ksz_nonrel(sp.freq_to_x(f))[:, None], f, 'K')
    assert np.allclose(dT, sp.TCMB * b, rtol=1e-12)


def test_dB_dT_is_the_planck_derivative():
    f = np.array([30., 150., 545.])

    def planck(T):  # MJy/sr
        x = sp.freq_to_x(f, T)
        return sp.I0(T) * x**3 / np.expm1(x)

    dT = 1e-4
    assert np.allclose(sp.dB_dT(f), (planck(sp.TCMB + dT) - planck(sp.TCMB - dT)) / (2 * dT), rtol=1e-7)


@pytest.mark.parametrize('unit', sp.UNITS)
def test_unit_roundtrip(unit):
    f = [90., 150., 220.]
    dn = np.random.default_rng(0).normal(size=(3, 4, 5)) * 1e-5
    out = sp.dn_to_unit(dn, f, unit, pixel_arcsec=10.)
    assert np.allclose(sp.unit_to_dn(out, f, unit, pixel_arcsec=10.), dn, rtol=1e-12, atol=0)


def test_jy_per_pixel_is_surface_brightness_times_pixel_area():
    f = [150.]
    dn = np.full((1, 2, 2), 1e-5)
    mjy_sr = sp.dn_to_unit(dn, f, 'MJy/sr')
    jy = sp.dn_to_unit(dn, f, 'Jy/pixel', pixel_arcsec=10.)
    assert np.allclose(jy, mjy_sr * 1e6 * (10. * sp.ARCSEC)**2, rtol=1e-12)


def test_bad_units():
    with pytest.raises(ValueError):
        sp.check_unit('mK')
    with pytest.raises(ValueError):
        sp.dn_to_unit(np.ones((1, 2, 2)), [150.], 'Jy/pixel')  # needs the pixel size


def test_interpolate_dn():
    f = np.linspace(80., 400., 30)
    x = sp.freq_to_x(f)
    dn = sp.tsz_nonrel(x)[:, None, None] * np.ones((1, 2, 3))
    new = np.array([95., 150., f[7], 353.])
    out = sp.interpolate_dn(dn, f, new)
    assert np.array_equal(out[2], dn[7])  # present frequencies are copied exactly
    assert np.allclose(out[:, 0, 0], sp.tsz_nonrel(sp.freq_to_x(new)), rtol=1e-4, atol=1e-6)
    with pytest.raises(ValueError, match='extrapolated'):
        sp.interpolate_dn(dn, f, [500.])
