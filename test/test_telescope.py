"""Shipped bandpasses, telescope presets and the mock observation (pymsz.telescope) on synthetic skies."""
import re

import numpy as np
import pytest

from pymsz import bandpasses
from pymsz import sz_spectra as sp
from pymsz.telescope import Band, SkyMaps, Telescope

PRESETS = Telescope.list_presets()
PRESET_BANDS = [(p, b.name) for p in PRESETS for b in Telescope.preset(p).bands]


# ------------------------------------------------------------------------------------------------
# bandpasses and band averages
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('name', bandpasses.available())
def test_shipped_bandpass(name):
    bp = bandpasses.load(name)
    nu, r = np.asarray(bp.nu), np.asarray(bp.response)
    assert nu.ndim == 1 and nu.shape == r.shape and nu.size > 10
    assert np.all(np.diff(nu) >= 0) and nu[0] > 0  # (the Planck HFI857 RIMO repeats one sample)
    assert np.all(np.isfinite(r)) and r.max() == pytest.approx(1.)
    # the centroid is close to the nominal frequency in the name (e.g. Planck/HFI143, ACT-DR6/pa5_f090)
    nominal = float(re.search(r'(?:f|HFI|LFI)(\d{3})', name.split('/')[-1]).group(1))
    centroid = np.trapezoid(nu * r, nu) / np.trapezoid(r, nu)
    assert centroid == pytest.approx(nominal, rel=0.15)


def test_unknown_bandpass():
    assert not bandpasses.is_available('Planck/HFI999')
    with pytest.raises((KeyError, ValueError)):
        bandpasses.load('Planck/HFI999')


@pytest.mark.parametrize('preset', PRESETS)
def test_preset(preset):
    tel = Telescope.preset(preset)
    f = tel.frequencies()
    assert np.all(np.diff(f) > 0)
    assert f[0] > 5.7 and f[-1] < 2840.  # inside the SZpack range 0.1 < x < 50
    assert np.all(np.isfinite(tel.tsz_response()))


@pytest.mark.parametrize('preset,band', PRESET_BANDS)
def test_band_calibration(preset, band):
    b = Telescope.preset(preset)[band]
    # a CMB temperature change of 1 K is 1 K_CMB in every band, whatever the bandpass
    assert b.response(sp.ksz_nonrel, 'K_CMB') == pytest.approx(sp.TCMB, rel=1e-10)
    # tSZ: decrement below the null, increment above
    y_uk = b.response(sp.tsz_nonrel)
    if b.freq < 180.:
        assert y_uk < -2e6  # uK_CMB per unit y
    elif b.freq > 260.:
        assert y_uk > 1e6
    # the band average from the SZ images at the nodes reproduces the full integral over the bandpass
    x = sp.freq_to_x(b.nodes)
    scale = abs(b.response(lambda xx: 4. * sp.G(xx)))  # size of the terms of the tSZ spectrum
    assert b.integrate(sp.tsz_nonrel(x)) == pytest.approx(y_uk, abs=2e-4 * scale)


@pytest.mark.parametrize('preset', ['Planck', 'SPT-3G', 'ACT-DR6', 'SPT-SZ'])
def test_band_integration_relativistic(szpack, preset):
    from pymsz.SZpack_models import _szpack_options, szpack_dn
    opts = _szpack_options('combo')
    Te = 20.

    def g(x):  # relativistic tSZ spectrum per unit y
        # SZpack covers 0.1 <= x <= 50 (GSL aborts outside, and for combo already from x ~ 49.8 at
        # Te >~ 5 keV); the measured Planck bandpasses extend far beyond with negligible weight: use the
        # non-relativistic spectrum there
        out = sp.tsz_nonrel(x)
        ok = np.nonzero((x >= 0.1) & (x <= 45.))[0]
        o = ok[np.argsort(x[ok])]
        out[o] = szpack_dn(szpack, [float(v) for v in x[o]], Te, 0., 1., 'combo', opts) / (Te / sp.ME_KEV)
        return out

    for b in Telescope.preset(preset).bands:
        full = b.response(g)
        nodes = b.integrate(g(sp.freq_to_x(b.nodes)))
        scale = max(abs(full), abs(b.response(lambda xx: 4. * sp.G(xx))))
        assert nodes == pytest.approx(full, abs=2e-4 * scale), b.name


def test_monochromatic_band():
    b = Band(150.)
    x = sp.freq_to_x(150.)
    assert np.allclose(b.nodes, [150.])
    assert b.response(sp.tsz_nonrel) == pytest.approx(sp.TCMB * 1e6 * sp.tsz_nonrel(x) / sp.G(x), rel=1e-12)


def test_noise_units_agree():
    # 1 uK-deg = 60 uK-arcmin; the MJy/sr noise of a band is its uK_CMB noise times the conversion
    assert Band(150., noise=1., noise_unit='uK-deg').noise_uk_arcmin() == pytest.approx(60.)
    b = Band(545., frac_bandwidth=0.3)
    mjy = 10. * 1e-6 * b.cmb_to('MJy/sr')  # 10 uK-arcmin in MJy/sr-arcmin
    assert Band(545., frac_bandwidth=0.3, noise=mjy, noise_unit='MJy/sr-arcmin').noise_uk_arcmin() == pytest.approx(10.)


# ------------------------------------------------------------------------------------------------
# mock observations of synthetic skies
# ------------------------------------------------------------------------------------------------
SKY_FREQS = np.geomspace(20., 1000., 6)


def blob_map(n=128, sigma_pix=3., amp=-1e-4):
    """A compact Gaussian source (Delta n), centred on the image."""
    yy, xx = np.mgrid[:n, :n] - (n - 1) / 2.
    return amp * np.exp(-(xx**2 + yy**2) / (2. * sigma_pix**2))


def blob_sky(n=128, ar=15.):
    """The blob with the same Delta n at all frequencies."""
    return SkyMaps(np.repeat(blob_map(n)[None], SKY_FREQS.size, axis=0), SKY_FREQS, unit='Dn', pixel_arcsec=ar)


def empty_sky(n=256, ar=30.):
    return SkyMaps(np.zeros((SKY_FREQS.size, n, n)), SKY_FREQS, unit='Dn', pixel_arcsec=ar)


def flat_cl(D=1000., lmax=12000):
    ell = np.arange(lmax + 1.)
    cl = np.zeros_like(ell)
    cl[2:] = 2. * np.pi * D / (ell[2:] * (ell[2:] + 1.))
    return ell, cl


@pytest.mark.parametrize('pixel', [None, 1.])
def test_white_noise_level(pixel):
    tel = Telescope([Band(150., noise=10.)], pixel=pixel)  # 10 uK-arcmin
    obs = tel.observe(empty_sky(), seed=1)
    p = 0.5 if pixel is None else pixel  # arcmin
    assert obs.pixel == pytest.approx(p)
    assert np.std(obs.noise[0]) == pytest.approx(10. / p, rel=0.03)
    assert np.array_equal(obs.maps, obs.noise)  # empty sky, no CMB


def test_beam_and_pixels_conserve_the_flux():
    sky = blob_sky()
    tel = Telescope([Band(150., fwhm=1.5)], pixel=0.5)
    obs = tel.observe(sky, noise=False, calibration=False)
    assert obs.maps.shape == (1, 64, 64)  # 128 x 15'' -> 64 x 0.5'
    flux_true = obs.sz_true[0].sum() * (15. / 60.)**2
    assert obs.sz[0].sum() * 0.5**2 == pytest.approx(flux_true, rel=1e-3)
    assert obs.sz[0].min() > 0.7 * obs.sz_true[0].min()  # the beam dilutes the peak
    # at the simulation resolution: the input Delta n as a temperature change, Delta T = Tcmb Delta n / G(x)
    assert np.allclose(obs.sz_true[0], blob_map() * sp.TCMB * 1e6 / sp.G(sp.freq_to_x(150.)), rtol=1e-9)


def test_seeds_and_shared_cmb():
    sky = blob_sky()
    cmb = flat_cl()
    a = Telescope([Band(150., fwhm=1.2, noise=5.)], name='A')
    b = Telescope([Band(150., fwhm=1.2, noise=20.)], name='B')
    oa = a.observe(sky, cmb=cmb, seed=1, cmb_seed=7)
    ob = b.observe(sky, cmb=cmb, seed=2, cmb_seed=7)
    assert oa.cmb_seed == 7
    assert np.array_equal(oa.cmb, ob.cmb)  # same CMB sky ...
    assert np.corrcoef(oa.noise.ravel(), ob.noise.ravel())[0, 1] < 0.05  # ... independent noise
    assert np.array_equal(oa.maps, a.observe(sky, cmb=cmb, seed=1, cmb_seed=7).maps)  # reproducible
    oc = a.observe(sky, cmb=cmb, seed=1, cmb_seed=8)
    assert np.array_equal(oc.noise, oa.noise) and not np.allclose(oc.cmb, oa.cmb)
    assert 20. < np.std(oa.cmb) < 300.  # uK, a sensible CMB realisation on a 32' patch


def test_calibration_errors():
    tel = Telescope([Band(150., calib_error=0.05)])
    gains = [tel.observe(blob_sky(), noise=False, seed=s).gains[0] for s in range(40)]
    assert 0.02 < np.std(gains) < 0.08
    assert tel.observe(blob_sky(), noise=False, calibration=False, seed=0).gains[0] == 1.


def test_observe_errors():
    with pytest.raises(ValueError):
        Telescope([Band(150., fwhm=1.)], pixel=60.).observe(blob_sky(), noise=False)  # pixel > image
    other = SkyMaps(np.zeros((SKY_FREQS.size, 128, 128)), SKY_FREQS, unit='Dn', pixel_arcsec=10.)
    with pytest.raises(ValueError):
        Telescope([Band(150.)]).observe([blob_sky(), other])  # different pixel sizes
    with pytest.raises(ValueError):
        Telescope([Band(150.), Band(150.)])  # band names must be unique
    with pytest.raises(ValueError):  # the images must cover the band frequencies
        Telescope([Band(150., frac_bandwidth=0.3)]).observe(
            SkyMaps(np.zeros((2, 8, 8)), [140., 160.], unit='Dn', pixel_arcsec=10.))


def test_write_fits(tmp_path):
    from astropy.io import fits
    obs = Telescope([Band(150., fwhm=1.2, noise=5.)], name='T').observe(blob_sky(), cmb=flat_cl(), seed=3,
                                                                        cmb_seed=4)
    f = str(tmp_path / 'obs.fits')
    obs.write_fits(f, header={'OBJECT': ('blob', 'test source')})
    with fits.open(f) as h:
        assert [x.name for x in h] == ['PRIMARY', 'SZ', 'CMB', 'NOISE', 'SZTRUE']
        hdr = h[0].header
        assert hdr['OBJECT'] == 'blob' and int(hdr['SEED']) == 3 and int(hdr['CMBSEED']) == 4
        assert np.allclose(h[0].data, obs.maps, rtol=1e-6, atol=1e-6 * np.abs(obs.maps).max())
        assert np.allclose(h['SZ'].data, obs.sz, rtol=1e-6, atol=1e-6 * np.abs(obs.sz).max())
    with pytest.raises(OSError):
        obs.write_fits(f)  # no overwrite by default
    obs.write_fits(f, components=False, overwrite=True)
    with fits.open(f) as h:
        assert len(h) == 1
