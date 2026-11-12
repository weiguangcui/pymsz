"""
Measured bandpasses (spectral responses) of CMB / SZ telescopes, shipped with pymsz in
pymsz/data/bandpasses (original files from the Planck Legacy Archive and NASA LAMBDA, see the
README.md there for sources, checksums and citations).

Every curve returned by load() is the detector response per unit sky specific intensity, R(nu):
the power on the detector is P = eta * int R(nu) I_nu dnu. This is the weight used by
pymsz.telescope.Band with aperture='flat'. Curves are normalised to a peak of 1 (calibration
removes the normalisation).

Conventions of the original files, checked against published numbers (README.md):

- Planck RIMO (HFI R3.00, LFI R3.31): transmission per unit I_nu (HFI in cm^-1). With the 2015
  RIMO, the K_CMB -> MJy/sr and K_CMB -> y factors of the PLA are reproduced to < 0.1%.
- SPT-3G (2019): response to a beam-filling flat-spectrum source, A Omega included (Sobrin et al.
  2022); the file centroids and widths reproduce their Table 6.
- SPT-SZ: per unit I_nu; the tSZ effective band centres of Bleem et al. 2022 (99.6, 155.5,
  220.0 GHz) are reproduced exactly.
- ACT DR6 .dat: the passbands of the DR6 likelihood = AdvACT_Passbands.h5 'mean_band' x nu^-2
  within the 'integration_bounds': per unit I_nu.
- ACT h5: response to a Rayleigh-Jeans source (FTS not divided by nu^2); load() multiplies it by
  nu^-2. Used for the bandpass uncertainties (variant=) and the PA7 low-frequency bands.

Examples
--------
>>> from pymsz import bandpasses
>>> bandpasses.available()
>>> bp = bandpasses.load('SPT-3G/f150')            # bp.nu (GHz), bp.response, bp.name, bp.reference
>>> pymsz.Band(150., fwhm=1.2, noise=4.4, bandpass='SPT-3G/f150')
>>> pymsz.Band(150., bandpass=bandpasses.load('ACT-DR6/pa5_f150', variant='upper'))
"""

import os
from collections import namedtuple
from functools import lru_cache
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'bandpasses')
GHZ_PER_INVCM = 29.9792458  # 1 cm^-1 in GHz

_trapezoid = getattr(np, 'trapezoid', None) or np.trapz  # numpy < 2

Bandpass = namedtuple('Bandpass', ['nu', 'response', 'name', 'reference'])
Bandpass.__doc__ = """Measured bandpass: nu (GHz), response per unit I_nu (peak 1), name, reference."""

_REF = {
    'Planck/LFI': 'Planck 2018 LFI RIMO R3.31 (Planck Legacy Archive); Zonca et al. 2009 (JINST 4, T12010)',
    'Planck/HFI': 'Planck 2018 HFI RIMO R3.00 (Planck Legacy Archive); Planck 2013 results IX (spectral response)',
    'ACT-DR6': 'ACT DR6.02 passbands (NASA LAMBDA, act_dr6.02_passbands); Louis et al. 2025 (arXiv:2503.14452)',
    'SPT-3G': 'SPT-3G 2019 bandpasses (NASA LAMBDA); Sobrin et al. 2022, ApJS 258, 42',
    'SPT-SZ': 'SPT-SZ bandpasses (NASA LAMBDA, Bleem et al. 2022, ApJS 258, 36, Sect. 2.1.4)',
}

# name -> how to read it. kinds: 'fits' (table columns), 'text' (2 columns), 'h5' (ACT), 'combine'
_REGISTRY = {}
for _c in ('030', '044', '070'):
    _REGISTRY['Planck/LFI' + _c] = dict(kind='fits', file='planck/LFI_RIMO_R3.31.fits', hdu='BANDPASS_' + _c,
                                        cols=('WAVENUMBER', 'TRANSMISSION'), scale=1., ref=_REF['Planck/LFI'])
for _c in ('100', '143', '217', '353', '545', '857'):
    _REGISTRY['Planck/HFI' + _c] = dict(kind='fits', file='planck/HFI_RIMO_R3.00.fits', hdu='BANDPASS_F' + _c,
                                        cols=('WAVENUMBER', 'TRANSMISSION'), scale=GHZ_PER_INVCM,
                                        ref=_REF['Planck/HFI'])
for _pa, _f in (('pa4', 'f150'), ('pa4', 'f220'), ('pa5', 'f090'), ('pa5', 'f150'), ('pa6', 'f090'), ('pa6', 'f150')):
    _REGISTRY['ACT-DR6/%s_%s' % (_pa, _f)] = dict(kind='text', file='act_dr6/passband_dr6_%s_%s.dat' % (_pa, _f),
                                                  h5key='%s_%s' % (_pa.upper(), _f), ref=_REF['ACT-DR6'])
for _f in ('f030', 'f040'):
    _REGISTRY['ACT-DR6/pa7_' + _f] = dict(kind='h5', h5key='PA7_' + _f, ref=_REF['ACT-DR6'])
# frequency coadds of the arrays (as in the DR6 coadded maps); approximate: equal weights per array
_REGISTRY['ACT-DR6/f090'] = dict(kind='combine', parts=('ACT-DR6/pa5_f090', 'ACT-DR6/pa6_f090'), ref=_REF['ACT-DR6'])
_REGISTRY['ACT-DR6/f150'] = dict(kind='combine', parts=('ACT-DR6/pa4_f150', 'ACT-DR6/pa5_f150', 'ACT-DR6/pa6_f150'),
                                 ref=_REF['ACT-DR6'])
_REGISTRY['ACT-DR6/f220'] = dict(kind='combine', parts=('ACT-DR6/pa4_f220',), ref=_REF['ACT-DR6'])
for _n, _f in (('f095', '95'), ('f150', '150'), ('f220', '220')):
    _REGISTRY['SPT-3G/' + _n] = dict(kind='text', file='spt3g/spt3g_bandpass_%sGHz_2019.txt' % _f, ref=_REF['SPT-3G'])
for _n, _f in (('f095', '90'), ('f150', '150'), ('f220', '220')):
    _REGISTRY['SPT-SZ/' + _n] = dict(kind='fits', file='spt_sz/SPTSZ_Bandpass_%sGHz_2009_through_2011_seasons.fits' % _f,
                                     hdu=1, cols=('NU', 'TRANS'), scale=1., ref=_REF['SPT-SZ'] + ', 2009-2011 seasons')
    _REGISTRY['SPT-SZ/%s_2008' % _n] = dict(kind='fits', file='spt_sz/SPTSZ_Bandpass_%sGHz_2008_season.fits' % _f,
                                            hdu=1, cols=('NU', 'TRANS'), scale=1., ref=_REF['SPT-SZ'] + ', 2008 season')
_KEYS = {k.lower(): k for k in _REGISTRY}


def available():
    """Names of the bandpasses shipped with pymsz."""
    return sorted(_REGISTRY)


def is_available(name):
    """True if `name` is the name of a bandpass shipped with pymsz (case insensitive)."""
    return isinstance(name, str) and name.lower() in _KEYS


def _path(f):
    return os.path.join(DATA_DIR, f)


def _read_h5(key, variant):
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is needed to read the ACT bandpass uncertainties / PA7 bands (pip install h5py)")
    with h5py.File(_path('act_dr6/AdvACT_Passbands.h5'), 'r') as g:
        nu = g[key + '/frequencies'][()]
        lo, hi = g[key + '/integration_bounds'][()]
        if variant in (None, 'mean'):
            band = g[key + '/mean_band'][()]
        elif variant in ('lower', 'upper'):
            band = g[key + '/%s_systematic' % variant][()]
        elif isinstance(variant, (int, np.integer)):
            band = g[key + '/statistical_sample'][()][int(variant)]
        else:
            raise ValueError("variant must be None, 'mean', 'lower', 'upper' or a sample index 0-99")
    keep = (nu >= lo) & (nu <= hi)
    nu = nu[keep]
    return nu, band[keep] * nu**-2.   # Rayleigh-Jeans response -> response per unit I_nu


def frequency_shift_uncertainty(name):
    """1-sigma uncertainty (GHz) of the band centre given with the ACT passbands (needs h5py)."""
    e = _REGISTRY[_KEYS[name.lower()]]
    if 'h5key' not in e:
        raise ValueError("No frequency-shift uncertainty is shipped for %s" % name)
    import h5py
    with h5py.File(_path('act_dr6/AdvACT_Passbands.h5'), 'r') as g:
        return float(g[e['h5key'] + '/frequency_shift_uncertainty'][()])


@lru_cache(maxsize=64)
def _load(name, variant):
    e = _REGISTRY[name]
    kind = e['kind']
    if variant is not None and 'h5key' not in e and kind != 'combine':
        raise ValueError("variant is only available for the ACT array passbands, not %s" % name)
    if kind == 'fits':
        from astropy.io import fits
        with fits.open(_path(e['file'])) as h:
            d = h[e['hdu']].data
            nu = np.asarray(d[e['cols'][0]], dtype=np.float64).ravel() * e['scale']
            tr = np.asarray(d[e['cols'][1]], dtype=np.float64).ravel()
    elif kind == 'text' and variant is None:
        d = np.loadtxt(_path(e['file']))
        nu, tr = d[:, 0], d[:, 1]
    elif kind in ('text', 'h5'):
        nu, tr = _read_h5(e['h5key'], variant)
    else:  # 'combine': coadd of CMB-calibrated maps
        parts = [_load(p, variant) for p in e['parts']]
        return combine([(p[0], p[1]) for p in parts])
    keep = nu > 0
    nu, tr = nu[keep], tr[keep]
    o = np.argsort(nu, kind='stable')
    nu, tr = nu[o], tr[o]
    return nu, tr / np.max(tr)


def load(name, variant=None):
    r"""A measured bandpass shipped with pymsz.

    Parameters
    ----------
    name : str
        Bandpass name, see :func:`available` (case insensitive).
    variant : None, str or int, optional
        ACT only, needs h5py. None (default): the DR6 likelihood passband. 'mean': the same, from the
        h5 file. 'lower' or 'upper': the systematic bounds. 0-99: statistical resamples.

    Returns
    -------
    Bandpass
        ``(nu, response, name, reference)``: nu in GHz; the response per unit sky intensity I_nu
        (use ``Band(aperture='flat')``, the default), with a peak of 1.
    """
    key = _KEYS.get(str(name).lower())
    if key is None:
        raise ValueError("Unknown bandpass %s. Available: %s" % (name, ', '.join(available())))
    nu, tr = _load(key, variant)
    ref = _REGISTRY[key]['ref']
    if _REGISTRY[key]['kind'] == 'combine':
        ref += '; coadd of %s with equal weights of the CMB-calibrated arrays (approximate)' % \
            ', '.join(p.split('/')[1] for p in _REGISTRY[key]['parts'])
    if variant is not None:
        ref += '; variant %s' % (variant,)
    return Bandpass(nu.copy(), tr.copy(), key, ref)


def combine(bands, weights=None, Tcmb=None):
    r"""Effective bandpass of a coadd of CMB-calibrated maps, e.g. several detector arrays.

    :math:`w_\mathrm{eff}(\nu) = \sum_a W_a R_a(\nu) / \int R_a\, \partial B_\nu/\partial T\, d\nu`,
    for responses :math:`R_a` per unit :math:`I_\nu`.

    Parameters
    ----------
    bands : list
        ``(nu, response)`` pairs or `Bandpass` objects.
    weights : array_like, optional
        Coadd weights W_a (e.g. inverse noise variance). Default: equal.
    Tcmb : float, optional
        CMB temperature in K. Default 2.7255.

    Returns
    -------
    nu, response : ndarray
        On the union of the frequency grids, with a peak of 1.
    """
    from pymsz import sz_spectra
    Tcmb = sz_spectra.TCMB if Tcmb is None else Tcmb
    bands = [(np.asarray(b[0], dtype=np.float64), np.asarray(b[1], dtype=np.float64)) for b in bands]
    W = np.ones(len(bands)) if weights is None else np.asarray(weights, dtype=np.float64)
    nu = np.unique(np.concatenate([b[0] for b in bands]))
    out = np.zeros_like(nu)
    for (n, r), w in zip(bands, W):
        n0 = _trapezoid(r * sz_spectra.dB_dT(n, Tcmb), n)
        out += w * np.interp(nu, n, r, left=0., right=0.) / n0
    return nu, out / out.max()
