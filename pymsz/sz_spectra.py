"""
Frequency dependence of the SZ effect and the unit conversions shared by all pymsz models
(TT_model, TK_model, SZpack_model) and by the mock telescope (pymsz.telescope).

All SZ signals are carried internally as the change of the photon occupation number Delta n(x),
x = h nu / (k T_CMB). Delta n is redshift independent, so nu are observed frequencies.

Non-relativistic limits (Sunyaev & Zeldovich 1972, 1980)::

    thermal   : Delta n = y        * G(x) * (x coth(x/2) - 4)
    kinematic : Delta n = tau beta * G(x),           beta = v_los/c > 0 towards the observer

with G(x) = x e^x / (e^x - 1)^2. Relativistic corrections are computed with SZpack, see SZpack_models.
"""

import warnings
import numpy as np

# physical constants (CODATA 2018)
KB = 1.380649e-23               # J / K
H = 6.62607015e-34              # J s
C = 299792458.0                 # m / s
KB_KEV = 8.617333262e-8         # Boltzmann constant in keV / K
ME_KEV = 510.99895000           # electron rest mass energy in keV
C_KMS = C / 1.0e3               # speed of light in km / s
SIGMA_T = 6.6524587321e-25      # Thomson cross section in cm^2
MP = 1.67262192369e-24          # proton mass in g
KPC = 3.0856775809623245e+21    # kpc in cm
ARCSEC = np.pi / 180. / 3600.   # arcsec in rad
ARCMIN = np.pi / 180. / 60.     # arcmin in rad
TCMB = 2.7255                   # CMB temperature today in K (Fixsen 2009)
I0_PER_T3 = 2.0 * KB**3 / (H * C)**2 / 1.0e-20   # 2 (k T)^3/(h c)^2 in MJy/sr for T = 1 K

# units of the SZ images produced by the models
UNITS = ('MJy/sr', 'Jy/pixel', 'K', 'uK', 'Dn')


def freq_to_x(freqs, Tcmb=TCMB):
    """Dimensionless frequency x = h nu / (k Tcmb) for frequencies in GHz."""
    return H / KB * np.asarray(freqs, dtype=np.float64) * 1.0e9 / Tcmb


def x_to_freq(x, Tcmb=TCMB):
    """Frequency in GHz for dimensionless frequencies x."""
    return np.asarray(x, dtype=np.float64) * KB * Tcmb / H / 1.0e9


def I0(Tcmb=TCMB):
    """Intensity unit I0 = 2 (k Tcmb)^3 / (h c)^2 in MJy/sr (~270.07 MJy/sr for 2.7255 K)."""
    return I0_PER_T3 * Tcmb**3


def G(x):
    """G(x) = x e^x / (e^x - 1)^2 = -x dn_Planck/dx: Delta n per unit Delta T/T_CMB."""
    x = np.asarray(x, dtype=np.float64)
    em1 = np.expm1(x)
    return x * (em1 + 1.0) / em1**2


def tsz_nonrel(x):
    """Non-relativistic thermal SZ spectrum: Delta n per unit Compton y, G(x) (x coth(x/2) - 4).
    Negative below x = 3.83 (217.5 GHz for Tcmb = 2.7255 K), positive above."""
    x = np.asarray(x, dtype=np.float64)
    return G(x) * (x / np.tanh(x / 2.) - 4.)


def ksz_nonrel(x):
    """Non-relativistic kinematic SZ spectrum: Delta n per unit tau * v_los/c (v_los > 0 towards
    the observer, i.e. Delta T/T_CMB = tau v_los / c). Same shape as a CMB temperature change."""
    return G(x)


def dB_dT(freqs, Tcmb=TCMB):
    """Derivative of the Planck spectrum with respect to temperature at Tcmb, in MJy/sr/K."""
    x = freq_to_x(freqs, Tcmb)
    return I0(Tcmb) / Tcmb * x**3 * G(x)


def check_unit(unit):
    """Return `unit` if it is one of UNITS, else raise a ValueError."""
    if unit not in UNITS:
        raise ValueError("unit must be one of %s, not %s" % (UNITS, unit))
    return unit


def dn_to_unit(dn, freqs, unit, Tcmb=TCMB, pixel_arcsec=None):
    r"""Convert Delta n images (first axis = frequency) into `unit`.

    Parameters
    ----------
    dn : array_like
        Delta n, first axis = frequency.
    freqs : array_like
        Frequencies in GHz.
    unit : str
        'Dn': Delta n (returned unchanged). 'MJy/sr': Delta I_nu = I0 x^3 Delta n. 'Jy/pixel':
        Delta I_nu times the pixel solid angle. 'K', 'uK': linearised CMB thermodynamic temperature,
        Delta T = Tcmb Delta n / G(x).
    Tcmb : float, optional
        CMB temperature in K. Default 2.7255.
    pixel_arcsec : float, optional
        Pixel size in arcsec, needed for 'Jy/pixel'.

    Returns
    -------
    ndarray
    """
    check_unit(unit)
    dn = np.asarray(dn, dtype=np.float64)
    if unit == 'Dn':
        return dn
    x = freq_to_x(freqs, Tcmb).reshape((-1,) + (1,) * (dn.ndim - 1))
    if unit in ('K', 'uK'):
        return Tcmb * dn / G(x) * (1.0e6 if unit == 'uK' else 1.)
    dI = I0(Tcmb) * x**3 * dn  # MJy/sr
    if unit == 'MJy/sr':
        return dI
    if pixel_arcsec is None:
        raise ValueError("unit='Jy/pixel' needs the pixel size (pixel_arcsec)")
    return dI * 1.0e6 * (pixel_arcsec * ARCSEC)**2


def unit_to_dn(data, freqs, unit, Tcmb=TCMB, pixel_arcsec=None):
    """Inverse of dn_to_unit."""
    check_unit(unit)
    data = np.asarray(data, dtype=np.float64)
    if unit == 'Dn':
        return data
    one = dn_to_unit(np.ones((np.size(freqs),) + (1,) * (data.ndim - 1)), np.atleast_1d(freqs), unit,
                     Tcmb, pixel_arcsec)
    return data / one


def write_sz_cube(model, fname, aux=(), extra=(), comments="None", overwrite=False,
                  origin='Software: PymSZ'):
    r"""Write ``model.szdata`` as the primary HDU of a FITS file, plus image extensions.

    The primary HDU has shape [nfreq, npixel, npixel], in ``model.unit``.

    Parameters
    ----------
    model : TT_model, TK_model or SZpack_model
        An object with the attributes szdata, freqs, unit, Tcmb, npl, ar, sp, cc, rr, red and pxs.
    fname : str
        File name; '.fits' is added if missing.
    aux : list, optional
        ``(EXTNAME, 2D array, pixval, comment)`` tuples, written as image extensions.
    extra : list, optional
        ``(KEY, value, comment)`` tuples added to the primary header.
    comments : str or list of str, optional
        Comments for the primary header.
    overwrite : bool, optional
        Replace an existing file. Default False.
    origin : str, optional
        Value of the ORIGIN keyword.
    """
    import astropy.io.fits as pf
    from astropy.time import Time
    from pymsz import version

    if len(np.shape(model.szdata)) != 3:
        raise ValueError("No SZ images to write. Compute them first (e.g. model.to_sz(freqs)).")
    if fname[-5:] != ".fits":
        fname = fname + ".fits"

    def header(hdr, pixval, pixcomment):
        hdr["RADECSYS"] = ('ICRS    ', "International Celestial Ref. System")
        hdr["CTYPE1"] = ('RA---TAN', "Coordinate type")
        hdr["CTYPE2"] = ('DEC--TAN', "Coordinate type")
        hdr["CUNIT1"] = ('deg     ', 'Units')
        hdr["CUNIT2"] = ('deg     ', 'Units')
        hdr["CRPIX1"] = (float(model.npl / 2.0), 'X of reference pixel')
        hdr["CRPIX2"] = (float(model.npl / 2.0), 'Y of reference pixel')
        hdr["CRVAL1"] = (float(model.sp[0]), 'RA of reference pixel (deg)')
        hdr["CRVAL2"] = (float(model.sp[1]), 'Dec of reference pixel (deg)')
        hdr["CD1_1"] = (-float(model.ar / 3600.), 'RA deg per column pixel')
        hdr["CD1_2"] = (float(0), 'RA deg per row pixel')
        hdr["CD2_1"] = (float(0), 'Dec deg per column pixel')
        hdr["CD2_2"] = (float(model.ar / 3600.), 'Dec deg per row pixel')
        hdr["PIXVAL"] = (pixval, pixcomment)

    hdu = pf.PrimaryHDU(np.asarray(model.szdata, dtype=np.float32))
    hdr = hdu.header
    header(hdr, model.unit, 'SZ signal, frequencies in FREQn keywords (GHz)')
    hdr["CTYPE3"] = ('FREQ-TAB', 'frequency index, see FREQn keywords')
    for k, f in enumerate(model.freqs):
        hdr["FREQ%d" % (k + 1)] = (float(f), 'frequency (GHz) of plane %d' % (k + 1))
    for key, val, com in extra:
        hdr[key] = (val, com)
    hdr["TCMB"] = (float(model.Tcmb), 'CMB temperature (K) used for x and units')
    hdr["RCVAL1"] = (float(model.cc[0]), 'Real center X of the data')
    hdr["RCVAL2"] = (float(model.cc[1]), 'Real center Y of the data')
    hdr["RCVAL3"] = (float(model.cc[2]), 'Real center Z of the data')
    hdr["UNITS"] = ("kpc", 'Units for the RCVAL and PSIZE')
    hdr["ORAD"] = (float(model.rr), 'Rcut in physical for the image.')
    hdr["REDSHIFT"] = (float(model.red), 'The redshift of the object being put to')
    hdr["PSIZE"] = (float(model.pxs), 'The pixel size in physical at simulation time')
    hdr["AGLRES"] = (float(model.ar), '\'observation\' angular resolution in arcsec')
    hdr["ORIGIN"] = (origin, 'https://github.com/weiguangcui/pymsz.git')
    hdr["VERSION"] = (version.version, 'Version of the software')
    hdr["DATE-OBS"] = Time.now().tt.isot
    if isinstance(comments, (list, tuple)):
        for c in comments:
            hdr["COMMENT"] = c
    elif isinstance(comments, str):
        hdr["COMMENT"] = comments
    else:
        raise ValueError("Do not accept this comments type! Please use str or list")

    hdus = [hdu]
    for name, data, pv, pc in aux:
        h = pf.ImageHDU(np.asarray(data, dtype=np.float32), name=name)
        header(h.header, pv, pc)
        hdus.append(h)
    pf.HDUList(hdus).writeto(fname, overwrite=overwrite)


def interpolate_dn(dn, freqs, new_freqs, Tcmb=TCMB, what='maps'):
    r"""Interpolate Delta n images given at `freqs` (first axis) to `new_freqs`.

    The interpolation is a cubic spline in x (linear for 2-3 frequencies). Frequencies that are
    already present are copied exactly. Extrapolation is refused: the SZ spectrum changes quickly, so
    the requested frequencies must lie inside the computed range."""
    freqs = np.asarray(freqs, dtype=np.float64)
    new_freqs = np.atleast_1d(np.asarray(new_freqs, dtype=np.float64))
    out = np.empty((new_freqs.size,) + np.shape(dn)[1:])
    match = np.full(new_freqs.size, -1)
    for i, f in enumerate(new_freqs):
        k = np.nonzero(np.isclose(freqs, f, rtol=1e-7, atol=0.))[0]
        if k.size:
            match[i] = k[0]
            out[i] = dn[k[0]]
    todo = match < 0
    if not todo.any():
        return out
    uf, ui = np.unique(freqs, return_index=True)
    lo, hi = uf[0], uf[-1]
    if uf.size < 2 or new_freqs[todo].min() < lo * (1 - 1e-9) or new_freqs[todo].max() > hi * (1 + 1e-9):
        raise ValueError("The %s only exist at %s GHz and cannot be extrapolated to %s GHz. Compute them "
                         "at the needed frequencies (for a Telescope: freqs=telescope.frequencies())."
                         % (what, np.round(uf, 3).tolist(), np.round(new_freqs[todo], 3).tolist()))
    if uf.size < 6:
        warnings.warn("Interpolating the SZ spectrum between only %d frequencies; compute the %s at the "
                      "telescope frequencies (telescope.frequencies()) for accurate band averages."
                      % (uf.size, what))
    xs = freq_to_x(uf, Tcmb)
    xn = freq_to_x(new_freqs[todo], Tcmb)
    d = np.asarray(dn)[ui]
    if uf.size < 4:
        w = np.empty((xn.size, uf.size))
        eye = np.eye(uf.size)
        for j in range(uf.size):
            w[:, j] = np.interp(xn, xs, eye[j])
        out[todo] = np.tensordot(w, d, axes=(1, 0))
    else:
        from scipy.interpolate import CubicSpline
        out[todo] = CubicSpline(xs, d, axis=0)(xn)
    return out
