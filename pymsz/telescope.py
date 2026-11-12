"""
Mock (sub-)millimetre telescope observations of simulated SZ images.

The SZ models of pymsz (TT_model + TK_model after to_sz, or SZpack_model) give the SZ signal at any
frequency at the resolution of the simulation. This module turns them into the maps a telescope would
produce in each of its frequency bands:

    1. bandpass integration of the SZ spectrum (exact for TT/TK, see Band) and calibration of every
       band to CMB temperature (or MJy/sr, IRAS convention);
    2. optional sky components: a Gaussian realisation of the primary CMB, user foreground maps;
    3. convolution with the beam (Gaussian FWHM or a measured B_ell) and the pointing jitter;
    4. the transfer function of the map making (e.g. scan-direction high-pass filtering);
    5. pixelisation onto the telescope map pixels (area-weighted, i.e. including the pixel window);
    6. calibration (gain) errors per band;
    7. instrument + atmospheric noise: white noise (uK-arcmin) plus a 1/f "knee" component,
       N_ell = N_white [1 + (ell / ell_knee)^alpha_knee], optionally correlated between bands.

All operations use the flat-sky approximation on the (small) simulated patch.

Example
-------
>>> tt = pymsz.TT_model(simd, npixel=512, AR=5, redshift=0.1)
>>> tk = pymsz.TK_model(simd, npixel=512, AR=5, redshift=0.1)
>>> tel = pymsz.Telescope.preset('SPT-3G')
>>> obs = tel.observe([tt, tk], cmb='camb_lensedCls.dat', seed=1)   # tSZ + kSZ + CMB + noise
>>> obs.maps.shape        # (nband, ny, nx) in uK_CMB on 0.25 arcmin pixels
>>> obs.write_fits('SPT3G_mock.fits')

or with SZpack (relativistic tSZ + kSZ), computed at the frequencies needed for the band integration:

>>> mm = pymsz.SZpack_model(simd, tel.frequencies(), npixel=512, AR=5, redshift=0.1)
>>> obs = tel.observe(mm)
"""

import os
import numpy as np
from pymsz import sz_spectra, bandpasses
from pymsz.sz_spectra import TCMB, ARCMIN, ARCSEC, freq_to_x

OBS_UNITS = ('uK_CMB', 'K_CMB', 'MJy/sr')
_UNIT_ALIASES = {'uk_cmb': 'uK_CMB', 'uk': 'uK_CMB', 'k_cmb': 'K_CMB', 'k': 'K_CMB', 'mjy/sr': 'MJy/sr'}
_NOISE_UNITS = ('uK-arcmin', 'uK-deg', 'MJy/sr-arcmin', 'kJy/sr-deg')
_APERTURES = {'flat': 'flat', 'single-mode': 'single-mode', 'rj': 'single-mode'}


def _obs_unit(unit):
    u = _UNIT_ALIASES.get(str(unit).lower())
    if u is None:
        raise ValueError("unit must be one of %s, not %s" % (OBS_UNITS, unit))
    return u


def gaussian_beam(ell, fwhm_arcmin):
    """Harmonic transform B_ell of a Gaussian beam with the given FWHM in arcmin."""
    sigma = fwhm_arcmin * ARCMIN / np.sqrt(8. * np.log(2.))
    ell = np.asarray(ell, dtype=np.float64)
    return np.exp(-0.5 * ell * (ell + 1.) * sigma**2)


def _spline_basis(nodes, nu):
    """Matrix L (len(nu), len(nodes)) of the interpolation f(nu) = sum_j L_j(nu) f(nodes_j): natural
    cubic spline (linear for < 4 nodes) inside the nodes, linear extrapolation outside."""
    n = nodes.size
    eye = np.eye(n)
    if n >= 4:
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(nodes, eye, axis=0, bc_type='natural')
        L = cs(np.clip(nu, nodes[0], nodes[-1]))
        d1 = cs(nodes[[0, -1]], 1)   # end slopes; natural ends -> linear continuation is C2
        lo, hi = nu < nodes[0], nu > nodes[-1]
        L[lo] += (nu[lo] - nodes[0])[:, None] * d1[0]
        L[hi] += (nu[hi] - nodes[-1])[:, None] * d1[1]
        return L
    from scipy.interpolate import interp1d
    return interp1d(nodes, eye, axis=0, kind='linear', fill_value='extrapolate')(nu)


# ----------------------------------------------------------------------------------------------
# frequency bands
# ----------------------------------------------------------------------------------------------
class Band(object):
    r"""One frequency band (detector channel) of a telescope.

    Parameters
    ----------
    freq : float
        Nominal band centre in GHz. It is also the reference frequency of the MJy/sr maps (IRAS
        convention, nu I_nu = constant).
    fwhm : float, optional
        FWHM of the Gaussian beam in arcmin. Default 0 (no beam). Ignored if `beam` is given.
    noise : float, optional
        White-noise level in `noise_unit`. Default 0 (no noise).
    name : str, optional
        Band name. Default ``'f%03d' % freq``.
    bandwidth : float, optional
        Width in GHz of a top-hat bandpass centred on `freq`. Default None.
    frac_bandwidth : float, optional
        Width of a top-hat bandpass as a fraction of `freq`. Default None. Without `bandwidth`,
        `frac_bandwidth` and `bandpass` the band is monochromatic at `freq`.
    bandpass : str, Bandpass or tuple, optional
        Measured bandpass; it overrides the top-hat. One of

        - the name of a bandpass shipped with pymsz, e.g. 'Planck/HFI143', 'ACT-DR6/pa5_f090',
          'SPT-3G/f150', 'SPT-SZ/f095' (see :func:`pymsz.bandpasses.available`);
        - a :class:`pymsz.bandpasses.Bandpass`, e.g. ``bandpasses.load('ACT-DR6/pa5_f150',
          variant='upper')``;
        - ``(nu_GHz, response)`` arrays, or the name of a text file with these two columns.
    noise_unit : str, optional
        'uK-arcmin' (CMB temperature, default), 'uK-deg', 'MJy/sr-arcmin' or 'kJy/sr-deg'.
    ell_knee : float, optional
        Knee of the 1/f (atmospheric) noise,
        :math:`N_\ell = N_\mathrm{white} [1 + (\ell/\ell_\mathrm{knee})^{\alpha_\mathrm{knee}}]`.
        Default None: no 1/f noise.
    alpha_knee : float, optional
        Slope of the 1/f noise. Default -3.5.
    calib_error : float, optional
        Fractional (1 sigma) calibration uncertainty. A random gain 1 + N(0, calib_error) is applied
        to the sky signal in every mock observation. Default 0.
    beam : tuple or callable, optional
        Measured beam: ``(ell, B_ell)`` arrays or a function B(ell). Default None: the Gaussian
        `fwhm`.
    nnodes : int, optional
        Number of frequencies (nodes) across the band at which the SZ images are computed. For the
        band integration, the temperature spectrum Delta n / G(x) (constant for the kSZ and the CMB,
        nearly linear for the tSZ) is interpolated between them with a natural cubic spline, and
        extrapolated linearly into the far tails of the bandpass. Default 9: errors < 2e-4 of the
        band-averaged signal for every shipped bandpass, even for 20 keV relativistic tSZ (13 nodes
        for the very wide Planck HFI bands: < 4e-5).
    aperture : {'flat', 'single-mode', 'RJ'}, optional
        How the bandpass curve relates to the response to a sky intensity I_nu. 'flat' (default):
        the curve is the response per unit I_nu; this is the convention of all bandpasses shipped
        with pymsz. 'single-mode' (alias 'RJ'): the curve is the response to a Rayleigh-Jeans (nu^2)
        source, or the transmission of a single-moded detector (A Omega = lambda^2); it is weighted
        by nu^-2.
    shift : float, optional
        Shift in GHz of the whole bandpass (nu -> nu + shift), e.g. to propagate the bandpass
        uncertainties. Default 0.
    node_cut : float, optional
        The nodes span the frequencies where the weight w(nu) exceeds `node_cut` times its maximum.
        The bandpass beyond is still integrated (with the extrapolated spectrum). Default 1e-4.

    Attributes
    ----------
    nodes : ndarray
        The node frequencies (GHz).
    bandpass : tuple
        ``(nu, response)`` used, after the `shift`.

    Notes
    -----
    The band-averaged signal of a sky intensity :math:`\Delta I_\nu` is calibrated as

    .. math::

        \Delta T = \frac{\int w(\nu)\,\Delta I_\nu\,d\nu}{\int w(\nu)\,\partial B_\nu/\partial T\,d\nu}
        \quad\text{(CMB temperature)}, \qquad
        \Delta I = \frac{\int w(\nu)\,\Delta I_\nu\,d\nu}{\int w(\nu)\,(\nu_0/\nu)\,d\nu}
        \quad\text{(MJy/sr, IRAS)},

    with :math:`w(\nu)` the bandpass (times :math:`\nu^{-2}` for 'single-mode') and
    :math:`\nu_0` = `freq`.
    """

    def __init__(self, freq, fwhm=0., noise=0., name=None, bandwidth=None, frac_bandwidth=None,
                 bandpass=None, noise_unit='uK-arcmin', ell_knee=None, alpha_knee=-3.5, calib_error=0.,
                 beam=None, nnodes=9, aperture='flat', shift=0., node_cut=1.0e-4):
        self.freq = float(freq)
        self.fwhm = float(fwhm)
        self.noise = float(noise)
        self.name = name if name is not None else 'f%03d' % int(round(self.freq))
        if noise_unit not in _NOISE_UNITS:
            raise ValueError("noise_unit must be one of %s" % (_NOISE_UNITS,))
        self.noise_unit = noise_unit
        self.ell_knee = ell_knee
        self.alpha_knee = float(alpha_knee)
        self.calib_error = float(calib_error)
        self.beam = beam
        ap = _APERTURES.get(str(aperture).lower())
        if ap is None:
            raise ValueError("aperture must be 'flat' or 'single-mode' ('RJ')")
        self.aperture = ap
        self.shift = float(shift)

        self.bandpass_name = None
        if bandpass is not None:
            if bandpasses.is_available(bandpass):
                bandpass = bandpasses.load(bandpass)
            if isinstance(bandpass, bandpasses.Bandpass):
                if ap != 'flat':
                    raise ValueError("the bandpasses of pymsz.bandpasses are responses per unit intensity: "
                                     "use aperture='flat'")
                nu, tr = bandpass.nu, bandpass.response
                self.bandpass_name = bandpass.name
            elif isinstance(bandpass, str):
                d = np.loadtxt(bandpass)
                nu, tr = d[:, 0], d[:, 1]
                self.bandpass_name = os.path.basename(bandpass)
            else:
                nu, tr = (np.asarray(a, dtype=np.float64) for a in bandpass)
                self.bandpass_name = 'tabulated'
            o = np.argsort(nu, kind='stable')
            nu, tr = np.asarray(nu, dtype=np.float64)[o], np.clip(np.asarray(tr, dtype=np.float64)[o], 0., None)
            nz = np.nonzero(tr > 0)[0]
            if nz.size < 2:
                raise ValueError("bandpass of band %s has fewer than 2 points with transmission > 0" % self.name)
            nu, tr = nu[max(nz[0] - 1, 0):nz[-1] + 2], tr[max(nz[0] - 1, 0):nz[-1] + 2]
            self.kind = 'tabulated'
        else:
            if bandwidth is not None and frac_bandwidth is not None:
                raise ValueError("give bandwidth or frac_bandwidth, not both")
            bw = bandwidth if bandwidth is not None else \
                (frac_bandwidth * self.freq if frac_bandwidth is not None else 0.)
            if bw > 0:
                if bw >= 2 * self.freq:
                    raise ValueError("bandwidth of band %s is too large" % self.name)
                nu = np.linspace(self.freq - bw / 2., self.freq + bw / 2., 257)
                tr = np.ones_like(nu)
                self.kind = 'tophat'
            else:
                nu, tr = np.array([self.freq]), np.array([1.])
                self.kind = 'delta'
        nu = nu + self.shift
        if nu[0] <= 0:
            raise ValueError("band %s extends to nu <= 0 GHz" % self.name)
        self.bandpass = (nu, tr)

        # quadrature weights on the fine grid (trapezoidal rule), normalised to unit sum
        w = tr * ((self.freq / nu)**2 if ap == 'single-mode' else 1.)
        if nu.size > 1:
            dnu = np.zeros_like(nu)
            dnu[1:] += 0.5 * np.diff(nu)
            dnu[:-1] += 0.5 * np.diff(nu)
            sel = w >= node_cut * w.max()
            support = (nu[sel].min(), nu[sel].max())
            w = w * dnu
        self._nu, self._w = nu, w / w.sum()

        # interpolation nodes: the sky is needed only there
        if self.kind == 'delta':
            self.nodes = nu.copy()
            self._L = np.ones((1, 1))
        else:
            self.nodes = np.linspace(support[0], support[1], max(int(nnodes), 2))
            self._L = _spline_basis(self.nodes, self._nu)   # (nfine, nnodes)

    # --- calibration -------------------------------------------------------------------------
    def _norm(self, unit, Tcmb):
        """Band average of the calibrator spectrum: MJy/sr per K_CMB, or the IRAS nu^-1 spectrum."""
        if unit in ('uK_CMB', 'K_CMB'):
            return np.sum(self._w * sz_spectra.dB_dT(self._nu, Tcmb))
        return np.sum(self._w * self.freq / self._nu)

    def _scale(self, unit):
        return 1.0e6 if unit == 'uK_CMB' else 1.

    def weights(self, unit='uK_CMB', Tcmb=TCMB):
        """Coefficients c_j such that the band-averaged signal in `unit` is sum_j c_j Delta n(nodes_j)."""
        unit = _obs_unit(unit)
        x = freq_to_x(self._nu, Tcmb)
        # Delta n(nu) = G(x) sum_j L_j(nu) [Delta n_j / G(x_j)]: interpolation of Delta n / G
        c = (self._w * sz_spectra.I0(Tcmb) * x**3 * sz_spectra.G(x)) @ self._L
        c = c / sz_spectra.G(freq_to_x(self.nodes, Tcmb))
        return c / self._norm(unit, Tcmb) * self._scale(unit)

    def integrate(self, dn_nodes, unit='uK_CMB', Tcmb=TCMB):
        """Band-averaged map in `unit` from the Delta n maps at self.nodes (first axis)."""
        return np.tensordot(self.weights(unit, Tcmb), np.asarray(dn_nodes), axes=(0, 0))

    def response(self, spectrum, unit='uK_CMB', Tcmb=TCMB):
        r"""Band-averaged signal in `unit` of a spectrum Delta n(x) given as a function, integrated on
        the full bandpass grid. E.g. response(sz_spectra.tsz_nonrel) is the band's (non-relativistic)
        tSZ conversion from y to uK_CMB, and response(sz_spectra.ksz_nonrel, 'K_CMB') = Tcmb."""
        unit = _obs_unit(unit)
        x = freq_to_x(self._nu, Tcmb)
        num = np.sum(self._w * sz_spectra.I0(Tcmb) * x**3 * spectrum(x))
        return num / self._norm(unit, Tcmb) * self._scale(unit)

    def cmb_to(self, unit, Tcmb=TCMB):
        """Value in `unit` of a 1 K_CMB temperature fluctuation in this band."""
        unit = _obs_unit(unit)
        return self._norm('K_CMB', Tcmb) / self._norm(unit, Tcmb) * self._scale(unit)

    def noise_uk_arcmin(self, Tcmb=TCMB):
        """White noise level in uK_CMB-arcmin."""
        v = self.noise
        if self.noise_unit == 'uK-arcmin':
            return v
        if self.noise_unit == 'uK-deg':
            return v * 60.
        if self.noise_unit == 'kJy/sr-deg':
            v = v * 1.0e-3 * 60.  # -> MJy/sr-arcmin
        return v / self.cmb_to('MJy/sr', Tcmb) * 1.0e6

    def noise_spectrum(self, ell, unit='uK_CMB', Tcmb=TCMB, part='total'):
        r"""Noise power spectrum N_ell in unit^2 sr: part='white', 'red' or 'total'."""
        ell = np.asarray(ell, dtype=np.float64)
        nw = (self.noise_uk_arcmin(Tcmb) * 1.0e-6 * self.cmb_to(unit, Tcmb) * ARCMIN)**2
        white = np.full_like(ell, nw)
        red = np.zeros_like(ell)
        if self.ell_knee:
            with np.errstate(divide='ignore'):
                red = np.where(ell > 0, nw * (np.where(ell > 0, ell, 1.) / self.ell_knee)**self.alpha_knee, 0.)
        return {'white': white, 'red': red, 'total': white + red}[part]

    def beam_ell(self, ell):
        """Beam transfer function B_ell."""
        ell = np.asarray(ell, dtype=np.float64)
        if self.beam is None:
            return gaussian_beam(ell, self.fwhm) if self.fwhm > 0 else np.ones_like(ell)
        if callable(self.beam):
            return np.asarray(self.beam(ell), dtype=np.float64)
        bl, bb = (np.asarray(a, dtype=np.float64) for a in self.beam)
        return np.interp(ell, bl, bb, right=bb[-1])

    def __repr__(self):
        s = "Band(%s: %.4g GHz, %s, FWHM %.3g', noise %.4g %s" % (self.name, self.freq, self.bandpass_info(),
                                                              self.fwhm, self.noise, self.noise_unit)
        if self.ell_knee:
            s += ", ell_knee %.4g alpha %.3g" % (self.ell_knee, self.alpha_knee)
        return s + ")"

    def bandpass_info(self):
        """Short description of the bandpass."""
        if self.kind == 'delta':
            s = 'monochromatic'
        elif self.kind == 'tophat':
            s = 'top-hat %.4g-%.4g GHz' % (self._nu[0], self._nu[-1])
        else:
            s = '%s (nodes %.4g-%.4g GHz)' % (self.bandpass_name, self.nodes[0], self.nodes[-1])
        if self.shift:
            s += ' shifted %+.3g GHz' % self.shift
        return s


# ----------------------------------------------------------------------------------------------
# telescope presets
# ----------------------------------------------------------------------------------------------
def _ukarcmin(nwhite_uk2_sr):
    """white noise power N_white (uK^2 sr) -> uK-arcmin"""
    return np.sqrt(nwhite_uk2_sr) / ARCMIN


def _knee(nred, nwhite, ell_ref=1000., alpha=-3.5):
    """ell where N_red (ell/ell_ref)^alpha = N_white"""
    return ell_ref * (nred / nwhite)**(-1. / alpha)


# Atmospheric 1/f knees at the Atacama site, from the CMB-S4 Chile-LAT noise model (N_red = N_white,
# alpha = -3.5; Table VIII of arXiv:2008.11688). Used as an approximation for SO and ACT as well.
_ATACAMA_KNEE = {27: 415., 39: 391., 93: 1932., 145: 3917., 225: 6740., 280: 6792.}

PRESETS = {
    'Planck': dict(
        description='Planck full-mission frequency maps (LFI + HFI)',
        reference='Planck 2018 results I (arXiv:1807.06205) Table 4: frequencies, effective beam FWHM, white '
                  'noise (uK_CMB deg; kJy/sr deg at 545/857 GHz) and calibration uncertainties. Bandpasses: '
                  'Planck 2018 RIMO (HFI R3.00, LFI R3.31; pymsz.bandpasses). measured_bandpass=False: 20% (LFI) '
                  'and 30% (HFI) top-hats.',
        pixel=1.7178,  # HEALPix Nside = 2048
        bands=[
            dict(name='LFI030', freq=28.4, fwhm=32.29, noise=2.5, noise_unit='uK-deg', frac_bandwidth=0.2, calib_error=0.0017,
                 bandpass='Planck/LFI030'),
            dict(name='LFI044', freq=44.1, fwhm=27.94, noise=2.7, noise_unit='uK-deg', frac_bandwidth=0.2, calib_error=0.0012,
                 bandpass='Planck/LFI044'),
            dict(name='LFI070', freq=70.4, fwhm=13.08, noise=3.5, noise_unit='uK-deg', frac_bandwidth=0.2, calib_error=0.0020,
                 bandpass='Planck/LFI070'),
            dict(name='HFI100', freq=100., fwhm=9.66, noise=1.29, noise_unit='uK-deg', frac_bandwidth=0.3, calib_error=0.00008,
                 bandpass='Planck/HFI100', nnodes=13),
            dict(name='HFI143', freq=143., fwhm=7.22, noise=0.55, noise_unit='uK-deg', frac_bandwidth=0.3, calib_error=0.00021,
                 bandpass='Planck/HFI143', nnodes=13),
            dict(name='HFI217', freq=217., fwhm=4.90, noise=0.78, noise_unit='uK-deg', frac_bandwidth=0.3, calib_error=0.00028,
                 bandpass='Planck/HFI217', nnodes=13),
            dict(name='HFI353', freq=353., fwhm=4.92, noise=2.56, noise_unit='uK-deg', frac_bandwidth=0.3, calib_error=0.00024,
                 bandpass='Planck/HFI353', nnodes=13),
            dict(name='HFI545', freq=545., fwhm=4.67, noise=0.78, noise_unit='kJy/sr-deg', frac_bandwidth=0.3, calib_error=0.01,
                 bandpass='Planck/HFI545', nnodes=13),
            dict(name='HFI857', freq=857., fwhm=4.22, noise=0.72, noise_unit='kJy/sr-deg', frac_bandwidth=0.3, calib_error=0.03,
                 bandpass='Planck/HFI857', nnodes=13),
        ]),
    'SPT-SZ': dict(
        description='SPT-SZ 2500 deg^2 survey (2008-2011)',
        reference='Bleem et al. 2015 (ApJS 216, 27): depths ~40, 18, 70 uK-arcmin at 95, 150, 220 GHz. Beam FWHM '
                  '(1.7, 1.2, 1.0 arcmin) are approximate. Bandpasses: 2009-2011 seasons (Bleem et al. 2022; '
                  'pymsz.bandpasses, also SPT-SZ/f*_2008). No 1/f noise: SPT time streams are high-pass '
                  'filtered, consider transfer=("scan", ell).',
        pixel=0.25,
        bands=[
            dict(name='f095', freq=95., fwhm=1.7, noise=40., frac_bandwidth=0.25, bandpass='SPT-SZ/f095'),
            dict(name='f150', freq=150., fwhm=1.2, noise=18., frac_bandwidth=0.25, bandpass='SPT-SZ/f150'),
            dict(name='f220', freq=220., fwhm=1.0, noise=70., frac_bandwidth=0.25, bandpass='SPT-SZ/f220'),
        ]),
    'SPT-3G': dict(
        description='SPT-3G Main field, 2019-2020 data (D1)',
        reference='SPT-3G D1 (arXiv:2506.20707, 2603.20163): temperature white noise 5.4, 4.4, 16.2 uK-arcmin and '
                  'beam FWHM 1.6, 1.2, 1.0 arcmin at 95, 150, 220 GHz. Bandpasses: SPT-3G 2019 FTS measurement '
                  '(Sobrin et al. 2022; pymsz.bandpasses). No 1/f noise, consider transfer=("scan", ell).',
        pixel=0.25,
        bands=[
            dict(name='f095', freq=95., fwhm=1.6, noise=5.4, frac_bandwidth=0.25, bandpass='SPT-3G/f095'),
            dict(name='f150', freq=150., fwhm=1.2, noise=4.4, frac_bandwidth=0.25, bandpass='SPT-3G/f150'),
            dict(name='f220', freq=220., fwhm=1.0, noise=16.2, frac_bandwidth=0.25, bandpass='SPT-3G/f220'),
        ]),
    'ACT-DR6': dict(
        description='ACT DR6 wide survey (~12,000 deg^2)',
        reference='Coulton et al. 2024 (arXiv:2307.01258) Table 1: white noise 20, 20, 65 uK-arcmin, beams 2.1, 1.4, '
                  '1.0 arcmin (f090, f150, f220). Bandpasses: DR6.02 passbands of the arrays (pymsz.bandpasses), '
                  'coadded with equal weights (f090: PA5+PA6, f150: PA4+PA5+PA6, f220: PA4); use e.g. '
                  "band_options={'f150': {'bandpass': 'ACT-DR6/pa5_f150'}} for one array. The atmospheric knees "
                  '(Atacama values of the CMB-S4 model) are approximate.',
        pixel=0.5,
        bands=[
            dict(name='f090', freq=90., fwhm=2.1, noise=20., frac_bandwidth=0.3, ell_knee=_ATACAMA_KNEE[93],
                 bandpass='ACT-DR6/f090'),
            dict(name='f150', freq=150., fwhm=1.4, noise=20., frac_bandwidth=0.3, ell_knee=_ATACAMA_KNEE[145],
                 bandpass='ACT-DR6/f150'),
            dict(name='f220', freq=220., fwhm=1.0, noise=65., frac_bandwidth=0.3, ell_knee=_ATACAMA_KNEE[225],
                 bandpass='ACT-DR6/f220'),
        ]),
    'SO-LAT': dict(
        description='Simons Observatory LAT wide survey, baseline sensitivity',
        reference='SO Collaboration 2019 (JCAP 02, 056; arXiv:1808.07445): beams and baseline white noise. Atmospheric '
                  'knees (alpha = -3.5) from the CMB-S4 Chile-LAT model (same site, approximate). 30% top-hat bands '
                  'are approximate. Red noise is uncorrelated between bands unless red_noise_corr is set.',
        pixel=0.5,
        bands=[
            dict(name='f027', freq=27., fwhm=7.4, noise=71., frac_bandwidth=0.3, ell_knee=_ATACAMA_KNEE[27]),
            dict(name='f039', freq=39., fwhm=5.1, noise=36., frac_bandwidth=0.3, ell_knee=_ATACAMA_KNEE[39]),
            dict(name='f093', freq=93., fwhm=2.2, noise=8.0, frac_bandwidth=0.3, ell_knee=_ATACAMA_KNEE[93]),
            dict(name='f145', freq=145., fwhm=1.4, noise=10., frac_bandwidth=0.3, ell_knee=_ATACAMA_KNEE[145]),
            dict(name='f225', freq=225., fwhm=1.0, noise=22., frac_bandwidth=0.3, ell_knee=_ATACAMA_KNEE[225]),
            dict(name='f280', freq=280., fwhm=0.9, noise=54., frac_bandwidth=0.3, ell_knee=_ATACAMA_KNEE[280]),
        ]),
    'SO-LAT-goal': dict(
        description='Simons Observatory LAT wide survey, goal sensitivity',
        reference='SO Collaboration 2019 (arXiv:1808.07445) goal white noise (also Table VI of arXiv:2008.11688). '
                  'Knees and bands as SO-LAT (approximate).',
        pixel=0.5,
        bands=[
            dict(name='f027', freq=27., fwhm=7.4, noise=52., frac_bandwidth=0.3, ell_knee=_ATACAMA_KNEE[27]),
            dict(name='f039', freq=39., fwhm=5.1, noise=27., frac_bandwidth=0.3, ell_knee=_ATACAMA_KNEE[39]),
            dict(name='f093', freq=93., fwhm=2.2, noise=5.8, frac_bandwidth=0.3, ell_knee=_ATACAMA_KNEE[93]),
            dict(name='f145', freq=145., fwhm=1.4, noise=6.3, frac_bandwidth=0.3, ell_knee=_ATACAMA_KNEE[145]),
            dict(name='f225', freq=225., fwhm=1.0, noise=15., frac_bandwidth=0.3, ell_knee=_ATACAMA_KNEE[225]),
            dict(name='f280', freq=280., fwhm=0.9, noise=37., frac_bandwidth=0.3, ell_knee=_ATACAMA_KNEE[280]),
        ]),
    'CMB-S4': dict(
        description='CMB-S4 Chile LATs wide survey (f_sky = 0.65), design of 2020',
        reference='arXiv:2008.11688 Table VIII (CMB-S4 wiki design): beams, N_white and ell_knee (N_red = N_white, '
                  'alpha = -3.5). 30% top-hat bands are approximate.',
        pixel=0.5,
        bands=[
            dict(name='f027', freq=27., fwhm=7.4, noise=_ukarcmin(3.9e-5), frac_bandwidth=0.3, ell_knee=415.),
            dict(name='f039', freq=39., fwhm=5.1, noise=_ukarcmin(1.2e-5), frac_bandwidth=0.3, ell_knee=391.),
            dict(name='f093', freq=93., fwhm=2.2, noise=_ukarcmin(3.0e-7), frac_bandwidth=0.3, ell_knee=1932.),
            dict(name='f145', freq=145., fwhm=1.4, noise=_ukarcmin(3.7e-7), frac_bandwidth=0.3, ell_knee=3917.),
            dict(name='f225', freq=225., fwhm=0.9, noise=_ukarcmin(4.0e-6), frac_bandwidth=0.3, ell_knee=6740.),
            dict(name='f278', freq=278., fwhm=0.7, noise=_ukarcmin(2.4e-5), frac_bandwidth=0.3, ell_knee=6792.),
        ]),
    'FYST': dict(
        description='CCAT / Fred Young Submillimeter Telescope, Prime-Cam wide survey (20,000 deg^2, 4000 h)',
        reference='CCAT-Prime Collaboration 2023 (ApJS 264, 7; arXiv:2107.10364) Table 1: band centres and widths, '
                  'beams, white noise (uK-arcmin); 1/f knee from N_red/N_white with ell_ref = 1000, alpha = -3.5.',
        pixel=0.125,
        bands=[
            dict(name='f220', freq=220., bandwidth=56., fwhm=59. / 60, noise=15., ell_knee=_knee(0.021, 2.0e-5)),
            dict(name='f280', freq=280., bandwidth=60., fwhm=47. / 60, noise=28., ell_knee=_knee(0.14, 6.8e-5)),
            dict(name='f350', freq=350., bandwidth=35., fwhm=37. / 60, noise=107., ell_knee=_knee(3.6, 9.4e-4)),
            dict(name='f410', freq=410., bandwidth=30., fwhm=32. / 60, noise=407., ell_knee=_knee(22., 1.4e-2)),
            dict(name='f850', freq=850., bandwidth=97., fwhm=15. / 60, noise=6.8e5, ell_knee=_knee(8.1e6, 3.9e4)),
        ]),
}
PRESETS['CCAT'] = PRESETS['FYST']


# ----------------------------------------------------------------------------------------------
# flat-sky helpers
# ----------------------------------------------------------------------------------------------
def _ell_grid(shape, pix_rad):
    """(lx, ly, ell) for numpy.fft.rfft2 of an image of `shape` with square pixels of pix_rad."""
    ly = 2. * np.pi * np.fft.fftfreq(shape[0], d=pix_rad)[:, None]
    lx = 2. * np.pi * np.fft.rfftfreq(shape[1], d=pix_rad)[None, :]
    ell = np.sqrt(lx**2 + ly**2)
    return np.broadcast_to(lx, ell.shape), np.broadcast_to(ly, ell.shape), ell


def gaussian_field(amp, white, pix_rad):
    """Gaussian random field with power spectrum C = amp^2 (unit^2 sr) on the rfft grid, from a map
    `white` of unit-variance white noise (pixel size pix_rad). amp is a 2D array on the rfft2 grid."""
    return np.fft.irfft2(np.fft.rfft2(white) * amp / pix_rad, s=white.shape)


def _rebin_matrix(n_in, p_in, n_out, p_out):
    """Sparse (n_out, n_in) matrix averaging centred pixels of size p_in into centred pixels of size p_out
    (area weighted, i.e. the exact pixel window)."""
    from scipy.sparse import csr_matrix
    e_in = (np.arange(n_in + 1) - n_in / 2.) * p_in
    e_out = (np.arange(n_out + 1) - n_out / 2.) * p_out
    i0 = np.clip(np.floor((e_out[:-1] - e_in[0]) / p_in).astype(int), 0, n_in - 1)
    nmax = int(np.ceil(p_out / p_in)) + 2
    rows = np.repeat(np.arange(n_out), nmax)
    cols = (i0[:, None] + np.arange(nmax)[None, :]).ravel()
    ok = cols < n_in
    rows, cols = rows[ok], cols[ok]
    ov = np.minimum(e_out[rows + 1], e_in[cols + 1]) - np.maximum(e_out[rows], e_in[cols])
    ok = ov > 1e-12 * p_out
    return csr_matrix((ov[ok] / p_out, (rows[ok], cols[ok])), shape=(n_out, n_in))


def _transfer_2d(spec, lx, ly, ell):
    r"""Evaluate a map transfer function specification on the Fourier grid.
    spec : None; a callable F(lx, ly); (ell, F_ell) arrays (isotropic, tabulated);
           a dict {'kind': 'highpass' | 'scan' | 'lowpass', 'ell': ell_c, 'order': 4, 'axis': 'x'};
           a tuple ('highpass' | 'scan' | 'lowpass', ell_c); or a list of these (multiplied).
    'highpass' : isotropic, F = 1 / (1 + (ell_c/ell)^(2 order))      (F(ell_c) = 1/2)
    'scan'     : the same in |ell_x| (or |ell_y| for axis='y'), as for time-stream filtering along scans
                 in RA (columns of the image)
    'lowpass'  : isotropic, F = 1 / (1 + (ell/ell_c)^(2 order))
    """
    if spec is None:
        return 1.
    if callable(spec):
        return np.asarray(spec(lx, ly), dtype=np.float64)
    if isinstance(spec, dict):
        kind, ec = spec['kind'], float(spec['ell'])
        n = 2 * spec.get('order', 4)
        if kind == 'highpass':
            with np.errstate(divide='ignore'):
                return np.where(ell > 0, 1. / (1. + (ec / np.where(ell > 0, ell, 1.))**n), 0.)
        if kind == 'scan':
            l1 = np.abs(lx) if spec.get('axis', 'x') == 'x' else np.abs(ly)
            with np.errstate(divide='ignore'):
                return np.where(l1 > 0, 1. / (1. + (ec / np.where(l1 > 0, l1, 1.))**n), 0.)
        if kind == 'lowpass':
            return 1. / (1. + (ell / ec)**n)
        raise ValueError("Unknown transfer function kind %s" % kind)
    if isinstance(spec, (tuple, list)):
        if len(spec) == 2 and isinstance(spec[0], str):
            return _transfer_2d({'kind': spec[0], 'ell': spec[1]}, lx, ly, ell)
        if len(spec) == 2 and not callable(spec[0]) and not isinstance(spec[0], (dict, str)):
            a0, a1 = np.asarray(spec[0]), np.asarray(spec[1])
            if a0.ndim == 1 and a0.size > 1 and a0.shape == a1.shape and a0.dtype.kind in 'iuf' \
                    and a1.dtype.kind in 'iuf':
                return np.interp(ell, a0.astype(np.float64), a1.astype(np.float64))
        out = 1.
        for s in spec:
            out = out * _transfer_2d(s, lx, ly, ell)
        return out
    raise ValueError("Cannot understand the transfer function %r" % (spec,))


def cmb_power_spectrum(cmb, lmax=12000):
    r"""Primary (lensed) CMB temperature power spectrum.

    Parameters
    ----------
    cmb : str, True or tuple
        'camb' or True: compute it with CAMB (``pip install camb``) for the Planck 2018 best fit.
        A file name: a CAMB-style text file with the columns ell, D_ell = ell (ell+1) C_ell / 2 pi
        (uK^2), .... ``(ell, C_ell)`` arrays with C_ell in uK^2.
    lmax : int, optional
        Maximum multipole for CAMB. Default 12000.

    Returns
    -------
    ell, C_ell : ndarray
        Multipoles and the power spectrum in uK^2.
    """
    if cmb is True or (isinstance(cmb, str) and cmb.lower() == 'camb'):
        try:
            import camb
        except ImportError:
            raise ImportError("cmb='camb' needs CAMB (pip install camb); or give (ell, C_ell) or a file")
        pars = camb.set_params(H0=67.36, ombh2=0.02237, omch2=0.1200, mnu=0.06, omk=0, tau=0.0544,
                               As=2.100e-9, ns=0.9649, lmax=lmax)
        res = camb.get_results(pars)
        cl = res.get_lensed_scalar_cls(lmax=lmax, CMB_unit='muK', raw_cl=True)[:, 0]
        return np.arange(cl.size, dtype=np.float64), cl
    if isinstance(cmb, str):
        d = np.loadtxt(cmb)
        ell, dl = d[:, 0], d[:, 1]
        return ell, 2. * np.pi * dl / (ell * (ell + 1.))
    ell, cl = (np.asarray(a, dtype=np.float64) for a in cmb)
    return ell, cl


# ----------------------------------------------------------------------------------------------
# sky inputs
# ----------------------------------------------------------------------------------------------
class SkyMaps(object):
    r"""SZ (or any) images at a set of frequencies, usable as the input of `Telescope.observe`.

    Parameters
    ----------
    maps : array_like
        Images, shape (nfreq, ny, nx) (or (ny, nx) for one frequency), in `unit`.
    freqs : float or array_like
        Frequencies in GHz.
    unit : str, optional
        'MJy/sr' (default), 'Jy/pixel', 'K', 'uK' or 'Dn'.
    pixel_arcsec : float
        Pixel size in arcsec (required).
    Tcmb : float, optional
        CMB temperature in K. Default 2.7255.
    sp : tuple, optional
        Sky position (RA, Dec) in degrees, for the headers. Default (0, 0).
    redshift : float, optional
        Redshift, for the headers. Default 0.

    Notes
    -----
    Images at other frequencies are interpolated (cubic spline in x) inside the given range. Give
    the images at ``telescope.frequencies()`` for accurate band averages.

    Examples
    --------
    >>> sky = pymsz.SkyMaps(maps, freqs=[90., 150., 220.], unit='MJy/sr', pixel_arcsec=5.)
    >>> sky = pymsz.SkyMaps.from_fits('SZ-cube.fits')     # written by pymsz
    >>> obs = pymsz.Telescope.preset('SPT-3G').observe(sky)
    """

    def __init__(self, maps, freqs, unit='MJy/sr', pixel_arcsec=None, Tcmb=TCMB, sp=(0., 0.), redshift=0.):
        if pixel_arcsec is None:
            raise ValueError("pixel_arcsec is needed")
        self.freqs = np.atleast_1d(np.asarray(freqs, dtype=np.float64))
        maps = np.asarray(maps, dtype=np.float64)
        if maps.ndim == 2:
            maps = maps[None]
        if maps.shape[0] != self.freqs.size:
            raise ValueError("maps must have shape (len(freqs), ny, nx)")
        self.ar = float(pixel_arcsec)
        self.Tcmb = Tcmb
        self.sp = sp
        self.red = redshift
        self.unit = sz_spectra.check_unit(unit)
        self.dndata = sz_spectra.unit_to_dn(maps, self.freqs, unit, Tcmb, self.ar)

    @classmethod
    def from_fits(cls, fname):
        """Read an SZ cube written by SZpack_model.write_fits_image or TT/TK_model.write_sz_fits."""
        import astropy.io.fits as pf
        with pf.open(fname) as hl:
            h, d = hl[0].header, np.asarray(hl[0].data, dtype=np.float64)
        n = d.shape[0]
        freqs = [h['FREQ%d' % (k + 1)] for k in range(n)]
        return cls(d, freqs, unit=h['PIXVAL'], pixel_arcsec=h['AGLRES'], Tcmb=h.get('TCMB', TCMB),
                   sp=(h.get('CRVAL1', 0.), h.get('CRVAL2', 0.)), redshift=h.get('REDSHIFT', 0.))

    def sz_maps(self, freqs, unit='Dn'):
        """Images at `freqs` (GHz) in `unit`, interpolated in x if needed. Used by Telescope.observe."""
        dn = sz_spectra.interpolate_dn(self.dndata, self.freqs, freqs, self.Tcmb, what='SkyMaps images')
        return sz_spectra.dn_to_unit(dn, np.atleast_1d(freqs), unit, self.Tcmb, self.ar)


# ----------------------------------------------------------------------------------------------
# telescope
# ----------------------------------------------------------------------------------------------
class Telescope(object):
    r"""A (sub-)millimetre telescope: a set of Bands plus map-level properties.

    Parameters
    ----------
    bands : list of Band or dict
        The bands, as `Band` objects or dicts of Band arguments.
    name : str, optional
        Telescope name. Default 'custom'.
    pixel : float, optional
        Map pixel size in arcmin. Default None: keep the pixels of the simulated images.
    transfer : optional
        Transfer function of the map making / filtering, applied to the sky signal (not to the
        noise, whose spectrum is that of the final maps). Default None. One of

        - ``('scan', ell_c)``: high-pass in abs(ell_x), as for time streams filtered along scans in
          RA (the image x axis), F = 1 / (1 + (ell_c / abs(ell_x))^8);
        - ``('highpass', ell_c)``: isotropic, F = 1 / (1 + (ell_c / ell)^8);
        - ``('lowpass', ell_c)``: isotropic, F = 1 / (1 + (ell / ell_c)^8);
        - ``{'kind': 'scan', 'ell': 300, 'order': 4, 'axis': 'x'}``: the same with the filter order
          (exponent 2 order) and, for 'scan', the axis;
        - ``(ell, F_ell)`` arrays: a tabulated isotropic transfer function;
        - a function F(lx, ly);
        - a list of these, which are multiplied.
    red_noise_corr : float or ndarray, optional
        Correlation coefficient of the 1/f (atmospheric) noise between bands: a number (the same for
        all pairs) or an (nband, nband) matrix. Default 0 (independent).
    pointing_rms : float, optional
        RMS pointing jitter in arcsec, which broadens the effective beam. Default 0.
    description, reference : str, optional
        Free text.

    Notes
    -----
    Use :meth:`Telescope.preset` for SPT-SZ, SPT-3G, ACT-DR6, Planck, SO-LAT, SO-LAT-goal, CMB-S4
    and FYST (CCAT). The preset numbers are published (or approximate, see `reference`) values.
    Planck, SPT-SZ, SPT-3G and ACT-DR6 use the measured bandpasses shipped with pymsz
    (`pymsz.bandpasses`), the others top-hats. Override them (e.g. with measured beams) with
    `band_options` for precision work.

    A telescope can be indexed by band number or name: ``tel[0]``, ``tel['f150']``.

    Examples
    --------
    >>> tel = pymsz.Telescope.preset('SPT-3G')
    >>> tel = pymsz.Telescope([pymsz.Band(150., fwhm=1.0, noise=3., frac_bandwidth=0.25, ell_knee=2000),
    ...                        pymsz.Band(220., fwhm=0.8, noise=10., bandpass='my_220GHz_band.txt')],
    ...                       pixel=0.25, transfer=('scan', 300), red_noise_corr=0.5)
    >>> obs = tel.observe([tt, tk], seed=1)
    """

    def __init__(self, bands, name='custom', pixel=None, transfer=None, red_noise_corr=0., pointing_rms=0.,
                 description='', reference=''):
        self.bands = [b if isinstance(b, Band) else Band(**b) for b in bands]
        if len(self.bands) == 0:
            raise ValueError("A telescope needs at least one band")
        names = [b.name for b in self.bands]
        if len(set(names)) != len(names):
            raise ValueError("Band names must be unique: %s" % names)
        self.name = name
        self.pixel = pixel
        self.transfer = transfer
        self.red_noise_corr = red_noise_corr
        self.pointing_rms = float(pointing_rms)
        self.description = description
        self.reference = reference

    @classmethod
    def preset(cls, name, bands=None, band_options=None, measured_bandpass=True, **kwargs):
        r"""A telescope from `PRESETS`.

        Parameters
        ----------
        name : str
            One of :meth:`list_presets` (case insensitive): 'Planck', 'SPT-SZ', 'SPT-3G', 'ACT-DR6',
            'SO-LAT', 'SO-LAT-goal', 'CMB-S4', 'FYST' (= 'CCAT').
        bands : list, optional
            Names (or nominal frequencies) of the bands to keep. Default: all.
        band_options : dict, optional
            Band arguments applied to all bands (e.g. ``{'nnodes': 5}``), or ``{band_name: {...}}``
            for individual bands, e.g. ``{'f150': {'bandpass': 'f150.txt'}}``,
            ``{'f090': {'bandpass': 'ACT-DR6/pa5_f090'}}`` or ``{'f150': {'frac_bandwidth': 0.3}}``.
            A `bandwidth` or `frac_bandwidth` replaces the measured bandpass; a `bandpass` replaces
            the top-hat.
        measured_bandpass : bool, optional
            True (default): the measured bandpasses of the preset, where available. False: the
            top-hat approximations.
        **kwargs
            Telescope arguments overriding the preset (`pixel`, `transfer`, `red_noise_corr`, ...).

        Returns
        -------
        Telescope

        Examples
        --------
        >>> tel = Telescope.preset('ACT-DR6', band_options={'f150': {'bandpass': 'ACT-DR6/pa5_f150'}})
        >>> tel = Telescope.preset('Planck', bands=['HFI143', 'HFI353'])
        """
        keys = {k.lower(): k for k in PRESETS}
        if str(name).lower() not in keys:
            raise ValueError("Unknown telescope %s. Presets: %s" % (name, cls.list_presets()))
        key = keys[str(name).lower()]
        p = PRESETS[key]
        blist = [dict(b) for b in p['bands']]
        if bands is not None:
            want = list(bands) if isinstance(bands, (list, tuple, np.ndarray)) else [bands]

            def match(b):
                for w in want:
                    if isinstance(w, str) and w == b['name']:
                        return True
                    if not isinstance(w, str) and np.isclose(float(w), b['freq']):
                        return True
                return False
            blist = [b for b in blist if match(b)]
            if not blist:
                raise ValueError("No band of %s matches %s" % (key, bands))
        per_band = bool(band_options) and all(isinstance(v, dict) for v in band_options.values())
        for b in blist:
            if not measured_bandpass:
                b.pop('bandpass', None)
                b.pop('nnodes', None)
            opt = (band_options.get(b['name'], {}) if per_band else band_options) or {}
            if 'bandwidth' in opt or 'frac_bandwidth' in opt:  # a top-hat replaces the measured bandpass
                b.pop('bandpass', None)
                b.pop('bandwidth', None)
                b.pop('frac_bandwidth', None)
            b.update(opt)
            if b.get('bandpass') is not None:  # a measured bandpass replaces the top-hat
                b.pop('frac_bandwidth', None)
                b.pop('bandwidth', None)
        args = dict(name=key, pixel=p.get('pixel'), description=p.get('description', ''),
                    reference=p.get('reference', ''))
        args.update(kwargs)
        return cls(blist, **args)

    @staticmethod
    def list_presets():
        """Names of the telescope presets."""
        return sorted(PRESETS)

    def __len__(self):
        return len(self.bands)

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self.bands[key]
        for b in self.bands:
            if b.name == key:
                return b
        raise KeyError(key)

    def __repr__(self):
        s = "Telescope %s (%d bands, pixel %s arcmin)\n" % (self.name, len(self.bands),
                                                            'input' if self.pixel is None else '%g' % self.pixel)
        return s + "\n".join("  " + repr(b) for b in self.bands)

    def frequencies(self):
        """Sorted frequencies (GHz) at which the sky images are needed for the band integrations.
        Give them to SZpack_model (freqs=telescope.frequencies()) for exact band averages."""
        f = np.concatenate([b.nodes for b in self.bands])
        return np.unique(np.round(f, 9))

    def tsz_response(self, unit='uK_CMB', Tcmb=TCMB):
        """Band-averaged non-relativistic tSZ signal per unit Compton y for every band."""
        return np.array([b.response(sz_spectra.tsz_nonrel, unit, Tcmb) for b in self.bands])

    def _corr_matrix(self):
        nb = len(self.bands)
        r = np.asarray(self.red_noise_corr, dtype=np.float64)
        if r.ndim == 0:
            m = np.full((nb, nb), float(r))
            np.fill_diagonal(m, 1.)
            return m
        if r.shape != (nb, nb):
            raise ValueError("red_noise_corr must be a number or a (%d, %d) matrix" % (nb, nb))
        return r

    # --------------------------------------------------------------------------------------------
    def observe(self, sky, unit='uK_CMB', noise=True, cmb=None, foregrounds=None, calibration=True,
                seed=None, pad=None, cmb_seed=None):
        r"""Mock observation of SZ images.

        Integrates the sky over the bandpasses, adds the CMB and foregrounds, applies the beam,
        pointing jitter, transfer function, pixelisation and calibration errors, and adds the noise.

        Parameters
        ----------
        sky : model or list of models
            An object with a ``sz_maps(freqs, unit)`` method: `TT_model`, `TK_model` (with the
            spectrum chosen in their ``to_sz``, non-relativistic by default), `SZpack_model` or
            `SkyMaps`. A list of them is added, e.g. ``[tt, tk]`` for tSZ + kSZ. All must have the
            same image and pixel sizes.
        unit : str, optional
            'uK_CMB' (default), 'K_CMB' or 'MJy/sr' (IRAS convention at the band frequency).
        noise : bool, optional
            Add the instrument and atmospheric noise. Default True.
        cmb : optional
            Add a primary CMB realisation: None (default, no CMB), 'camb', a CAMB-style file name, or
            ``(ell, C_ell)`` with C_ell in uK^2. See :func:`cmb_power_spectrum`.
        foregrounds : ndarray or dict, optional
            Extra sky maps (e.g. CIB, radio sources, Galactic dust), already band-averaged and in
            `unit`, at the resolution of the simulated images: an array (nband, ny, nx) or a dict
            ``{band_name: map}``. They get the same beam, filtering, pixelisation and gain as the SZ
            signal.
        calibration : bool, optional
            Draw the gain errors of the bands (`Band` ``calib_error``). Default True.
        seed : int, optional
            Seed of the random numbers (CMB, gains, noise).
        pad : float, optional
            Zero padding (in units of the image size, on each side) for the Fourier convolutions.
            Default: 3 beam FWHM, at most half the image size (half the image size with a CMB).
        cmb_seed : int, optional
            Separate seed of the CMB realisation. Default None: drawn with `seed`. With the same
            `cmb_seed` (and the default `pad`), telescopes observing the same SZ images see the same
            CMB sky, while their noise (from `seed`) stays independent.

        Returns
        -------
        MockObservation
            The maps and their components.
        """
        rng = np.random.default_rng(seed)
        unit = _obs_unit(unit)
        skies = list(sky) if isinstance(sky, (list, tuple)) else [sky]
        for s in skies:
            if not hasattr(s, 'sz_maps'):
                raise TypeError("sky must provide sz_maps(freqs, unit) (TT_model, TK_model, SZpack_model or "
                                "SkyMaps), got %s" % type(s))
        ar = float(skies[0].ar)
        Tcmb = float(getattr(skies[0], 'Tcmb', TCMB))
        for s in skies[1:]:
            if not np.isclose(float(s.ar), ar, rtol=1e-6):
                raise ValueError("The sky components have different pixel sizes (%g, %g arcsec)" % (ar, s.ar))
            if not np.isclose(float(getattr(s, 'Tcmb', TCMB)), Tcmb):
                raise ValueError("The sky components use different Tcmb")

        # 1. band-averaged SZ signal at the simulation resolution
        nodes = self.frequencies()
        dn = None
        for s in skies:
            d = np.asarray(s.sz_maps(nodes, 'Dn'), dtype=np.float64)
            if dn is not None and d.shape != dn.shape:
                raise ValueError("The sky components have different image sizes %s, %s" % (dn.shape, d.shape))
            dn = d if dn is None else dn + d
        ny, nx = dn.shape[1:]
        nb = len(self.bands)
        sz_true = np.empty((nb, ny, nx))
        for i, b in enumerate(self.bands):
            idx = [int(np.argmin(np.abs(nodes - f))) for f in b.nodes]
            sz_true[i] = b.integrate(dn[idx], unit, Tcmb)
        del dn

        # foregrounds
        fg = None
        if foregrounds is not None:
            fg = np.zeros((nb, ny, nx))
            if isinstance(foregrounds, dict):
                for k, v in foregrounds.items():
                    fg[[b.name for b in self.bands].index(k)] = v
            else:
                fg[:] = np.asarray(foregrounds, dtype=np.float64)

        # 2. geometry: padded fine grid for the convolutions, output grid
        pix_in = ar * ARCSEC
        fwhm_max = max([b.fwhm for b in self.bands] + [self.pointing_rms / 60. * 2.355])
        if pad is None:
            npad = int(min(np.ceil(3. * fwhm_max * 60. / ar), max(ny, nx) // 2))
        else:
            npad = int(np.ceil(pad * max(ny, nx)))
        if cmb is not None and cmb is not False:
            npad = max(npad, max(ny, nx) // 2)  # better large-scale CMB on the patch
        shape_p = (ny + 2 * npad, nx + 2 * npad)
        lx, ly, ell = _ell_grid(shape_p, pix_in)
        common = _transfer_2d(self.transfer, lx, ly, ell)
        if self.pointing_rms > 0:
            common = common * np.exp(-0.5 * ell**2 * (self.pointing_rms * ARCSEC)**2)

        p_out = ar / 60. if self.pixel is None else float(self.pixel)  # arcmin
        same = np.isclose(p_out, ar / 60., rtol=1e-9)
        if same:
            my, mx = ny, nx
        else:
            my, mx = int(np.floor(ny * ar / 60. / p_out + 1e-9)), int(np.floor(nx * ar / 60. / p_out + 1e-9))
            if my < 1 or mx < 1:
                raise ValueError("The telescope pixel (%g arcmin) is larger than the image" % p_out)
            Ry = _rebin_matrix(ny, ar / 60., my, p_out)
            Rx = _rebin_matrix(nx, ar / 60., mx, p_out)

        def pixelize(img):
            return img if same else Ry @ (Rx @ img.T).T

        def smooth(img_p, filt):
            out = np.fft.irfft2(np.fft.rfft2(img_p) * filt, s=shape_p)
            return pixelize(out[npad:npad + ny, npad:npad + nx])

        def padded(img):
            out = np.zeros(shape_p)
            out[npad:npad + ny, npad:npad + nx] = img
            return out

        # CMB realisation (uK_CMB) on the padded grid, same for all bands
        cmb_field = None
        if cmb is not None and cmb is not False:
            cl_ell, cl = cmb_power_spectrum(cmb)
            amp = np.sqrt(np.clip(np.interp(ell, cl_ell, cl, left=0., right=0.), 0., None))
            amp[0, 0] = 0.
            crng = rng if cmb_seed is None else np.random.default_rng(cmb_seed)
            cmb_field = gaussian_field(amp, crng.standard_normal(shape_p), pix_in)

        # 3-5. beam, pointing, transfer function, pixelisation
        out_sz = np.empty((nb, my, mx))
        out_cmb = np.zeros((nb, my, mx)) if cmb_field is not None else None
        out_fg = np.zeros((nb, my, mx)) if fg is not None else None
        for i, b in enumerate(self.bands):
            filt = b.beam_ell(ell) * common
            out_sz[i] = smooth(padded(sz_true[i]), filt)
            if cmb_field is not None:
                out_cmb[i] = smooth(cmb_field * 1.0e-6 * b.cmb_to(unit, Tcmb), filt)
            if fg is not None:
                out_fg[i] = smooth(padded(fg[i]), filt)

        # 6. calibration
        gains = np.ones(nb)
        if calibration:
            gains = 1. + np.array([rng.normal(0., b.calib_error) if b.calib_error > 0 else 0. for b in self.bands])

        # 7. noise on the output grid
        out_noise = None
        if noise:
            pix_o = p_out * ARCMIN
            lxo, lyo, ello = _ell_grid((my, mx), pix_o)
            out_noise = np.zeros((nb, my, mx))
            white = rng.standard_normal((nb, my, mx))
            red = rng.standard_normal((nb, my, mx))
            if np.any(self._corr_matrix() != np.eye(nb)):
                L = np.linalg.cholesky(self._corr_matrix())
                red = np.tensordot(L, red, axes=(1, 0))
            for i, b in enumerate(self.bands):
                nw = b.noise_spectrum(ello, unit, Tcmb, 'white')
                out_noise[i] = gaussian_field(np.sqrt(nw), white[i], pix_o)
                if b.ell_knee:
                    nr = b.noise_spectrum(ello, unit, Tcmb, 'red')
                    out_noise[i] += gaussian_field(np.sqrt(nr), red[i], pix_o)

        sky_tot = out_sz.copy()
        if out_cmb is not None:
            sky_tot += out_cmb
        if out_fg is not None:
            sky_tot += out_fg
        maps = gains[:, None, None] * sky_tot
        if out_noise is not None:
            maps += out_noise

        return MockObservation(self, maps=maps, unit=unit, pixel=p_out, sz=out_sz, cmb=out_cmb,
                               foreground=out_fg, noise=out_noise, gains=gains, sz_true=sz_true,
                               input_pixel=ar / 60., Tcmb=Tcmb, seed=seed,
                               sp=getattr(skies[0], 'sp', (0., 0.)), redshift=getattr(skies[0], 'red', 0.),
                               cmb_seed=cmb_seed if cmb_field is not None else None)


class MockObservation(object):
    r"""The result of :meth:`Telescope.observe`.

    Index it with a band name or number to get one map: ``obs['f150']``, ``obs[0]``.

    Attributes
    ----------
    maps : ndarray
        Observed maps, shape (nband, ny, nx), in `unit`: ``gains * (sz + cmb + foreground) + noise``.
    sz : ndarray
        SZ signal after bandpass, beam, filtering and pixelisation (without gain errors).
    cmb, foreground, noise : ndarray or None
        The other components (None if not simulated).
    gains : ndarray
        Calibration factors applied to the bands.
    sz_true : ndarray
        Band-averaged SZ signal at the resolution of the simulation (no beam, no filtering).
    telescope : Telescope
        The telescope; `bands` is its list of bands.
    unit : str
        Unit of the maps.
    pixel, input_pixel : float
        Pixel sizes (arcmin) of the maps and of the input images.
    Tcmb, seed, cmb_seed, sp, red
        CMB temperature, random seeds, sky position (RA, Dec) and redshift.
    """

    def __init__(self, telescope, maps, unit, pixel, sz, cmb, foreground, noise, gains, sz_true,
                 input_pixel, Tcmb, seed, sp, redshift, cmb_seed=None):
        self.telescope = telescope
        self.bands = telescope.bands
        self.maps, self.unit, self.pixel = maps, unit, pixel
        self.sz, self.cmb, self.foreground, self.noise = sz, cmb, foreground, noise
        self.gains, self.sz_true, self.input_pixel = gains, sz_true, input_pixel
        self.Tcmb, self.seed, self.sp, self.red = Tcmb, seed, sp, redshift
        self.cmb_seed = cmb_seed

    def band_index(self, name):
        """Index of the band called `name`."""
        return [b.name for b in self.bands].index(name)

    def __getitem__(self, name):
        return self.maps[self.band_index(name) if isinstance(name, str) else name]

    def __repr__(self):
        return ("MockObservation(%s, %d bands %s, %dx%d pixels of %.4g arcmin, unit %s)"
                % (self.telescope.name, len(self.bands), [b.name for b in self.bands],
                   self.maps.shape[1], self.maps.shape[2], self.pixel, self.unit))

    def write_fits(self, fname, components=True, overwrite=False, header=None):
        r"""Write the observed maps and their components to a FITS file.

        The maps are the primary HDU, shape [nband, ny, nx]. The header describes every band
        (BANDn, FREQn, BPASSn, FWHMn, NOISEn, GAINn) and the seeds.

        Parameters
        ----------
        fname : str
            File name; '.fits' is added if missing.
        components : bool, optional
            Also write the SZ, CMB, FOREGROUND, NOISE and SZTRUE (input resolution) cubes as image
            extensions, when they exist. Default True.
        overwrite : bool, optional
            Replace an existing file. Default False.
        header : dict, optional
            Extra primary-header cards, ``{KEY: value}`` or ``{KEY: (value, comment)}``.
        """
        import astropy.io.fits as pf
        from astropy.time import Time
        from pymsz import version

        if fname[-5:] != ".fits":
            fname = fname + ".fits"

        def wcs(hdr, ny, nx, pix_arcmin):
            hdr["CTYPE1"] = ('RA---TAN', "Coordinate type")
            hdr["CTYPE2"] = ('DEC--TAN', "Coordinate type")
            hdr["CUNIT1"] = ('deg     ', 'Units')
            hdr["CUNIT2"] = ('deg     ', 'Units')
            hdr["CRPIX1"] = (float((nx + 1) / 2.), 'X of reference pixel (image centre)')
            hdr["CRPIX2"] = (float((ny + 1) / 2.), 'Y of reference pixel (image centre)')
            hdr["CRVAL1"] = (float(self.sp[0]), 'RA of reference pixel (deg)')
            hdr["CRVAL2"] = (float(self.sp[1]), 'Dec of reference pixel (deg)')
            hdr["CD1_1"] = (-float(pix_arcmin / 60.), 'RA deg per column pixel')
            hdr["CD1_2"] = (0., 'RA deg per row pixel')
            hdr["CD2_1"] = (0., 'Dec deg per column pixel')
            hdr["CD2_2"] = (float(pix_arcmin / 60.), 'Dec deg per row pixel')
            hdr["BUNIT"] = (self.unit, 'map unit')

        hdu = pf.PrimaryHDU(np.asarray(self.maps, dtype=np.float32))
        h = hdu.header
        wcs(h, self.maps.shape[1], self.maps.shape[2], self.pixel)
        h["TELESCOP"] = (str(self.telescope.name), 'mock telescope')
        for k, b in enumerate(self.bands):
            h["BAND%d" % (k + 1)] = (b.name, 'name of plane %d' % (k + 1))
            h["FREQ%d" % (k + 1)] = (b.freq, 'nominal frequency (GHz) of plane %d' % (k + 1))
            h["BPASS%d" % (k + 1)] = (b.bandpass_info()[:50], 'bandpass of plane %d' % (k + 1))
            h["FWHM%d" % (k + 1)] = (b.fwhm, 'beam FWHM (arcmin) of plane %d' % (k + 1))
            h["NOISE%d" % (k + 1)] = (float(b.noise_uk_arcmin(self.Tcmb)), 'white noise uK_CMB-arcmin')
            h["GAIN%d" % (k + 1)] = (float(self.gains[k]), 'calibration factor applied')
        h["TCMB"] = (float(self.Tcmb), 'CMB temperature (K)')
        h["REDSHIFT"] = (float(self.red), 'redshift of the object')
        h["INPIX"] = (float(self.input_pixel), 'pixel (arcmin) of the input images')
        h["SEED"] = (str(self.seed), 'random seed')
        if self.cmb is not None and self.cmb_seed is not None:
            h["CMBSEED"] = (str(self.cmb_seed), 'random seed of the CMB realisation')
        h["ORIGIN"] = ('PymSZ mock telescope', 'https://github.com/weiguangcui/pymsz.git')
        h["VERSION"] = (version.version, 'Version of the software')
        h["DATE"] = Time.now().tt.isot
        for k, v in (header or {}).items():
            h[k] = v
        hdus = [hdu]
        if components:
            for name, data, pix in [('SZ', self.sz, self.pixel), ('CMB', self.cmb, self.pixel),
                                    ('FOREGROUND', self.foreground, self.pixel), ('NOISE', self.noise, self.pixel),
                                    ('SZTRUE', self.sz_true, self.input_pixel)]:
                if data is None:
                    continue
                e = pf.ImageHDU(np.asarray(data, dtype=np.float32), name=name)
                wcs(e.header, data.shape[1], data.shape[2], pix)
                hdus.append(e)
        pf.HDUList(hdus).writeto(fname, overwrite=overwrite)
