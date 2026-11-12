"""
Mock SZ images (thermal + kinematic, with relativistic corrections) computed with SZpack v2.

SZpack (https://github.com/CMBSPEC/SZpack) is compiled into ``pymsz.szpack.SZpack`` from the sources
in the top-level ``SZpack/`` folder, see ``pymsz/szpack/__init__.py`` and ``build_szpack.sh``.

For details on the computations involved please refer to (and cite) the following references:
Chluba, Nagai, Sazonov, Nelson, MNRAS, 2012, arXiv:1205.5778
Chluba, Switzer, Nagai, Nelson, MNRAS, 2012, arXiv:1211.3206
Lee, Chluba & Holder, MNRAS, 2022, arXiv:2112.10666; Lee & Chluba, 2024
"""

import warnings
import numpy as np
from astropy.cosmology import FlatLambdaCDM, WMAP7
from pymsz.rotate_data import rotate_data, SPH_smoothing
from pymsz import sz_spectra
from pymsz.sz_spectra import (KB as _KB, H as _H, KB_KEV as _KB_KEV, ME_KEV as _ME_KEV, C_KMS as _C_KMS,
                              SIGMA_T as _SIGMA_T, KPC as _KPC, TCMB as _TCMB)

_BETA_REF = 0.01        # velocity used to extract the velocity dependence of the tabulated spectra
_BETA_TINY = 1.0e-10    # "zero" velocity (SZpack is not always happy with betac == 0)
_TE_TABLE_MIN = 0.01    # keV; below this the per-unit-y spectrum is non-relativistic to < 1e-4

# SZpack methods: kind and validity range of Te in keV
_METHODS = {
    'nonrel':     ('analytic', 0., np.inf),    # non-relativistic tSZ + kSZ
    'asymptotic': ('analytic', 0., 20.),      # asymptotic expansion, best for Te < 2 keV
    'CNSN':       ('analytic', 2., 75.),      # basis-function interpolation, 2 keV < Te < 75 keV
    'CNSN_opt':   ('analytic', 0., 60.),      # optimised CNSN basis (upper limit depends on kmax)
    'combo':      ('analytic', 0., 75.),      # asymptotic (< 2 keV) + CNSN (> 2 keV)
    'means':      ('analytic', 0., 75.),      # combo + temperature/velocity moments along LOS
    'precise':    ('numerical', 0., np.inf),  # combo for Te < 75 keV, 3D integration above
    '3D':         ('numerical', 0., np.inf),  # numerical integration of 3D collision integral
    '5D':         ('numerical', 0., np.inf),  # full 5D numerical integration (very slow!)
}


def _canonical_method(method):
    names = {k.lower(): k for k in _METHODS}
    if str(method).lower() not in names:
        raise ValueError("Unknown SZpack method %s. Choose from %s" % (method, list(_METHODS)))
    return names[str(method).lower()]


def _szpack_options(method, szpack_options=None):
    opts = {'T_order': 10 if method == 'asymptotic' else 20, 'beta_order': 2,
            'relative_accuracy': 1.0e-4, 'kmax': 5, 'accuracy_level': 1}
    if szpack_options is not None:
        unknown = set(szpack_options) - set(opts)
        if unknown:
            raise ValueError("Unknown szpack_options %s" % unknown)
        opts.update(szpack_options)
    return opts


def _check_x(x, Tcmb):
    if np.min(x) < 0.1 or np.max(x) > 50.:
        raise ValueError("SZpack needs 0.1 <= x = h nu/(k Tcmb) <= 50, i.e. %.2f GHz <= nu <= %.1f GHz"
                         % (0.1 * Tcmb * _KB / _H / 1e9, 50 * Tcmb * _KB / _H / 1e9))


def szpack_dn(S, xs, Te, betac, muc, method, opts, omega=0., sigma=0., kappa=0.):
    r"""Delta n per unit optical depth from SZpack, for an observer at rest in the CMB frame.

    Parameters
    ----------
    S : module
        The compiled SZpack module, :func:`pymsz.szpack.require`.
    xs : list of float
        Increasing dimensionless frequencies :math:`x = h\nu / k T_\mathrm{CMB}`.
    Te : float
        Electron temperature in keV.
    betac : float
        Cluster speed v/c.
    muc : float
        Direction cosine of the velocity with respect to the line of sight (> 0: towards the
        observer).
    method : str
        Canonical SZpack method name ('combo', 'CNSN', '3D', ..., see `SZpack_model`).
    opts : dict
        SZpack accuracy settings (``T_order``, ``beta_order``, ``relative_accuracy``, ``kmax``,
        ``accuracy_level``).
    omega, sigma, kappa : float, optional
        Temperature and velocity moments along the line of sight, for ``method='means'`` only.

    Returns
    -------
    ndarray
        Delta n / tau at `xs`.
    """
    o, x, m = opts, xs, method
    The = Te / _ME_KEV
    betac = max(betac, _BETA_TINY)
    if m == 'nonrel':
        dn = S.Nonrelativistic.compute_from_variables(x, The, betac, muc, "all", False)
    elif m == 'asymptotic':
        dn = S.Asymptotic.compute_from_variables(x, The, betac, muc, o['T_order'], o['beta_order'],
                                                 "all", False, True)
    elif m == 'CNSN':
        dn = S.CNSN_basis.compute_from_variables(x, The, betac, muc, o['T_order'], o['beta_order'],
                                                 "all", False, True)
    elif m == 'CNSN_opt':
        dn = S.CNSN_basis_opt.compute_from_variables(x, The, betac, muc, o['kmax'], o['accuracy_level'],
                                                     o['beta_order'], "all", False, True)
    elif m == 'combo':
        dn = S.compute_combo_from_variables(x, The, betac, muc, o['T_order'], o['beta_order'],
                                            "all", False, True)
    elif m == 'means':
        dn = S.compute_means_from_variables(x, The, betac, muc, "all", omega, sigma, kappa, False, False)
    elif m == 'precise':
        dn = S.compute_precise_from_variables(x, The, betac, muc, "all", False)
    elif m == '3D':
        dn = S.Integral3D.compute_from_variables(x, The, betac, muc, o['relative_accuracy'], "all", False)
    elif m == '5D':
        dn = S.Integral5D.compute_from_variables(x, The, betac, muc, o['relative_accuracy'], "full", False)
    else:
        raise ValueError("Unknown SZpack method %s" % m)
    return np.asarray(dn, dtype=np.float64)


def thermal_table(freqs, Te_nodes, method='combo', szpack_options=None, Tcmb=_TCMB):
    r"""Relativistic thermal SZ spectrum per unit Compton y, from SZpack.

    :math:`g(x, T_e) = \Delta n(x, T_e) / y`, with :math:`y = \tau\, kT_e / m_e c^2`, for gas without
    bulk velocity and an observer in the CMB frame.

    Parameters
    ----------
    freqs : array_like
        Frequencies in GHz.
    Te_nodes : array_like
        Electron temperatures in keV.
    method : str, optional
        SZpack method, see `SZpack_model`. Default 'combo'.
    szpack_options : dict, optional
        SZpack accuracy settings, see `SZpack_model`.
    Tcmb : float, optional
        CMB temperature in K. Default 2.7255.

    Returns
    -------
    ndarray
        Shape (len(Te_nodes), len(freqs)). For Te -> 0 it tends to
        :func:`pymsz.sz_spectra.tsz_nonrel`.
    """
    from pymsz.szpack import require
    S = require()
    m = _canonical_method(method)
    opts = _szpack_options(m, szpack_options)
    x = sz_spectra.freq_to_x(np.atleast_1d(freqs), Tcmb)
    _check_x(x, Tcmb)
    order = np.argsort(x)
    xs = [float(v) for v in x[order]]
    nodes = np.atleast_1d(np.asarray(Te_nodes, dtype=np.float64))
    out = np.zeros((nodes.size, x.size))
    for k, Te in enumerate(nodes):
        out[k, order] = szpack_dn(S, xs, Te, _BETA_TINY, 1., m, opts) / (Te / _ME_KEV)
    if not np.all(np.isfinite(out)):
        raise RuntimeError("SZpack returned non-finite values while tabulating the thermal spectrum")
    return out


class SZpack_model(object):
    r"""Thermal + kinetic SZ images with relativistic corrections, computed with SZpack v2.

    SZpack's analytic models or its numerical integration give the spectrum. By default it is summed
    exactly, particle by particle, along the line of sight. Needs the compiled SZpack extension.

    Parameters
    ----------
    simudata : load_data
        The simulation data (snapshot data only).
    freqs : float or array_like
        Observed frequencies in GHz. In terms of :math:`x = h\nu / k T_\mathrm{CMB}` the SZ spectrum
        does not depend on redshift, so these are observed frequencies. They must satisfy
        0.1 <= x <= 50, i.e. 5.7 GHz < nu < 2840 GHz. For a mock observation use
        ``freqs=telescope.frequencies()``.
    method : str, optional
        The SZpack model. Default 'combo'. Analytic models:

        - 'nonrel': non-relativistic tSZ + kSZ (for comparison);
        - 'asymptotic': asymptotic temperature expansion (best for Te < ~2 keV, valid to 20 keV);
        - 'CNSN': basis-function interpolation of Chluba et al. 2012 (2 keV < Te < 75 keV);
        - 'CNSN_opt': optimised CNSN basis (Chluba et al. 2013), see `szpack_options`;
        - 'combo': 'asymptotic' for Te < 2 keV and 'CNSN' above (Te < 75 keV). Recommended;
        - 'means': 'combo' plus the temperature and velocity dispersion along each line of sight
          (moment method, Chluba et al. 2013). Only with ``projection='pixel'``.

        Numerical integration:

        - '3D': numerical integration of the collision term (2 angles done analytically);
        - '5D': full 5D numerical integration (very slow, mostly for tests);
        - 'precise': 'combo' for Te < 75 keV and '3D' above.
    projection : {'particle', 'pixel'}, optional
        How the line-of-sight integration is done. Default 'particle'.

        - 'particle': the spectrum per unit optical depth is tabulated with the chosen method on a
          grid of electron temperatures, including its dependence on the line-of-sight and
          transverse velocities up to beta^2. It is evaluated for every gas particle, and the particle
          spectra are SPH-smoothed onto the image. This is the exact single-scattering sum along the
          line of sight, and it is fast also for the numerical methods.
        - 'pixel': the optical-depth-weighted Te and velocity (and, for 'means', their dispersions)
          are projected first; SZpack is then called for every pixel. Slow for '3D' and '5D'.
    kinetic : bool, optional
        Include the gas velocities (kinetic SZ and its relativistic corrections). Default True. If
        False, only the thermal SZ effect is computed.
    bulk_velocity : None, True or array_like, optional
        Velocity (km/s, simulation axes, before the rotation) added to all gas particles. `load_data`
        removes the bulk motion of the object from the velocities.

        - None (default): use the velocities as loaded, i.e. relative to the object, as `TK_model`;
        - True: add back the removed bulk velocity ``simudata.bulkvel`` (peculiar velocities);
        - ``[vx, vy, vz]``: add this velocity, e.g. the velocity of the halo relative to the CMB.

        The observer is at +infinity along the line of sight: a positive line-of-sight velocity is
        gas moving towards the observer (a kSZ increment).
    unit : str, optional
        Unit of the images. Default 'MJy/sr' (intensity change Delta I_nu). 'Jy/pixel': Delta I_nu
        times the pixel solid angle (needs redshift > 0). 'K', 'uK': linearised CMB thermodynamic
        temperature change. 'Dn': change of the photon occupation number.
    Tcmb : float, optional
        CMB temperature today (K), for x and the unit conversions. Default 2.7255.
    szpack_options : dict, optional
        SZpack accuracy settings. Default None: ``{'T_order': 10 ('asymptotic') or 20 (others),
        'beta_order': 2, 'relative_accuracy': 1e-4 ('3D', '5D', 'precise'), 'kmax': 5,
        'accuracy_level': 1 ('CNSN_opt')}``.
    Te_grid : array_like, optional
        Temperature nodes (keV) of the spectral table (``projection='particle'``). Default None: 64
        log-spaced nodes from 0.01 keV to the hottest gas, within the validity range of the method.
        Gas outside the nodes uses the spectrum of the closest node, per unit y (thermal) or per unit
        tau (kinetic); a warning is given if this happens above the upper limit.
    npixel, neighbours, axis, AR, SD, SP, Memreduce, Ncpu, Ntasks, redshift, zthick, sph_kernel
        The image parameters, as in `TT_model`.

    Attributes
    ----------
    freqs : ndarray
        The frequencies (GHz), in the input order.
    x : ndarray
        Dimensionless frequencies :math:`h\nu / k T_\mathrm{CMB}`.
    szdata : ndarray
        SZ images, shape (len(freqs), npixel, npixel), in `unit`.
    dndata : ndarray
        The same images as Delta n.
    tau : ndarray
        Thomson optical depth map.
    Te : ndarray
        Optical-depth-weighted electron temperature (keV).
    ydata : ndarray
        Compton-y map, :math:`\tau\, kT_e / m_e c^2`; identical to `TT_model.ydata`.
    bpar : ndarray
        Optical-depth-weighted line-of-sight velocity v/c (only if `kinetic`).
    table : dict
        The spectral table of ``projection='particle'``.

    Examples
    --------
    >>> simd = pymsz.load_data(...)
    >>> mm = pymsz.SZpack_model(simd, [90., 150., 220., 353.], method='combo', npixel=512, AR=5)
    >>> mm.szdata[0]   # 90 GHz image in MJy/sr
    >>> mm.write_fits_image("SZ-cube.fits")

    For a mock observation, compute the model at the band frequencies of the telescope:

    >>> tel = pymsz.Telescope.preset('SPT-3G')
    >>> mm = pymsz.SZpack_model(simd, tel.frequencies(), npixel=512, AR=5)
    >>> obs = tel.observe(mm)
    """

    def __init__(self, simudata, freqs, method='combo', projection='particle', kinetic=True,
                 bulk_velocity=None, unit='MJy/sr', Tcmb=2.7255, szpack_options=None, Te_grid=None,
                 npixel=500, neighbours=None, axis='z', AR=None, SD=2, SP=[194.95, 27.98],
                 Memreduce=False, Ncpu=None, Ntasks=None, redshift=None, zthick=None, sph_kernel='cubic'):
        from pymsz.szpack import require
        self._S = require()

        # ---- SZpack settings ----
        self.method = _canonical_method(method)
        self.kind, self.Te_min, self.Te_max = _METHODS[self.method]
        self.projection = str(projection).lower()
        if self.projection not in ('particle', 'pixel'):
            raise ValueError("projection must be 'particle' or 'pixel'")
        if self.method == 'means' and self.projection != 'pixel':
            raise ValueError("method='means' works on projected moments: use projection='pixel'")
        self.kinetic = kinetic
        self.bulk_velocity = bulk_velocity
        self.unit = sz_spectra.check_unit(unit)
        self.Tcmb = Tcmb
        self.opts = _szpack_options(self.method, szpack_options)
        self.Te_grid = Te_grid

        # ---- frequencies ----
        self.freqs = np.atleast_1d(np.asarray(freqs, dtype=np.float64))
        self.x = sz_spectra.freq_to_x(self.freqs, self.Tcmb)
        _check_x(self.x, self.Tcmb)

        # ---- image settings, same as TT_model ----
        if isinstance(npixel, str):
            self.npl = npixel.lower()
        else:
            self.npl = npixel
        self.ngb = neighbours
        self.ax = axis
        self.ar = AR
        if redshift is None:
            self.red = simudata.cosmology['z']
        else:
            self.red = redshift
        self.zthick = zthick
        self.pxs = None
        self.SD = SD
        self.Memreduce = Memreduce
        self.ncpu = Ncpu
        self.ntask = Ntasks
        self.sph_kn = sph_kernel
        self.sp = SP
        self.szdata = self.dndata = np.array([])
        self.tau = self.Te = self.ydata = self.bpar = np.array([])
        self._pix = None  # projected moments (projection='pixel'), used by sz_maps

        if self.ar is None and self.npl == 'auto':
            print("Do not accept AR == None and npixel=='AUTO' !! \n The npixel is reset to 500.!")
            self.npl = 500
        if self.SD not in [2, 3]:
            raise ValueError("smoothing dimension must be 2 or 3")

        if simudata.data_type == "snapshot":
            self._cal_snap(simudata)
        else:
            raise ValueError("SZpack_model only supports snapshot data (load_data(..., snapshot=True))"
                             ", not %s" % simudata.data_type)

    # ------------------------------------------------------------------------------------------
    # SZpack interface
    # ------------------------------------------------------------------------------------------
    def spectrum(self, Te, betac=0., muc=1., omega=0., sigma=0., kappa=0., x=None):
        r"""Delta n(x) per unit optical depth, with the method and settings of this model.

        For an observer at rest in the CMB frame.

        Parameters
        ----------
        Te : float
            Electron temperature in keV.
        betac : float, optional
            Cluster speed v/c. Default 0.
        muc : float, optional
            Direction cosine of the velocity with respect to the line of sight (> 0: towards the
            observer). Default 1.
        omega, sigma, kappa : float, optional
            Temperature and velocity moments, only used by ``method='means'``.
        x : array_like, optional
            Dimensionless frequencies. Default: `x` of the model.

        Returns
        -------
        ndarray
            Delta n / tau at `x`.
        """
        x = self.x if x is None else np.atleast_1d(x)
        order = np.argsort(x)  # SZpack expects an increasing x array
        dn = szpack_dn(self._S, [float(v) for v in x[order]], Te, betac, muc, self.method, self.opts,
                       omega, sigma, kappa)
        out = np.empty(x.size)
        out[order] = dn  # back to the input frequency order
        return out

    def _build_table(self, Te_lo, Te_hi):
        r"""Tabulate the spectrum per unit tau as
        Dn/tau = The * G0(Te) + beta_par * F1(Te) + beta_par^2 * F2(Te) + beta_perp^2 * F3(Te)
        on log-spaced Te nodes and return cubic-spline interpolators in log(Te)."""
        from scipy.interpolate import CubicSpline
        if self.Te_grid is not None:
            nodes = np.unique(np.asarray(self.Te_grid, dtype=np.float64))
        else:
            Te_hi = max(Te_hi, Te_lo * 1.1)
            nodes = np.geomspace(Te_lo, Te_hi, 64)
        b = _BETA_REF
        nn, nx = nodes.size, self.x.size
        G0 = np.zeros((nn, nx))
        F1, F2, F3 = np.zeros((nn, nx)), np.zeros((nn, nx)), np.zeros((nn, nx))
        print("SZpack_model: tabulating the '%s' spectrum at %d temperatures (%.3g - %.3g keV)..."
              % (self.method, nn, nodes[0], nodes[-1]))
        for k, Te in enumerate(nodes):
            s0 = self.spectrum(Te, _BETA_TINY, 1.)
            G0[k] = s0 / (Te / _ME_KEV)
            if self.kinetic:
                sp, sm, s90 = self.spectrum(Te, b, 1.), self.spectrum(Te, b, -1.), self.spectrum(Te, b, 0.)
                F1[k] = (sp - sm) / (2. * b)
                F2[k] = (0.5 * (sp + sm) - s0) / b**2
                F3[k] = (s90 - s0) / b**2
        if not np.all(np.isfinite(G0)) or not (np.all(np.isfinite(F1)) and np.all(np.isfinite(F2))):
            raise RuntimeError("SZpack returned non-finite values while tabulating the spectrum")
        lnT = np.log(nodes)
        if nn == 1:
            spl = [lambda t, a=a: np.broadcast_to(a[0], (np.size(t), nx)) for a in (G0, F1, F2, F3)]
        else:
            spl = [CubicSpline(lnT, a, axis=0) for a in (G0, F1, F2, F3)]
        self.table = {'Te': nodes, 'G0': G0, 'F1': F1, 'F2': F2, 'F3': F3}
        return nodes, spl

    # ------------------------------------------------------------------------------------------
    # image geometry, identical to TT_model
    # ------------------------------------------------------------------------------------------
    def _geometry(self, simd):
        a2p = 1. / simd.cosmology['h'] / (1 + simd.cosmology['z'])  # to physical kpc
        if self.red <= 0. and self.npl == 'auto':
            print("Do not accept redshift == 0 and npixel=='AUTO' !!\n The npixel is reset to 500.!")
            self.npl = 500

        self.cc = simd.center * a2p
        self.rr = simd.radius * a2p
        pos = rotate_data(simd.pos * a2p, self.ax)[0]
        keep = np.ones(pos.shape[0], dtype=bool)
        if self.zthick is not None:
            self.zthick = self.zthick * a2p
            keep &= (pos[:, 2] > -self.zthick) & (pos[:, 2] < self.zthick)

        if isinstance(simd.hsml, type(0)) or simd.hsml is None:
            hsml = None
        else:
            hsml = simd.hsml * a2p

        if self.npl != 'auto':
            pk = pos[keep]
            self.pxs = np.min([pk[:, 0].max() - pk[:, 0].min(), pk[:, 1].max() - pk[:, 1].min()]) / self.npl
        if self.red > 0.:
            if simd.cosmology['omega_matter'] != 0:
                cosmo = FlatLambdaCDM(H0=simd.cosmology['h'] * 100, Om0=simd.cosmology['omega_matter'])
            else:
                print('No cosmology loaded, assume WMAP7!')
                cosmo = WMAP7
            if self.ar is not None:
                self.pxs = self.ar / cosmo.arcsec_per_kpc_proper(self.red).value  # in kpc physical
                if self.npl == 'auto':
                    self.npl = np.int32(self.rr * 2 / self.pxs) + 1
            else:
                self.ar = self.pxs * cosmo.arcsec_per_kpc_proper(self.red).value
        else:
            if self.ar is None:
                self.ar = 1.

        # cut out unused data
        if self.npl != 'auto':
            keep &= (pos[:, 0] >= -self.npl * self.pxs / 2.) & (pos[:, 0] <= self.npl * self.pxs / 2.) &\
                (pos[:, 1] >= -self.npl * self.pxs / 2.) & (pos[:, 1] <= self.npl * self.pxs / 2.)
            if self.npl != np.int32(self.rr * 2 / self.pxs) + 1:
                self.rr = self.pxs * (self.npl - 1) / 2.

        if self.ngb is not None:
            tempr = np.sqrt(np.sum((simd.pos - simd.pos[simd.rho.argmin()])**2, axis=1))
            tempr.sort()
            ngbinp = tempr[self.ngb + 1] * a2p / self.pxs
            print('Convert the neighbor count in pixel size: ', ngbinp)
        else:
            ngbinp = self.ngb
        if hsml is None and ngbinp is None:
            ngbinp = 27  # as TK_model: no smoothing lengths in the data
        return pos[keep], (None if hsml is None else hsml[keep]), keep, ngbinp

    def _smooth(self, fields, pos, hsml, ngb):
        """SPH-smooth a dict of per-particle quantities onto the image and divide by the pixel area
        (self.pxs**2, physical kpc^2). E.g. the electron cross-section sigma_T*N_e in kpc^2 -> tau."""
        out = {}
        keys = list(fields.keys())
        for i in range(0, len(keys), 20):  # SPH_smoothing accepts at most 20 fields at once
            sub = {k: np.ascontiguousarray(fields[k], dtype=np.float64) for k in keys[i:i + 20]}
            p = pos[:, :2] if self.SD == 2 else pos
            res = SPH_smoothing(sub, p, self.pxs, ngb, hsml=hsml, pxln=self.npl, Memreduce=self.Memreduce,
                                Ncpu=self.ncpu, Ntasks=self.ntask, kernel_name=self.sph_kn)
            for k in sub:
                d = res[k] if self.SD == 2 else np.sum(res[k], axis=2)
                out[k] = np.asarray(d, dtype=np.float64).T / self.pxs**2
        return out

    # ------------------------------------------------------------------------------------------
    def _cal_snap(self, simd):
        simd.prep_ss_TT()  # electron_number per particle
        pos, hsml, keep, ngb = self._geometry(simd)

        # per-particle total electron cross-section sigma_T * N_e, converted from cm^2 to kpc^2
        # (/ _KPC**2 is only a unit change). _smooth divides by the pixel area pxs**2 (kpc^2) -> tau.
        wtau = _SIGMA_T * simd.electron_number[keep] / _KPC**2
        Te = np.maximum(np.asarray(simd.temp[keep], dtype=np.float64) * _KB_KEV, 1.0e-8)
        if self.kinetic:
            vel = np.asarray(simd.vel, dtype=np.float64)
            if self.bulk_velocity is True:
                vel = vel + simd.bulkvel
            elif self.bulk_velocity is not None and self.bulk_velocity is not False:
                vel = vel + np.asarray(self.bulk_velocity, dtype=np.float64)
            vel = vel[keep]
            vlos = rotate_data(vel, self.ax)[0][:, 2]  # rotate as positions: full rotated vector
            v2 = np.sum(vel**2, axis=1)
        else:
            vlos = v2 = np.zeros_like(wtau)

        if self.projection == 'particle':
            self._project_particles(pos, hsml, ngb, wtau, Te, vlos, v2)
        else:
            self._project_pixels(pos, hsml, ngb, wtau, Te, vlos, v2)

        # unit conversion of Delta n (kept in self.dndata)
        self.dndata = self.szdata
        self.szdata = self._convert(self.dndata)

    def _project_particles(self, pos, hsml, ngb, wtau, Te, vlos, v2):
        hot = wtau > 0
        lo = max(_TE_TABLE_MIN, self.Te_min)
        hi = min(Te[hot].max() if hot.any() else lo, self.Te_max)
        if hi < lo:
            hi = lo
        nodes, (G0, F1, F2, F3) = self._build_table(lo, hi)
        nabove = np.sum(wtau[Te > nodes[-1] * 1.0001])
        if nabove > 0 and nodes[-1] < Te.max():
            warnings.warn("%.2g%% of the optical depth is in gas hotter than %.3g keV (max %.3g keV), the "
                          "limit of the '%s' tabulation; the spectrum shape at %.3g keV is used for it."
                          % (100 * nabove / wtau.sum(), nodes[-1], Te.max(), self.method, nodes[-1]))
        if self.method == 'CNSN':
            nbelow = np.sum(wtau[(Te < nodes[0]) & (Te > _TE_TABLE_MIN)])
            if nbelow > 0:
                warnings.warn("'CNSN' is only valid for Te > 2 keV; colder gas (%.2g%% of tau) uses the 2 keV "
                              "spectrum per unit y. Use method='combo' to avoid this." % (100 * nbelow / wtau.sum()))
        lnT = np.log(np.clip(Te, nodes[0], nodes[-1]))
        bpar = vlos / _C_KMS
        bperp2 = np.maximum(v2 / _C_KMS**2 - bpar**2, 0.)
        dn = (Te / _ME_KEV)[:, None] * G0(lnT)
        if self.kinetic:
            dn += bpar[:, None] * F1(lnT) + (bpar**2)[:, None] * F2(lnT) + bperp2[:, None] * F3(lnT)
        dn *= wtau[:, None]

        fields = {'tau': wtau, 'tauTe': wtau * Te}
        if self.kinetic:
            fields['taubpar'] = wtau * bpar
        for k in range(self.x.size):
            fields['dn%d' % k] = dn[:, k]
        del dn
        maps = self._smooth(fields, pos, hsml, ngb)
        self._set_aux_maps(maps)
        self.szdata = np.array([maps['dn%d' % k] for k in range(self.x.size)])

    def _project_pixels(self, pos, hsml, ngb, wtau, Te, vlos, v2):
        bpar = vlos / _C_KMS
        fields = {'tau': wtau, 'tauTe': wtau * Te}
        if self.kinetic:
            fields.update({'taubpar': wtau * bpar, 'taub2': wtau * v2 / _C_KMS**2, 'taubpar2': wtau * bpar**2})
        if self.method == 'means':
            fields.update({'tauTe2': wtau * Te**2, 'tauTebpar': wtau * Te * bpar})
        maps = self._smooth(fields, pos, hsml, ngb)
        self._set_aux_maps(maps)

        tau = maps['tau']
        good = tau > 0
        t = np.where(good, tau, 1.)
        mTe = maps['tauTe'] / t
        if self.kinetic:
            mbpar = maps['taubpar'] / t
            bperp2 = np.maximum((maps['taub2'] - maps['taubpar2']) / t, 0.)
        else:
            mbpar = bperp2 = np.zeros_like(tau)
        if self.method == 'means':
            mT = np.where(mTe > 0, mTe, 1.)
            omega = np.maximum(maps['tauTe2'] / t / mT**2 - 1., 0.)
            sigma = (maps['tauTebpar'] / t - mTe * mbpar) / mT if self.kinetic else np.zeros_like(tau)
            kappa = np.maximum(maps['taubpar2'] / t - mbpar**2, 0.) if self.kinetic else np.zeros_like(tau)

        Tlo, Thi = max(self.Te_min, 1.0e-6), self.Te_max
        nclip = np.sum(good & ((mTe < Tlo) | (mTe > Thi)))
        if nclip:
            warnings.warn("%d pixels have Te outside the validity range of '%s' (%.3g - %.3g keV) and are "
                          "computed at the closest limit." % (nclip, self.method, Tlo, Thi))
        betac = np.sqrt(mbpar**2 + bperp2)
        if np.any(betac > 0.1):
            warnings.warn("%d pixels have v > 0.1 c, clipped to 0.1 c (SZpack limit)" % np.sum(betac > 0.1))
        self._pix = {'good': good, 'tau': tau, 'Te': np.clip(mTe, Tlo, Thi), 'betac': np.minimum(betac, 0.1),
                     'mu': np.where(betac > 0, mbpar / np.where(betac > 0, betac, 1.), 1.)}
        if self.method == 'means':
            self._pix.update({'omega': omega, 'sigma': sigma, 'kappa': kappa})
        self.szdata = self._pixel_dn(self.x)

    def _pixel_dn(self, x):
        """Delta n maps at x from the projected moments, calling SZpack for every pixel."""
        p = self._pix
        if self.kind == 'numerical':
            print("SZpack_model: calling the '%s' numerical integration for %d pixels; this can take long. "
                  "projection='particle' is much faster." % (self.method, p['good'].sum()))
        out = np.zeros((np.size(x),) + p['tau'].shape)
        for i, j in zip(*np.nonzero(p['good'])):
            if self.method == 'means':
                s = self.spectrum(p['Te'][i, j], p['betac'][i, j], p['mu'][i, j], p['omega'][i, j],
                                  p['sigma'][i, j], p['kappa'][i, j], x=x)
            else:
                s = self.spectrum(p['Te'][i, j], p['betac'][i, j], p['mu'][i, j], x=x)
            out[:, i, j] = p['tau'][i, j] * s
        return out

    def sz_maps(self, freqs, unit='Dn'):
        r"""SZ images at the frequencies `freqs` (GHz) in `unit`, shape (len(freqs), npixel, npixel).

        This is the interface used by :meth:`pymsz.Telescope.observe`. Frequencies computed at
        construction are returned exactly. For other frequencies, with ``projection='pixel'`` SZpack
        is called again for every pixel (exact); with ``projection='particle'`` the spectrum is
        interpolated (cubic spline in x) between the computed frequencies, which must bracket them.
        For exact band averages with ``projection='particle'``, construct the model with
        ``freqs=telescope.frequencies()``."""
        freqs = np.atleast_1d(np.asarray(freqs, dtype=np.float64))
        known = np.array([np.any(np.isclose(self.freqs, f, rtol=1e-7, atol=0.)) for f in freqs])
        if known.all() or self.projection == 'particle':
            dn = sz_spectra.interpolate_dn(self.dndata, self.freqs, freqs, self.Tcmb,
                                           what="SZpack_model images (projection='particle')")
        else:
            x = sz_spectra.freq_to_x(freqs, self.Tcmb)
            _check_x(x, self.Tcmb)
            dn = np.empty((freqs.size,) + self.dndata.shape[1:])
            if known.any():
                dn[known] = sz_spectra.interpolate_dn(self.dndata, self.freqs, freqs[known], self.Tcmb)
            dn[~known] = self._pixel_dn(x[~known])
        return sz_spectra.dn_to_unit(dn, freqs, unit, self.Tcmb, self.ar)

    def _set_aux_maps(self, maps):
        self.tau = maps['tau']
        t = np.where(self.tau > 0, self.tau, 1.)
        self.Te = np.where(self.tau > 0, maps['tauTe'] / t, 0.)
        self.ydata = maps['tauTe'] / _ME_KEV
        if self.kinetic:
            self.bpar = np.where(self.tau > 0, maps['taubpar'] / t, 0.)

    def _convert(self, dn):
        if self.unit == 'Jy/pixel' and self.red <= 0:
            warnings.warn("unit='Jy/pixel' needs redshift > 0; the pixel solid angle uses AR = %g arcsec" % self.ar)
        return sz_spectra.dn_to_unit(dn, self.freqs, self.unit, self.Tcmb, self.ar)

    # ------------------------------------------------------------------------------------------
    def write_fits_image(self, fname, comments="None", overwrite=False):
        r"""Write the SZ image cube and the tau, Te, y and beta_par maps to a FITS file.

        The images are the primary HDU, shape [nfreq, npixel, npixel], with the frequencies in the
        FREQn keywords. The maps are the extensions TAU, TE (keV), YSZ and BPAR (if `kinetic`).

        Parameters
        ----------
        fname : str
            File name; '.fits' is added if missing.
        comments : str or list of str, optional
            Comments for the primary header. Default 'None'.
        overwrite : bool, optional
            Replace an existing file. Default False.
        """
        aux = [('TAU', self.tau, 'tau', 'Thomson optical depth'),
               ('TE', self.Te, 'keV', 'optical depth weighted electron temperature'),
               ('YSZ', self.ydata, 'y', 'Compton y parameter')]
        if self.kinetic:
            aux.append(('BPAR', self.bpar, 'v/c', 'tau-weighted v_los/c, >0 towards observer'))
        extra = [("SZMETHOD", self.method, 'SZpack method'),
                 ("SZPROJ", self.projection, 'line of sight integration: particle or pixel'),
                 ("KINETIC", bool(self.kinetic), 'kinematic SZ included')]
        sz_spectra.write_sz_cube(self, fname, aux=aux, extra=extra, comments=comments, overwrite=overwrite,
                                 origin='Software: PymSZ + SZpack')
