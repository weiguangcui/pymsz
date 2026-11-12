import warnings
import numpy as np
from pymsz.rotate_data import rotate_data, SPH_smoothing
from astropy.cosmology import FlatLambdaCDM, WMAP7
from astropy.coordinates import SkyCoord
from astropy.time import Time
from pymsz import version
from pymsz import sz_spectra
from pymsz.sz_spectra import TCMB, KB_KEV
# scipy must >= 0.17 to properly use this!
# from scipy.stats import binned_statistic_2d


class TT_model(object):
    r"""Compton-y map of the thermal SZ effect (and the y-weighted electron temperature).

    Every gas particle contributes :math:`\sigma_T N_e k T_e / m_e c^2`, with :math:`N_e` its
    number of electrons. The contributions are SPH-smoothed onto the image and divided by the pixel
    area (physical kpc^2), which gives the dimensionless Compton y.

    Parameters
    ----------
    simudata : load_data
        The simulation data.
    label : {'tSZ', 'ne'}, optional
        'tSZ' (default): the Compton-y map. 'ne': the electron column density (electrons per
        physical kpc^2) in `ydata` instead.
    npixel : int or 'auto', optional
        Pixels per side of the (square) image. Default 500. 'auto': enough pixels to cover the loaded
        region (``2 radius``) at the pixel size `AR`; this needs `AR` and ``redshift > 0``,
        otherwise npixel is reset to 500.
    neighbours : int, optional
        Smooth every particle over this many neighbours, ignoring the smoothing lengths of the
        snapshot (snapshot data only). Default None: use the smoothing lengths.
    axis : str, list or ndarray, optional
        Line of sight: 'x', 'y', 'z' (default), the angles ``[alpha, beta, gamma]`` (degrees) of the
        rotations around the x, y and z axes, or a 3x3 rotation matrix. See
        :func:`pymsz.rotate_data.rotate_data`.
    AR : float, optional
        Pixel size (angular resolution) in arcsec. With `redshift` it sets the physical pixel size.
        Default None: the region is projected onto `npixel` pixels and AR is computed from the
        resulting pixel size. At redshift 0, AR is set to a nominal 1 arcsec.
    SD : {2, 3}, optional
        Dimensions of the SPH smoothing. Default 2 (on the image plane). 3: on a 3D grid, then summed
        along the line of sight (slow, needs much memory).
    SP : list of 2 floats, optional
        Sky position ``[RA, Dec]`` (degrees) of the image centre, for the FITS WCS. Default
        ``[194.95, 27.98]`` (Coma).
    Memreduce : bool, optional
        Reduce the memory of the parallel smoothing by passing the cKDTree to the workers instead of
        the neighbour lists. Default False.
    Ncpu : int, optional
        Number of processes for the SPH smoothing. Default None: all CPUs.
    Ntasks : int, optional
        Number of tasks the smoothing is split into. Default None: `Ncpu`. Increase it if you get
        ``IO Error: bad message length``.
    redshift : float, optional
        Redshift at which the object is observed. Default None: the snapshot redshift. Place the
        object at redshift > 0: at redshift 0 the angular units are not defined. The physical
        positions of the particles are always those at the snapshot time.
    zthick : float, optional
        Use only the gas within +-zthick (comoving kpc/h) of the centre along the line of sight.
        Default None: all loaded gas.
    sph_kernel : str, optional
        SPH kernel: 'cubic' (default), 'quartic', 'quintic', 'wendland2', 'wendland4' or
        'wendland6'.
    yweighted_Te : bool, optional
        Also project the y-weighted electron temperature `Tydata`, which ``to_sz(...,
        relativistic=True)`` uses. Default True (only for ``label='tSZ'`` and snapshot data).

    Attributes
    ----------
    ydata : ndarray
        The Compton-y map (or the electron column density for ``label='ne'``), shape
        (npixel, npixel), indexed [y, x].
    Tydata : ndarray or None
        y-weighted electron temperature (keV), if `yweighted_Te`.
    pxs : float
        Pixel size in physical kpc (at the snapshot redshift).
    ar : float
        Pixel size in arcsec.
    npl : int
        Number of pixels per side.
    red : float
        Redshift of the observation.
    szdata, freqs, unit
        The SZ images computed by :meth:`to_sz`, their frequencies (GHz) and unit.

    See Also
    --------
    TK_model : the kinetic SZ map.
    SZpack_model : relativistic thermal and kinetic SZ, summed particle by particle.
    pymsz.Telescope : mock observations.

    Notes
    -----
    The pixel size is set by `AR` and `redshift` if `AR` is given, else by the size of the region
    divided by `npixel`. Distances use a flat LCDM cosmology with the h and Omega_m of the snapshot
    (WMAP7 if there is none). ``npixel='auto'`` with ``AR=None`` or at redshift 0 is not possible:
    npixel is then reset to 500.

    Examples
    --------
    >>> tt = pymsz.TT_model(simd, npixel=512, AR=5, redshift=0.1)
    >>> tt.ydata                                  # Compton-y map
    >>> tt.to_sz([90., 150., 220.], unit='uK')    # thermal SZ images, non-relativistic
    >>> tt.to_sz([90., 150., 220.], relativistic=True, method='combo')   # with SZpack corrections
    """

    def __init__(self, simudata, label='tSZ', npixel=500, neighbours=None, axis='z', AR=None, SD=2, SP=[194.95, 27.98],
                Memreduce=False, Ncpu=None, Ntasks=None, redshift=None, zthick=None, sph_kernel='cubic',
                yweighted_Te=True):
        if isinstance(npixel, type("")) or isinstance(npixel, type('')):
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
        # self.periodic = periodic
        self.Memreduce=Memreduce
        self.ncpu = Ncpu
        self.ntask = Ntasks
        self.ydata = np.array([])
        self.sph_kn = sph_kernel
        # self.ad = 1.  # angular diameter distance normalized to 1 kpc

        self.sp = SP
        self.label = label
        self.yweighted_Te = yweighted_Te and label == 'tSZ'
        self.Tydata = None
        # frequency maps, filled by to_sz
        self.Tcmb = TCMB
        self.freqs = np.array([])
        self.unit = None
        self.szdata = np.array([])
        self.sz_settings = {'relativistic': False, 'Te': None, 'method': 'combo', 'szpack_options': None}

        if self.ar is None and self.npl == 'auto':
            print("Do not accept AR == None and npixel=='AUTO' !! \n The npixel is reset to 500.!")
            self.npl = 500
        if self.SD not in [2, 3]:
            raise ValueError("smoothing dimension must be 2 or 3" % SD)

        if simudata.data_type == "snapshot":
            self._cal_snap(simudata, label=label)
        elif simudata.data_type == "yt_data":
            self._cal_yt(simudata)
        else:
            raise ValueError("Do not accept this data type %s"
                             "Please try to use load_data to get the data" % simudata.data_type)

    # def TH_ymap(simd, npixel=500, neighbours=None, axis='z', AR=None, redshift=None):

    def _cal_snap(self, simd, label='tSZ'):
        # Kpc = 3.0856775809623245e+21  # cm
        simd.prep_ss_TT()

        if self.red <= 0. and self.npl == 'auto':
            print("Do not accept redshift == 0 and npixel=='AUTO' !!\n The npixel is reset to 500.!")
            self.npl = 500

        self.cc = simd.center/simd.cosmology['h']/(1+simd.cosmology['z'])
        self.rr = simd.radius/simd.cosmology['h']/(1+simd.cosmology['z'])
        pos = rotate_data(simd.pos/simd.cosmology['h']/(1+simd.cosmology['z']), self.ax)[0]  # to proper distance
        # electron temperature in keV, for the y-weighted temperature map
        Tkev = np.asarray(simd.temp, dtype=np.float64) * KB_KEV if self.yweighted_Te else None
        if self.zthick is not None:
            self.zthick = self.zthick/simd.cosmology['h']/(1+simd.cosmology['z'])
            idc = (pos[:, 2] > -self.zthick) & (pos[:, 2] < self.zthick)
            pos = pos[idc]
            if label=='tSZ':
                Tszdata = simd.Tszdata[idc]
            elif label=='ne':
                Tszdata = simd.electron_number[idc] # hacky way to get options for electron numdens or temp
            if Tkev is not None:
                Tkev = Tkev[idc]
        else:
            if label=='tSZ':
                Tszdata = np.copy(simd.Tszdata)
            elif label=='ne':
                Tszdata = np.copy(simd.electron_number)


        if isinstance(simd.hsml, type(0)):
            # self.ngb = 64
            hsml = None
        else:                      # with hsml
            # if self.ngb is None:               # estimate neighbors
            #     self.ngb=0
            #     ids = np.sum((simd.pos - simd.pos[simd.hsml.argmin()])**2, axis=1) <= simd.hsml.min()**2
            #     if simd.hsml[ids].size > self.ngb:
            #         self.ngb = simd.hsml[ids].size
            #     ids = np.sum((simd.pos - simd.pos[simd.hsml.argmax()])**2, axis=1) <= simd.hsml.max()**2
            #     if simd.hsml[ids].size > self.ngb:
            #         self.ngb = simd.hsml[ids].size
            #     self.ngb = np.int32(self.ngb*1.05)+1 #to be safe
            #     print('self.ngb = ', self.ngb)

            if self.zthick is not None:
                hsml = simd.hsml[idc]
            else:
                hsml = np.copy(simd.hsml)
            hsml = hsml/simd.cosmology['h']/(1+simd.cosmology['z'])
            # else:               # have set the neighbors, need to ignore the hsml
            #     hsml = None

        if self.npl != 'auto':
            minx = pos[:, 0].min()
            maxx = pos[:, 0].max()
            miny = pos[:, 1].min()
            maxy = pos[:, 1].max()
            # if self.SD == 3:
            #     minz = pos[:, 2].min()
            #     maxz = pos[:, 2].max()
            #     self.pxs = np.min([maxx - minx, maxy - miny, maxz - minz]) / self.npl
            # else:
            self.pxs = np.min([maxx-minx, maxy-miny]) / self.npl  # only for projected plane
        if self.red > 0.:
            if simd.cosmology['omega_matter'] != 0:
                cosmo = FlatLambdaCDM(H0=simd.cosmology['h'] * 100,
                                      Om0=simd.cosmology['omega_matter'])
            else:
                print('No cosmology loaded, assume WMAP7!')
                cosmo = WMAP7

            if self.ar is not None:
                self.pxs = self.ar/cosmo.arcsec_per_kpc_proper(self.red).value  # in kpc physical
                if self.npl == 'auto':
                    self.npl = np.int32(self.rr*2/self.pxs)+1
            else:
                self.ar = self.pxs * cosmo.arcsec_per_kpc_proper(self.red).value
        else:
            if self.ar is None:
                self.ar = 1.

        # cut out unused data
        if self.npl != 'auto':
            idc = (pos[:, 0] >= -self.npl*self.pxs/2.) & (pos[:, 0] <= self.npl*self.pxs/2.) &\
                (pos[:, 1] >= -self.npl*self.pxs/2.) & (pos[:, 1] <= self.npl*self.pxs/2.)
            pos = pos[idc]
            Tszdata = Tszdata[idc]
            if Tkev is not None:
                Tkev = Tkev[idc]
            if hsml is not None:
                hsml = hsml[idc]
            if self.npl != np.int32(self.rr*2/self.pxs)+1:
                self.rr = self.pxs * (self.npl-1) / 2.

        # Tszdata /= (self.pxs * Kpc / simd.cosmology["h"])**2
        if self.ngb is not None:
            # need to change into pixel scale roughly calculate the maxi distance
            tempr=np.sqrt(np.sum((simd.pos - simd.pos[simd.rho.argmin()])**2, axis=1))
            tempr.sort()
            ngbinp=tempr[self.ngb+1]/simd.cosmology['h']/(1+simd.cosmology['z'])/self.pxs
            print('Convert the neighbor count in pixel size: ', ngbinp)
        else:
            ngbinp=self.ngb

        # with yweighted_Te, y and y*Te are smoothed together (same kernel weights)
        wdata = Tszdata if Tkev is None else {'y': Tszdata, 'yT': Tszdata * Tkev}
        if self.SD == 2:
            res = SPH_smoothing(wdata, pos[:, :2], self.pxs, ngbinp, hsml=hsml,
                                pxln=self.npl, Memreduce=self.Memreduce, Ncpu=self.ncpu,
                                Ntasks=self.ntask, kernel_name=self.sph_kn)
        else:
            # be ware that zthick could cause some problems if it is larger than pxs*npl!!
            # This has been taken in care in the rotate_data function.
            res = SPH_smoothing(wdata, pos, self.pxs, ngbinp, hsml=hsml,
                                pxln=self.npl, Memreduce=self.Memreduce, Ncpu=self.ncpu,
                                Ntasks=self.ntask, kernel_name=self.sph_kn)
            if isinstance(res, dict):
                res = {k: np.sum(v, axis=2) for k, v in res.items()}
            else:
                res = np.sum(res, axis=2)
        if isinstance(res, dict):
            self.ydata = res['y'].T / self.pxs**2
            yT = res['yT'].T / self.pxs**2
            self.Tydata = np.where(self.ydata > 0, yT / np.where(self.ydata > 0, self.ydata, 1.), 0.)
        else:
            self.ydata = res.T / self.pxs**2

    def _cal_yt(self, simd):  # Not work properly right now
        # from yt.units import cm
        Ptype = simd.prep_yt_TT()
        if self.red is None:
            self.red = simd.yt_ds.current_redshift
        if self.ar is None:
            rr = 2. * simd.radius
        else:
            if self.red <= 0.0:
                self.red = 0.05

            if simd.yt_ds.omega_matter != 0:
                cosmo = FlatLambdaCDM(H0=simd.yt_ds.hubble_constant * 100,
                                      Om0=simd.yt_ds.omega_matter)
            else:
                cosmo = WMAP7
            self.pxs = cosmo.arcsec_per_kpc_proper(self.red) * self.ar * simd.yt_ds.hubble_constant
            rr = self.npl * self.pxs
        if isinstance(self.ax, type('x')):
            projection = simd.yt_ds.proj(('deposit', Ptype + '_smoothed_Tsz'), self.ax,
                                         center=simd.center, data_source=simd.yt_sp)
            FRB = projection.to_frb(rr, self.npl)
            self.ydata = FRB[('deposit', Ptype + '_smoothed_Tsz')]

    # ------------------------------------------------------------------------------------------
    # y-map -> SZ images at given frequencies
    # ------------------------------------------------------------------------------------------
    def to_sz(self, freqs, unit='MJy/sr', relativistic=False, Te=None, method='combo',
              szpack_options=None, Tcmb=TCMB):
        r"""Thermal SZ images at the observed frequencies `freqs`, from the Compton-y map.

        The images have the same form as those of `SZpack_model`. The settings are kept, so that
        :meth:`sz_maps` (used by :meth:`pymsz.Telescope.observe`) gives the same spectrum.

        Parameters
        ----------
        freqs : float or array_like
            Observed frequencies in GHz.
        unit : str, optional
            'MJy/sr' (Delta I_nu, default), 'Jy/pixel', 'K' or 'uK' (Delta T_CMB, linearised
            thermodynamic temperature), or 'Dn' (photon occupation number Delta n).
        relativistic : bool, optional
            False (default): the non-relativistic spectrum, :math:`\Delta n = y\,G(x)\,(x
            \coth(x/2) - 4)`. True: the relativistic thermal SZ spectrum per unit y from SZpack
            (`method`), evaluated in every pixel at the y-weighted electron temperature,
            :math:`\Delta n = y\,g(x, T_y)`. This is correct to first order in the temperature
            spread along the line of sight. For the exact line-of-sight sum (and relativistic kSZ)
            use `SZpack_model`.
        Te : float or ndarray, optional
            Temperature (keV) for ``relativistic=True``. Default None: the y-weighted temperature map
            `Tydata`. A number (isothermal) or a map of shape (npixel, npixel).
        method : str, optional
            SZpack method for ``relativistic=True``: 'combo' (default), 'asymptotic', 'CNSN',
            'CNSN_opt', '3D', ... (see `SZpack_model`). Needs the compiled SZpack extension.
        szpack_options : dict, optional
            SZpack accuracy settings, see `SZpack_model`.
        Tcmb : float, optional
            CMB temperature in K. Default 2.7255.

        Returns
        -------
        ndarray
            The SZ images, shape (len(freqs), npixel, npixel). They are also kept in `szdata`, with
            `freqs` and `unit`.
        """
        if self.label != 'tSZ':
            raise ValueError("to_sz needs a Compton-y map (label='tSZ'), not label='%s'" % self.label)
        self.unit = sz_spectra.check_unit(unit)
        self.Tcmb = Tcmb
        self.sz_settings = {'relativistic': bool(relativistic), 'Te': Te, 'method': method,
                            'szpack_options': szpack_options}
        self.freqs = np.atleast_1d(np.asarray(freqs, dtype=np.float64))
        self.x = sz_spectra.freq_to_x(self.freqs, self.Tcmb)
        self.szdata = self.sz_maps(self.freqs, unit)
        return self.szdata

    def sz_maps(self, freqs, unit='Dn'):
        r"""Thermal SZ images at `freqs` (GHz) in `unit`, shape (len(freqs), npixel, npixel).

        Uses the spectrum chosen in :meth:`to_sz` (non-relativistic by default). This is the interface
        used by :meth:`pymsz.Telescope.observe`."""
        if self.label != 'tSZ':
            raise ValueError("sz_maps needs a Compton-y map (label='tSZ'), not label='%s'" % self.label)
        freqs = np.atleast_1d(np.asarray(freqs, dtype=np.float64))
        if self.sz_settings['relativistic']:
            dn = self._rsz_dn(freqs)
        else:
            x = sz_spectra.freq_to_x(freqs, self.Tcmb)
            dn = sz_spectra.tsz_nonrel(x)[:, None, None] * np.asarray(self.ydata)[None]
        return sz_spectra.dn_to_unit(dn, freqs, unit, self.Tcmb, self.ar)

    def _rsz_dn(self, freqs, nnodes=48):
        """Relativistic Delta n = y g(x, T) per pixel, g tabulated with SZpack on log T nodes."""
        from scipy.interpolate import CubicSpline
        from pymsz.SZpack_models import thermal_table, _METHODS, _canonical_method, _TE_TABLE_MIN
        s = self.sz_settings
        y = np.asarray(self.ydata, dtype=np.float64)
        Te = s['Te']
        if Te is None:
            if self.Tydata is None:
                raise ValueError("No y-weighted temperature map: construct TT_model with yweighted_Te=True "
                                 "(snapshot data) or give Te (keV) to to_sz.")
            Te = self.Tydata
        Te = np.broadcast_to(np.asarray(Te, dtype=np.float64), y.shape)
        m = _canonical_method(s['method'])
        tmin, tmax = _METHODS[m][1:]
        lo = max(_TE_TABLE_MIN, tmin)
        good = (y != 0) & (Te > 0)
        Tmax = Te[good].max() if good.any() else lo
        hi = max(min(Tmax, tmax), lo * 1.1)
        if Tmax > tmax:
            frac = np.sum(np.abs(y[good & (Te > tmax)])) / np.sum(np.abs(y[good]))
            warnings.warn("%.2g%% of y is in pixels with T_y > %.3g keV, the limit of SZpack method '%s'; the "
                          "spectrum at %.3g keV is used there." % (100 * frac, tmax, m, tmax))
        if tmin > _TE_TABLE_MIN and np.any(good & (Te < tmin)):
            frac = np.sum(np.abs(y[good & (Te < tmin)])) / np.sum(np.abs(y[good]))
            warnings.warn("'%s' is only valid for Te > %g keV; %.2g%% of y is colder and uses the %g keV "
                          "spectrum. Use method='combo' to avoid this." % (m, tmin, 100 * frac, tmin))
        nodes = np.geomspace(lo, hi, nnodes)
        table = thermal_table(freqs, nodes, m, s['szpack_options'], self.Tcmb)  # (nnodes, nfreq)
        g = CubicSpline(np.log(nodes), table, axis=0)(np.log(np.clip(Te, lo, hi)))  # (..., nfreq)
        return np.moveaxis(g, -1, 0) * y[None]

    def write_sz_fits(self, fname, comments="None", overwrite=False):
        r"""Write the SZ images of :meth:`to_sz` and the y map to a FITS file.

        The images are the primary HDU, shape [nfreq, npixel, npixel], with the frequencies in the
        FREQn keywords. The y map (extension YSZ) and the y-weighted temperature (TEY) are image
        extensions. This is the layout of :meth:`SZpack_model.write_fits_image`.

        Parameters
        ----------
        fname : str
            File name; '.fits' is added if missing.
        comments : str or list of str, optional
            Comments for the primary header. Default 'None'.
        overwrite : bool, optional
            Replace an existing file. Default False.
        """
        aux = [('YSZ', self.ydata, 'y', 'Compton y parameter')]
        if self.Tydata is not None:
            aux.append(('TEY', self.Tydata, 'keV', 'Compton-y weighted electron temperature'))
        s = self.sz_settings
        extra = [("SZMODEL", 'TT_model', 'thermal SZ from the y map'),
                 ("RELATIV", bool(s['relativistic']), 'relativistic tSZ corrections (SZpack)')]
        if s['relativistic']:
            extra.append(("SZMETHOD", str(s['method']), 'SZpack method'))
        sz_spectra.write_sz_cube(self, fname, aux=aux, extra=extra, comments=comments, overwrite=overwrite)

    def write_fits_image(self, fname, comments="None", overwrite=False):
        r"""Write the y map (2D) to a FITS file, with a WCS header.

        Parameters
        ----------
        fname : str
            File name; '.fits' is added if missing.
        comments : str or list of str, optional
            Comments for the header. Default 'None'.
        overwrite : bool, optional
            Replace an existing file. Default False.
        """
        # import pyfits as pf
        import astropy.io.fits as pf

        if fname[-5:] != ".fits":
            fname = fname + ".fits"

        
        hdu = pf.PrimaryHDU(self.ydata)
        hdu.header["SIMPLE"] = True
        hdu.header.comments["SIMPLE"] = 'conforms to FITS standard'
        hdu.header["BITPIX"] = int(-32)
        hdu.header.comments["BITPIX"] = '32 bit floating point'
        hdu.header["NAXIS"] = int(2)
        hdu.header["NAXIS1"] = int(self.ydata.shape[0])
        hdu.header["NAXIS2"] = int(self.ydata.shape[1])
        hdu.header["EXTEND"] = True
        hdu.header.comments["EXTEND"] = 'Extensions may be present'
        hdu.header["RADECSYS"] = 'ICRS    '
        hdu.header.comments["RADECSYS"] = "International Celestial Ref. System"
        hdu.header["CTYPE1"] = 'RA---TAN'
        hdu.header.comments["CTYPE1"] = "Coordinate type"
        hdu.header["CTYPE2"] = 'DEC--TAN'
        hdu.header.comments["CTYPE2"] = "Coordinate type"
        hdu.header["CUNIT1"] = 'deg     '
        hdu.header.comments["CUNIT1"] = 'Units'
        hdu.header["CUNIT2"] = 'deg     '
        hdu.header.comments["CUNIT2"] = 'Units'
        hdu.header["CRPIX1"] = float(self.npl/2.0)
        hdu.header.comments["CRPIX1"] = 'X of reference pixel'
        hdu.header["CRPIX2"] = float(self.npl/2.0)
        hdu.header.comments["CRPIX2"] = 'Y of reference pixel'
        hdu.header["CRVAL1"] = float(self.sp[0])
        hdu.header.comments["CRVAL1"] = 'RA of reference pixel (deg)'
        hdu.header["CRVAL2"] = float(self.sp[1])
        hdu.header.comments["CRVAL2"] = 'Dec of reference pixel (deg)'
        hdu.header["CD1_1"] = -float(self.ar/3600.)
        hdu.header.comments["CD1_1"] = 'RA deg per column pixel'
        hdu.header["CD1_2"] = float(0)
        hdu.header.comments["CD1_2"] = 'RA deg per row pixel'
        hdu.header["CD2_1"] = float(0)
        hdu.header.comments["CD2_1"] = 'Dec deg per column pixel'
        hdu.header["CD2_2"] = float(self.ar/3600.)
        hdu.header.comments["CD2_2"] = 'Dec deg per row pixel'

        hdu.header["RCVAL1"] = float(self.cc[0])
        hdu.header.comments["RCVAL1"] = 'Real center X of the data'
        hdu.header["RCVAL2"] = float(self.cc[1])
        hdu.header.comments["RCVAL2"] = 'Real center Y of the data'
        hdu.header["RCVAL3"] = float(self.cc[2])
        hdu.header.comments["RCVAL3"] = 'Real center Z of the data'
        hdu.header["UNITS"] = "kpc"
        hdu.header.comments["UNITS"] = 'Units for the RCVAL and PSIZE'
        hdu.header["PIXVAL"] = "y parameter"
        hdu.header.comments["PIXVAL"] = 'The y parameter for thermal SZ effect.'
        hdu.header["ORAD"] = float(self.rr)
        hdu.header.comments["ORAD"] = 'Rcut in physical for the image.'
        hdu.header["REDSHIFT"] = float(self.red)
        hdu.header.comments["REDSHIFT"] = 'The redshift of the object being put to'
        hdu.header["PSIZE"] = float(self.pxs)
        hdu.header.comments["PSIZE"] = 'The pixel size in physical at simulation time'

        hdu.header["AGLRES"] = float(self.ar)
        hdu.header.comments["AGLRES"] = '\'observation\' angular resolution in arcsec'

        hdu.header["ORIGIN"] = 'Software: PymSZ'
        hdu.header.comments["ORIGIN"] = 'https://github.com/weiguangcui/pymsz.git'
        hdu.header["VERSION"] = version.version  # get_property('__version__')
        hdu.header.comments["VERSION"] = 'Version of the software'
        hdu.header["DATE-OBS"] = Time.now().tt.isot
        if isinstance(comments, type([])) or isinstance(comments, type(())):
            for j in range(len(comments)):
                hdu.header["COMMENT"+str(j+1)] = comments[j]
        elif isinstance(comments, type("")) or isinstance(comments, type('')):
            hdu.header["COMMENT"] = comments
        else:
            raise ValueError("Do not accept this comments type! Please use str or list")
        hdu.writeto(fname, overwrite=overwrite)


class TK_model(object):
    r"""Map of :math:`b = \tau\,v_\mathrm{los}/c` for the kinetic SZ effect.

    Every gas particle contributes :math:`\sigma_T N_e v_\mathrm{los} / c`. The contributions are
    SPH-smoothed onto the image and divided by the pixel area. The kinetic SZ temperature change is
    :math:`\Delta T = T_\mathrm{CMB}\, b`. The velocities are relative to the bulk motion removed by
    `load_data` (see its `hmrad`).

    Parameters
    ----------
    simudata : load_data
        The simulation data.
    npixel : int or 'auto', optional
        Pixels per side of the image. Default 500. See `TT_model`.
    neighbours : int, optional
        Smooth every particle over this many neighbours, ignoring the smoothing lengths of the
        snapshot. Default None: use the smoothing lengths (27 neighbours if there are none).
    axis : str, list or ndarray, optional
        Line of sight: 'x', 'y', 'z' (default), rotation angles ``[alpha, beta, gamma]`` (degrees)
        or a 3x3 rotation matrix. See `TT_model`.
    AR : float, optional
        Pixel size in arcsec. Default None. See `TT_model`.
    SD : {2, 3}, optional
        Dimensions of the SPH smoothing. Default 2.
    SP : list of 2 floats, optional
        Sky position ``[RA, Dec]`` (degrees) of the image centre. Default ``[194.95, 27.98]``.
    Memreduce : bool, optional
        Lower-memory parallel smoothing. Default False. See `TT_model`.
    Ncpu : int, optional
        Number of processes for the SPH smoothing. Default None: all CPUs.
    Ntasks : int, optional
        Number of tasks for the SPH smoothing. Default None: `Ncpu`.
    redshift : float, optional
        Redshift at which the object is observed. Default None: the snapshot redshift. Use
        redshift > 0.
    zthick : float, optional
        Use only the gas within +-zthick (comoving kpc/h) of the centre along the line of sight.
        Default None: all loaded gas.
    sph_kernel : str, optional
        SPH kernel: 'cubic' (default), 'quartic', 'quintic', 'wendland2', 'wendland4' or
        'wendland6'.

    Attributes
    ----------
    bdata : ndarray
        The map of :math:`\tau\,v_\mathrm{los}/c`, shape (npixel, npixel), indexed [y, x]. Positive
        for gas moving towards the observer (a kSZ increment).
    bvel : float
        Line-of-sight component of the bulk velocity removed by `load_data` (km/s).
    pxs, ar, npl, red
        Pixel size (physical kpc and arcsec), number of pixels per side, redshift.
    szdata, freqs, unit
        The SZ images computed by :meth:`to_sz`, their frequencies (GHz) and unit.

    See Also
    --------
    TT_model : the thermal SZ map; it uses the same pixel grid, so the maps can be added.
    SZpack_model : kinetic SZ with relativistic corrections.

    Examples
    --------
    >>> tk = pymsz.TK_model(simd, npixel=512, AR=5, redshift=0.1)
    >>> tk.bdata                                  # tau v_los / c
    >>> tk.to_sz([90., 150., 220.], unit='uK')    # kinetic SZ images, Delta T = Tcmb * bdata
    """

    def __init__(self, simudata, npixel=500, neighbours=None, axis='z', AR=None, SD=2,
                 SP=[194.95, 27.98], Memreduce=False, Ncpu=None, Ntasks=None, redshift=None, zthick=None,
                 sph_kernel='cubic'):
        if isinstance(npixel, type("")) or isinstance(npixel, type('')):
            self.npl = npixel.lower()
        else:
            self.npl = npixel
        self.ngb = neighbours
        self.ax = axis
        self.ar = AR
        self.Memreduce = Memreduce
        self.ncpu = Ncpu
        self.ntask = Ntasks
        # frequency maps, filled by to_sz
        self.Tcmb = TCMB
        self.freqs = np.array([])
        self.unit = None
        self.szdata = np.array([])
        self.bvel = 0  # bulk velocity after rotation
        if redshift is None:
            self.red = simudata.cosmology['z']
        else:
            self.red = redshift
        self.zthick = zthick
        self.pxs = None
        self.SD = SD
        self.bdata = np.array([])
        self.sph_kn = sph_kernel
        # self.ad = 1.  # angular diameter distance normalized to 1 kpc

        # if len(SP) == 2:
        self.sp = SP
        # self.cc = simd.center
        # elif len(SP) == 3:
        #     self.sp = False
        # self.cc = SP
        # else:
        #     raise ValueError("SP length should be either 2 or 3!")

        if self.ar is None and self.npl == 'auto':
            print("Do not accept AR == None and npixel=='AUTO' !! \n npixel is reset to 500 !")
            self.npl = 500
        if self.SD not in [2, 3]:
            raise ValueError("smoothing dimension must be 2 or 3" % SD)

        if simudata.data_type == "snapshot":
            self._cal_snap(simudata)
        elif simudata.data_type == "yt_data":
            raise NotImplementedError("TK_model does not support yt data yet: load the snapshot with "
                                      "load_data(..., snapshot=True)")
        else:
            raise ValueError("Do not accept this data type %s"
                             "Please try to use load_data to get the data" % simudata.data_type)

    def _cal_snap(self, simd):

        pos, vel, self.bvel = rotate_data(simd.pos/simd.cosmology['h']/(1+simd.cosmology['z']),
                                          self.ax, vel=simd.vel, bvel=simd.bulkvel)  # pos in physical
        simd.prep_ss_KT(vel)

        if self.red <= 0. and self.npl == 'auto':
            print("Do not accept redshift == 0 and npixel=='AUTO' !! \n npixel is reset to 500! ")
            self.npl = 500

        self.cc = simd.center/simd.cosmology['h']/(1+simd.cosmology['z'])
        self.rr = simd.radius/simd.cosmology['h']/(1+simd.cosmology['z'])
        if self.zthick is not None:
            self.zthick = self.zthick/simd.cosmology['h']/(1+simd.cosmology['z'])
            idc = (pos[:, 2] > -self.zthick) & (pos[:, 2] < self.zthick)
            pos = pos[idc]
            Kszdata = simd.Kszdata[idc]
        else:
            Kszdata = np.copy(simd.Kszdata)

        if isinstance(simd.hsml, type(0)):
            self.ngb = 27
            hsml = None
        else:  # with hsml
            if self.ngb is None:             # need to estimate neighbors
                self.ngb=0
                ids = np.sum((simd.pos - simd.pos[simd.hsml.argmin()])**2, axis=1) <= simd.hsml.min()**2
                if simd.hsml[ids].size > self.ngb:
                    self.ngb = simd.hsml[ids].size
                ids = np.sum((simd.pos - simd.pos[simd.hsml.argmax()])**2, axis=1) <= simd.hsml.max()**2
                if simd.hsml[ids].size > self.ngb:
                    self.ngb = simd.hsml[ids].size
                self.ngb = np.int32(self.ngb*1.05)+1 #to be safe
                print('self.ngb = ', self.ngb)

                if self.zthick is not None:
                    hsml = simd.hsml[idc]
                else:
                    hsml = np.copy(simd.hsml)
                hsml = hsml/simd.cosmology['h']/(1+simd.cosmology['z'])
            else:               # have set the neighbors, need to ignore the hsml
                hsml = None

        if self.npl != 'auto':
            minx = pos[:, 0].min()
            maxx = pos[:, 0].max()
            miny = pos[:, 1].min()
            maxy = pos[:, 1].max()
            # same pixel size as TT_model (projected plane only), so tSZ and kSZ maps can be added
            self.pxs = np.min([maxx - minx, maxy - miny]) / self.npl
        if self.red > 0.:
            if simd.cosmology['omega_matter'] != 0:
                cosmo = FlatLambdaCDM(H0=simd.cosmology['h'] * 100,
                                      Om0=simd.cosmology['omega_matter'])
            else:
                print('No cosmology loaded, assume WMAP7!')
                cosmo = WMAP7
            # self.ad = cosmo.angular_diameter_distance(self.red).to("kpc").value  # in cm

            if self.ar is not None:
                self.pxs = self.ar / cosmo.arcsec_per_kpc_proper(self.red).value  # in kpc physical
                if self.npl == 'auto':
                    self.npl = np.int32(self.rr*2/self.pxs)+1
            else:
                self.ar = self.pxs * cosmo.arcsec_per_kpc_proper(self.red).value
        else:
            if self.ar is None:
                self.ar = 1.

        # cut out unused data
        if self.npl != 'auto':
            idc = (pos[:, 0] >= -self.npl*self.pxs/2.) & (pos[:, 0] <= self.npl*self.pxs/2.) &\
                (pos[:, 1] >= -self.npl*self.pxs/2.) & (pos[:, 1] <= self.npl*self.pxs/2.)
            pos = pos[idc]
            Kszdata = Kszdata[idc]
            if hsml is not None:
                hsml = hsml[idc]
            if self.npl != np.int32(self.rr*2/self.pxs)+1:
                self.rr = self.pxs * (self.npl-1) / 2.

        if self.SD == 2:
            self.bdata = SPH_smoothing(Kszdata, pos[:, :2], self.pxs, self.ngb, hsml=hsml,
                                       pxln=self.npl, Memreduce=self.Memreduce, Ncpu=self.ncpu,
                                       Ntasks=self.ntask, kernel_name=self.sph_kn)
        else:
            self.bdata = SPH_smoothing(Kszdata, pos, self.pxs, self.ngb, hsml=hsml,
                                       pxln=self.npl, Memreduce=self.Memreduce, Ncpu=self.ncpu,
                                       Ntasks=self.ntask, kernel_name=self.sph_kn)
            self.bdata = np.sum(self.bdata, axis=2)
        self.bdata = self.bdata.T / self.pxs**2

    # ------------------------------------------------------------------------------------------
    # b-map -> SZ images at given frequencies
    # ------------------------------------------------------------------------------------------
    def to_sz(self, freqs, unit='MJy/sr', Tcmb=TCMB):
        r"""Kinetic SZ images at the observed frequencies `freqs`, from the b map.

        Non-relativistic: :math:`\Delta n = G(x)\, b`, i.e. :math:`\Delta T = T_\mathrm{CMB}\, b`.
        The relativistic kSZ corrections are included in `SZpack_model`.

        Parameters
        ----------
        freqs : float or array_like
            Observed frequencies in GHz.
        unit : str, optional
            'MJy/sr' (default), 'Jy/pixel', 'K', 'uK' or 'Dn', see :meth:`TT_model.to_sz`.
        Tcmb : float, optional
            CMB temperature in K. Default 2.7255.

        Returns
        -------
        ndarray
            The SZ images, shape (len(freqs), npixel, npixel), also kept in `szdata`.
        """
        self.unit = sz_spectra.check_unit(unit)
        self.Tcmb = Tcmb
        self.freqs = np.atleast_1d(np.asarray(freqs, dtype=np.float64))
        self.x = sz_spectra.freq_to_x(self.freqs, self.Tcmb)
        self.szdata = self.sz_maps(self.freqs, unit)
        return self.szdata

    def sz_maps(self, freqs, unit='Dn'):
        r"""Kinetic SZ images at `freqs` (GHz) in `unit`, shape (len(freqs), npixel, npixel).

        This is the interface used by :meth:`pymsz.Telescope.observe`."""
        freqs = np.atleast_1d(np.asarray(freqs, dtype=np.float64))
        x = sz_spectra.freq_to_x(freqs, self.Tcmb)
        dn = sz_spectra.ksz_nonrel(x)[:, None, None] * np.asarray(self.bdata)[None]
        return sz_spectra.dn_to_unit(dn, freqs, unit, self.Tcmb, self.ar)

    def write_sz_fits(self, fname, comments="None", overwrite=False):
        r"""Write the kinetic SZ images of :meth:`to_sz` and the b map to a FITS file.

        The images are the primary HDU, shape [nfreq, npixel, npixel]; the b map is the extension
        BSZ. This is the layout of :meth:`SZpack_model.write_fits_image`.

        Parameters
        ----------
        fname : str
            File name; '.fits' is added if missing.
        comments : str or list of str, optional
            Comments for the primary header. Default 'None'.
        overwrite : bool, optional
            Replace an existing file. Default False.
        """
        aux = [('BSZ', self.bdata, 'tau*v/c', 'tau v_los/c, >0 towards observer')]
        extra = [("SZMODEL", 'TK_model', 'kinematic SZ from the b map')]
        sz_spectra.write_sz_cube(self, fname, aux=aux, extra=extra, comments=comments, overwrite=overwrite)

    def write_fits_image(self, fname, comments='None', overwrite=False):
        r"""Write the b map (2D) to a FITS file, with a WCS header.

        Parameters
        ----------
        fname : str
            File name; '.fits' is added if missing.
        comments : str or list of str, optional
            Comments for the header. Default 'None'.
        overwrite : bool, optional
            Replace an existing file. Default False.
        """
        import astropy.io.fits as pf

        if fname[-5:] != ".fits":
            fname = fname + ".fits"

        hdu = pf.PrimaryHDU(self.bdata)
        hdu.header["SIMPLE"] = 'T'
        hdu.header.comments["SIMPLE"] = 'conforms to FITS standard'
        hdu.header["BITPIX"] = int(-32)
        hdu.header.comments["BITPIX"] = '32 bit floating point'
        hdu.header["NAXIS"] = int(2)
        hdu.header["NAXIS1"] = int(self.bdata.shape[0])
        hdu.header["NAXIS2"] = int(self.bdata.shape[1])
        hdu.header["EXTEND"] = True
        hdu.header.comments["EXTEND"] = 'Extensions may be present'
        hdu.header["RADECSYS"] = 'ICRS    '
        hdu.header.comments["RADECSYS"] = "International Celestial Ref. System"
        hdu.header["CTYPE1"] = 'RA---TAN'
        hdu.header.comments["CTYPE1"] = "Coordinate type"
        hdu.header["CTYPE2"] = 'DEC--TAN'
        hdu.header.comments["CTYPE2"] = "Coordinate type"
        hdu.header["CUNIT1"] = 'deg     '
        hdu.header.comments["CUNIT1"] = 'Units'
        hdu.header["CUNIT2"] = 'deg     '
        hdu.header.comments["CUNIT2"] = 'Units'
        hdu.header["CRPIX1"] = float(self.npl/2.0)
        hdu.header.comments["CRPIX1"] = 'X of reference pixel'
        hdu.header["CRPIX2"] = float(self.npl/2.0)
        hdu.header.comments["CRPIX2"] = 'Y of reference pixel'
        hdu.header["CRVAL1"] = float(self.sp[0])
        hdu.header.comments["CRVAL1"] = 'RA of reference pixel (deg)'
        hdu.header["CRVAL2"] = float(self.sp[1])
        hdu.header.comments["CRVAL2"] = 'Dec of reference pixel (deg)'
        hdu.header["CD1_1"] = -float(self.ar/3600.)
        hdu.header.comments["CD1_1"] = 'RA deg per column pixel'
        hdu.header["CD1_2"] = float(0)
        hdu.header.comments["CD1_2"] = 'RA deg per row pixel'
        hdu.header["CD2_1"] = float(0)
        hdu.header.comments["CD2_1"] = 'Dec deg per column pixel'
        hdu.header["CD2_2"] = float(self.ar/3600.)
        hdu.header.comments["CD2_2"] = 'Dec deg per row pixel'

        hdu.header["RCVAL1"] = float(self.cc[0])
        hdu.header.comments["RCVAL1"] = 'Real center X of the data'
        hdu.header["RCVAL2"] = float(self.cc[1])
        hdu.header.comments["RCVAL2"] = 'Real center Y of the data'
        hdu.header["RCVAL3"] = float(self.cc[2])
        hdu.header.comments["RCVAL3"] = 'Real center Z of the data'
        hdu.header["UNITS"] = "kpc"
        hdu.header.comments["UNITS"] = 'Units for the RCVAL and PSIZE'
        hdu.header["PIXVAL"] = "omega"
        hdu.header.comments["PIXVAL"] = 'T_{kSZ} = omega*T_{cmb}(~ 2.73)'
        hdu.header["ORAD"] = float(self.rr)
        hdu.header.comments["ORAD"] = 'Rcut in physical for the image'
        hdu.header["REDSHIFT"] = float(self.red)
        hdu.header.comments["REDSHIFT"] = 'The redshift of the object being put to'
        hdu.header["PSIZE"] = float(self.pxs)
        hdu.header.comments["PSIZE"] = 'The pixel size in physical at simulation time'

        hdu.header["AGLRES"] = float(self.ar)
        hdu.header.comments["AGLRES"] = '\'observation\' angular resolution in arcsec'

        hdu.header["ORIGIN"] = 'Software: PymSZ'
        hdu.header.comments["ORIGIN"] = 'https://github.com/weiguangcui/pymsz.git'
        hdu.header["VERSION"] = version.version  # get_property('__version__')
        hdu.header.comments["VERSION"] = 'Version of the software'
        hdu.header["DATE-OBS"] = Time.now().tt.isot
        if isinstance(comments, type([])) or isinstance(comments, type(())):
            for j in range(len(comments)):
                hdu.header["COMMENT"+str(j+1)] = comments[j]
        elif isinstance(comments, type("")) or isinstance(comments, type('')):
            hdu.header["COMMENT"] = comments
        else:
            raise ValueError("Do not accept this comments type! Please use str or list")
        hdu.writeto(fname, overwrite=overwrite)
