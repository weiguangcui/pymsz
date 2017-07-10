import numpy as np
from rotate_data import rotate_data, SPH_smoothing
from astropy.cosmology import FlatLambdaCDM, WMAP7
# scipy must >= 0.17 to properly use this!
# from scipy.stats import binned_statistic_2d


# def SPH(x):  # 3D Cubic
#     data = np.zeros(x.size, dtype=float)
#     ids = (x > 0) & (x <= 0.5)
#     data[ids] = 1 - 6 * x[ids]**2 + 6 * x[ids]**3
#     ids = (x > 0.5) & (x < 1)
#     data[ids] = 2 * (1 - x[ids])**3
#     return data * 2.5464790894703255
# def SPH(x, h):  # 2D
#     if (x > 0) and (x <= 0.5):
#         return 10*(1 - 3*x**2/2. + 3*x**3/4.)/7./np.pi/h**2
#     elif (x > 1) and (x <= 2):
#         return 10*(2 - x)**3/4./7./np.pi/h**2
#     else:
#         return 0


class TT_model(object):
    r""" Theoretical calculation of y and T_sz -map for the thermal SZ effect.
    model = TT_model(model_file, npixel, axis)

    Parameters
    ----------
    simudata : the simulation data from load_data
    npixel   : number of pixels for your image, int.
                Assume that x-y have the same number of pixels
    axis     : can be 'x', 'y', 'z', or a list of degrees [alpha, beta, gamma],
               which will rotate the data points by $\alpha$ around the x-axis,
               $\beta$ around the y-axis, and $\gamma$ around the z-axis
    neighbours: this parameter only works with simulation data (not yt data).
                If this is set, it will force the SPH particles smoothed into nearby N
                neighbours, HSML from the simulation will be ignored.
                If no HSML provided in the simulation, neighbours = 27
    AR       : angular resolution in arcsec.
                Default : 0, which gives npixel = 2 * cluster radius
                and ignores the cluster's redshift.
                Otherwise, cluster's redshift with AR decides how large the cluster looks.
    SD       : dimensions for SPH smoothing. Type: int. Default: 2.
                Must be 2 or 3!
    redshift : The redshift where the cluster is at.
                Default : None, we will look it from simulation data.
                If redshift = 0, it will be automatically put into 0.02,
                unless AR is set to None.
    zthick  : The thickness in projection direction. Default: None.
                If None, use all data from cutting region. Otherwise set a value in simulation
                length unit (kpc/h normally), then a slice of data [center-zthick, center+zthick]
                will be used to make the y-map.
    sph_kernel : The kernel used to smoothing the y values. Default : "cubic"
                Choose from 'cubic': cubic spline; 'quartic': quartic spline;
                'quintic': quintic spline; 'wendland2': Wendland C2; 'wendland4': Wendland C4;
                'wendland6': Wendland C6;

    Returns
    -------
    Theoretical projected y-map in a given direction. 2D mesh data right now.

    See also
    --------
    SZ_models for the mock SZ signal at different frequencies.

    Notes
    -----


    Example
    -------
    mm=pymsz.TT_models(simudata, 1024, "z")
    """

    def __init__(self, simudata, npixel=500, neighbours=None, axis='z', AR=0, SD=2,
                 redshift=None, zthick=None, sph_kernel='cubic'):
        self.npl = npixel
        self.ngb = neighbours
        self.ax = axis
        self.ar = AR
        self.red = redshift
        self.zthick = zthick
        self.pxs = 0
        self.SD = SD
        self.ydata = np.array([])
        self.sph_kn = sph_kernel

        if self.SD not in [2, 3]:
            raise ValueError("smoothing dimension must be 2 or 3" % SD)

        if simudata.data_type == "snapshot":
            self.cc = simudata.center
            self.rr = simudata.radius
            self._cal_snap(simudata)
        elif simudata.data_type == "yt_data":
            self._cal_yt(simudata)
        else:
            raise ValueError("Do not accept this data type %s"
                             "Please try to use load_data to get the data" % simudata.data_type)

    # def TH_ymap(simd, npixel=500, neighbours=None, axis='z', AR=None, redshift=None):

    def _cal_snap(self, simd):
        # Kpc = 3.0856775809623245e+21  # cm
        simd.prep_ss_TT()

        if self.red is None:
            self.red = simd.cosmology['z']

        pos = rotate_data(simd.pos, self.ax)
        if self.zthick is not None:
            idc = (pos[:, 2] > -self.zthick) & (pos[:, 2] < self.zthick)
            pos = pos[idc]
            Tszdata = simd.Tszdata[idc]
        else:
            Tszdata = simd.Tszdata
        # if simd.radius is not None:
        #     idc = (pos[:, 2] > -1 * simd.radius) & (pos[:, 2] <= simd.radius) & \
        #           (pos[:, 0] > -1 * simd.radius) & (pos[:, 0] <= simd.radius) & \
        #           (pos[:, 1] > -1 * simd.radius) & (pos[:, 1] <= simd.radius)
        #     pos = pos[idc]
        # Tszdata = simd.Tszdata[idc]

        if isinstance(simd.hsml, type(0)):
            self.ngb = 27
            hsml = None
        else:
            if self.zthick is not None:
                hsml = simd.hsml[idc]
            else:
                hsml = simd.hsml
            self.ngb = None

        if self.ar is 0:
            minx = pos[:, 0].min()
            maxx = pos[:, 0].max()
            miny = pos[:, 1].min()
            maxy = pos[:, 1].max()
            minz = pos[:, 2].min()
            maxz = pos[:, 2].max()
            self.pxs = np.min([maxx - minx, maxy - miny, maxz - minz]) / self.npl

            # Tszdata /= (self.pxs * Kpc / simd.cosmology["h"])**2
            # if self.SD == 2:
            #     self.ydata = SPH_smoothing(Tszdata, pos[:, :2], self.pxs, hsml=hsml,
            #                                neighbors=self.ngb, kernel_name=self.sph_kn)
            # else:
            #     self.ydata = SPH_smoothing(Tszdata, pos, self.pxs, hsml=hsml,
            #                                neighbors=self.ngb, kernel_name=self.sph_kn)
        else:
            if self.red <= 0.0:
                self.red = 0.02
            if simd.cosmology['omega_matter'] != 0:
                cosmo = FlatLambdaCDM(H0=simd.cosmology['h'] * 100,
                                      Om0=simd.cosmology['omega_matter'])
            else:
                cosmo = WMAP7
            self.pxs = self.ar / cosmo.arcsec_per_kpc_comoving(self.red).value * simd.cosmology['h']  # in kpc/h

        # Tszdata /= (Kpc / simd.cosmology["h"])**2
        self.ydata /= self.pxs**2

        if self.SD == 2:
            self.ydata = SPH_smoothing(Tszdata, pos[:, :2], self.pxs, hsml=hsml,
                                       neighbors=self.ngb, pxln=self.npl,
                                       kernel_name=self.sph_kn)
        else:
            self.ydata = SPH_smoothing(Tszdata, pos, self.pxs, hsml=hsml,
                                       neighbors=self.ngb,  pxln=self.npl,
                                       kernel_name=self.sph_kn)
            self.ydata = np.sum(self.ydata, axis=2)

        # self.ydata /= self.pxs**2

    def _cal_yt(self, simd):
        # from yt.units import cm
        Ptype = simd.prep_yt_TT()
        if self.red is None:
            self.red = simd.yt_ds.current_redshift
        if self.ar is 0:
            rr = 2. * simd.radius
        else:
            if self.red <= 0.0:
                self.red = 0.02

            if simd.yt_ds.omega_matter != 0:
                cosmo = FlatLambdaCDM(H0=simd.yt_ds.hubble_constant
                                      * 100, Om0=simd.yt_ds.omega_matter)
            else:
                cosmo = WMAP7
            self.pxs = cosmo.arcsec_per_kpc_proper(self.red) * self.ar * simd.yt_ds.hubble_constant
            rr = self.npl * self.pxs
        if isinstance(self.ax, type('x')):
            projection = simd.yt_ds.proj(('deposit', Ptype + '_smoothed_Tsz'), self.ax,
                                         center=simd.center, data_source=simd.yt_sp)
            FRB = projection.to_frb(rr, self.npl)
            self.ydata = FRB[('deposit', Ptype + '_smoothed_Tsz')]

    def write_fits_image(self, fname, clobber=False):
        r"""
        Generate a image by binning X-ray counts and write it to a FITS file.

        Parameters
        ----------
        imagefile : string
            The name of the image file to write.
        clobber : boolean, optional
            Set to True to overwrite a previous file.
        """
        import pyfits as pf

        if fname[-5:] != ".fits":
            fname = fname + ".fits"

        hdu = pf.PrimaryHDU(self.ydata)
        hdu.header["RCVAL1"] = float(self.cc[0])
        hdu.header["RCVAL2"] = float(self.cc[1])
        hdu.header["RCVAL3"] = float(self.cc[2])
        hdu.header["UNITS"] = "kpc/h"
        hdu.header["ORAD"] = float(self.rr)
        hdu.header["REDSHIFT"] = float(self.red)
        hdu.header["PSIZE"] = float(self.pxs)
        hdu.header["AGLRES"] = float(self.ar)
        hdu.header["NOTE"] = ""
        hdu.writeto(fname, clobber=clobber)


class TK_model(object):
    r""" Theoretical calculation of sz-map for the kinetic SZ effect.
    model = TK_model(model_file, npixel, axis)

    Parameters
    ----------
    simudata : the simulation data from load_data
    npixel   : number of pixels for your image, int.
                Assume that x-y have the same number of pixels
    axis     : can be 'x', 'y', 'z', or a list of degrees [alpha, beta, gamma],
               which will rotate the data points by $\alpha$ around the x-axis,
               $\beta$ around the y-axis, and $\gamma$ around the z-axis
    neighbours: this parameter only works with simulation data (not yt data).
                If this is set, it will force the SPH particles smoothed into nearby N
                neighbours, HSML from the simulation will be ignored.
                If no HSML provided in the simulation, neighbours = 27
    AR       : angular resolution in arcsec.
                Default : 0, which gives npixel = 2 * cluster radius
                and ignores the cluster's redshift.
                Otherwise, cluster's redshift with AR decides how large the cluster looks.
    SD       : dimensions for SPH smoothing. Type: int. Default: 2.
                Must be 2 or 3!
    redshift : The redshift where the cluster is at.
                Default : None, we will look it from simulation data.
                If redshift = 0, it will be automatically put into 0.02,
                unless AR is set to None.
    zthick  : The thickness in projection direction. Default: None.
                If None, use all data from cutting region. Otherwise set a value in simulation
                length unit (kpc/h normally), then a slice of data [center-zthick, center+zthick]
                will be used to make the y-map.
    sph_kernel : The kernel used to smoothing the y values. Default : "cubic"
                Choose from 'cubic': cubic spline; 'quartic': quartic spline;
                'quintic': quintic spline; 'wendland2': Wendland C2; 'wendland4': Wendland C4;
                'wendland6': Wendland C6;

    Returns
    -------
    Theoretical projected y-map in a given direction. 2D mesh data right now.

    See also
    --------
    SZ_models for the mock SZ signal at different frequencies.

    Notes
    -----


    Example
    -------
    mm=pymsz.TK_models(simudata, 1024, "z")
    """

    def __init__(self, simudata, npixel=500, neighbours=None, axis='z', AR=0, SD=2,
                 redshift=None, zthick=None, sph_kernel='cubic'):
        self.npl = npixel
        self.ngb = neighbours
        self.ax = axis
        self.ar = AR
        self.red = redshift
        self.zthick = zthick
        self.pxs = 0
        self.SD = SD
        self.ydata = np.array([])
        self.sph_kn = sph_kernel

        if self.SD not in [2, 3]:
            raise ValueError("smoothing dimension must be 2 or 3" % SD)

        if simudata.data_type == "snapshot":
            self.cc = simudata.center
            self.rr = simudata.radius
            self._cal_snap(simudata)
        elif simudata.data_type == "yt_data":
            self._cal_yt(simudata)
        else:
            raise ValueError("Do not accept this data type %s"
                             "Please try to use load_data to get the data" % simudata.data_type)

    def _cal_snap(self, simd):
        # Kpc = 3.0856775809623245e+21  # cm
        simd.prep_ss_TT()
