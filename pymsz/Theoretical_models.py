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
                Assume that x-y always have the same number of pixels.
                It can be set to 'AUTO', then it will be decided by the halo radius and AR.
                So, npixel='AUTO' and AR=None can not be set at the same time!
    axis     : can be 'x', 'y', 'z', or a list of degrees [alpha, beta, gamma],
               which will rotate the data points by $\alpha$ around the x-axis,
               $\beta$ around the y-axis, and $\gamma$ around the z-axis
    neighbours: this parameter only works with simulation data (not yt data).
                If this is set, it will force the SPH particles smoothed into nearby N
                neighbours, HSML from the simulation will be ignored.
                If no HSML provided in the simulation, neighbours = 27
    AR       : angular resolution in arcsec. Default : None.
                Cluster's redshift with AR decides the image pixel size.
                If None, the whole cluster will be projected to the image with npixel resolution.
    SD       : dimensions for SPH smoothing. Type: int. Default: 2.
                Must be 2 or 3!
    # pxsize   : physical/proper pixel size of the image. Type: float, unit: kpc.
    #             Default: None
    #             If set, this will invaided the calculation from AR.
    Ncpu     : number of CPU for parallel calculation. Type: int. Default: None, all cpus from the
                computer will be used.
                Note, this parallel calculation is only for the SPH smoothing.
    # periodic : periodic condition applied for the SPH smoothing region. Tyep: bool. Default: False.
    #             periodic condition works for the too fine mesh (which means oversmoothing),
    #             you can turn this on to avoid small boundary effects. So, this is only for SPH.
    redshift : The redshift where the cluster is at.
                Default : None, we will look it from simulation data.
                Note : If redshift = 0, the returning results will be y_int, i.e. y*d^2_A,
                which takes out the angular diameter distance.
                This will also ingore the set of AR. The image pixel size =
                2 * cluster radius/npixel, so the npixel MUST NOT be 'AUTO' at redshift = 0.
    zthick  : The thickness in projection direction. Default: None.
                If None, use all data from cutting region.
                Otherwise set a value in simulation length unit kpc/h normally,
                then a slice of data [center-zthick, center+zthick] will be used to make the y-map.
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
    This program does not accpte redshift=0 or AR == None with npixel = "AUTO".
    If npixel is "AUTO" and AR != None, then the whole cluster will project to the image with
    npixel = 2*radius/pixelsize, where pxielsize is given by AR.

    Example
    -------
    mm=pymsz.TT_models(simudata, npixel=1024, "z")
    """

    def __init__(self, simudata, npixel=500, neighbours=None, axis='z', AR=None, SD=2,
                 Ncpu=None, redshift=None, zthick=None, sph_kernel='cubic'):
        self.npl = npixel
        self.ngb = neighbours
        self.ax = axis
        self.ar = AR
        self.red = redshift
        self.zthick = zthick
        self.pxs = None
        self.SD = SD
        # self.periodic = periodic
        self.ncpu = Ncpu
        self.ydata = np.array([])
        self.sph_kn = sph_kernel
        # self.ad = 1.  # angular diameter distance normalized to 1 kpc

        if self.ar is None and self.npl == 'AUTO':
            raise ValueError("Do not accept AR == None and npixel=='AUTO' !!")
        if self.SD not in [2, 3]:
            raise ValueError("smoothing dimension must be 2 or 3" % SD)

        if simudata.data_type == "snapshot":
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
        if self.red <= 0. and self.npl == 'AUTO':
            raise ValueError("Do not accept redshift == 0 and npixel=='AUTO' !!")

        self.cc = simd.center/simd.cosmology['h']/(1+self.red)
        self.rr = simd.radius/simd.cosmology['h']/(1+self.red)
        pos = rotate_data(simd.pos/simd.cosmology['h']/(1+self.red), self.ax)  # to proper distance
        if self.zthick is not None:
            self.zthick = self.zthick/simd.cosmology['h']/(1+self.red)
            idc = (pos[:, 2] > -self.zthick) & (pos[:, 2] < self.zthick)
            pos = pos[idc]
            Tszdata = simd.Tszdata[idc]
        else:
            Tszdata = np.copy(simd.Tszdata)
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
                hsml = np.copy(simd.hsml)
            hsml = hsml/simd.cosmology['h']/(1+self.red)
            self.ngb = None

        if self.npl != 'AUTO':
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
            # self.ad = cosmo.angular_diameter_distance(self.red).to("kpc").value  # in cm

            if self.ar is not None:
                self.pxs = self.ar/cosmo.arcsec_per_kpc_proper(self.red).value  # in kpc
                if self.npl == 'AUTO':
                    self.npl = np.int32(self.rr*2/self.pxs)+1

        # cut out unused data
        if self.npl != 'AUTO':
            idc = (pos[:, 0] >= -self.npl*self.pxs/2.) & (pos[:, 0] <= self.npl*self.pxs/2.) &\
                (pos[:, 1] >= -self.npl*self.pxs/2.) & (pos[:, 1] <= self.npl*self.pxs/2.)
            pos = pos[idc]
            Tszdata = Tszdata[idc]
            if hsml is not None:
                hsml = hsml[idc]

        # Tszdata /= (self.pxs * Kpc / simd.cosmology["h"])**2

        if self.SD == 2:
            self.ydata = SPH_smoothing(Tszdata, pos[:, :2], self.pxs, hsml=hsml,
                                       neighbors=self.ngb, pxln=self.npl, Ncpu=self.ncpu,
                                       kernel_name=self.sph_kn)
        else:
            # be ware that zthick could cause some problems if it is larger than pxs*npl!!
            # This has been taken in care in the rotate_data function.
            self.ydata = SPH_smoothing(Tszdata, pos, self.pxs, hsml=hsml,
                                       neighbors=self.ngb, pxln=self.npl,
                                       Ncpu=self.ncpu, kernel_name=self.sph_kn)
            self.ydata = np.sum(self.ydata, axis=2)
        self.ydata = self.ydata.T
        self.ydata /= self.pxs**2

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
        hdu.header["UNITS"] = "kpc"
        hdu.header["RADIUS"] = float(self.rr)
        hdu.header["REDSHIFT"] = float(self.red)
        hdu.header["PSIZE"] = float(self.pxs)
        if self.ar is None:
            hdu.header["AGLRES"] = 0
        else:
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
                Assume that x-y always have the same number of pixels.
                It can be set to 'AUTO', then it will be decided by the halo radius and AR.
                So, npixel='AUTO' and AR=None can not be set at the same time!
    axis     : can be 'x', 'y', 'z', or a list of degrees [alpha, beta, gamma],
               which will rotate the data points by $\alpha$ around the x-axis,
               $\beta$ around the y-axis, and $\gamma$ around the z-axis
    neighbours: this parameter only works with simulation data (not yt data).
                If this is set, it will force the SPH particles smoothed into nearby N
                neighbours, HSML from the simulation will be ignored.
                If no HSML provided in the simulation, neighbours = 27
    AR       : angular resolution in arcsec. Default : None.
                Cluster's redshift with AR decides the image pixel size.
                If None, the whole cluster will be projected to the image with npixel resolution.
    SD       : dimensions for SPH smoothing. Type: int. Default: 2.
                Must be 2 or 3!
    # pxsize   : pixel size of the image. Type: float, unit: kpc. Default: None
    #             If set, this will invaided the calculation from AR.
    Ncpu     : number of CPU for parallel calculation. Type: int. Default: None, all cpus from the
                computer will be used.
                This parallel calculation is only for the SPH smoothing.
    # periodic : periodic condition for the SPH smoothing region. Tyep: bool. Default: False.
    #             periodic condition works for the too fine mesh (which means oversmoothing),
    #             you can consider turn this on to avoid boundary effects. So, this is also for SPH.
    redshift : The redshift where the cluster is at.
                Default : None, we will look it from simulation data.
                Note : If redshift = 0, the returning results will be y_int, i.e. y*d^2_A,
                which takes out the angular diameter distance.
                This will also ingore the set of AR. The image pixel size =
                2 * cluster radius/npixel, so the npixel MUST NOT be 'AUTO' at redshift = 0.
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
    Theoretical projected omega-map in a given direction. 2D mesh data right now.

    See also
    --------
    SZ_models for the mock SZ signal at different frequencies.

    Notes
    -----
    The retrun is omega map, not the delta T_{kSZ}!, which equals omega*T_{cmb} ~ 2.73 k
    This program does not accpte redshift=0 or AR == None with npixel = "AUTO".
    If npixel is "AUTO" and AR != None, then the whole cluster will project to the image with
    npixel = 2*radius/pixelsize, where pxielsize is given by AR.

    Example
    -------
    mm=pymsz.TK_models(simudata, npixel=1024, "z")
    mm.bdata  # this contains the b-map
    """

    def __init__(self, simudata, npixel=500, neighbours=None, axis='z', AR=0, SD=2,
                 Ncpu=None, redshift=None, zthick=None, sph_kernel='cubic'):
        self.npl = npixel
        self.ngb = neighbours
        self.ax = axis
        self.ar = AR
        self.ncpu = Ncpu
        # self.periodic = periodic
        self.red = redshift
        self.zthick = zthick
        self.pxs = None
        self.SD = SD
        self.bdata = np.array([])
        self.sph_kn = sph_kernel
        # self.ad = 1.  # angular diameter distance normalized to 1 kpc

        if self.ar is None and self.npl == 'AUTO':
            raise ValueError("Do not accept AR == None and npixel=='AUTO' !!")
        if self.SD not in [2, 3]:
            raise ValueError("smoothing dimension must be 2 or 3" % SD)

        if simudata.data_type == "snapshot":
            self._cal_snap(simudata)
        elif simudata.data_type == "yt_data":
            self._cal_yt(simudata)
        else:
            raise ValueError("Do not accept this data type %s"
                             "Please try to use load_data to get the data" % simudata.data_type)

    def _cal_snap(self, simd):

        pos, vel = rotate_data(simd.pos/simd.cosmology['h']/(1+self.red), self.ax, vel=simd.vel)
        simd.prep_ss_KT(vel)

        if self.red is None:
            self.red = simd.cosmology['z']
        if self.red <= 0. and self.npl == 'AUTO':
            raise ValueError("Do not accept redshift == 0 and npixel=='AUTO' !!")

        self.cc = simd.center/simd.cosmology['h']/(1+self.red)
        self.rr = simd.radius/simd.cosmology['h']/(1+self.red)
        if self.zthick is not None:
            self.zthick = self.zthick/simd.cosmology['h']/(1+self.red)
            idc = (pos[:, 2] > -self.zthick) & (pos[:, 2] < self.zthick)
            pos = pos[idc]
            Kszdata = simd.Kszdata[idc]
        else:
            Kszdata = np.copy(simd.Kszdata)

        if isinstance(simd.hsml, type(0)):
            self.ngb = 27
            hsml = None
        else:
            if self.zthick is not None:
                hsml = simd.hsml[idc]
            else:
                hsml = np.copy(simd.hsml)
            hsml = hsml/simd.cosmology['h']/(1+self.red)
            self.ngb = None

        if self.npl != 'AUTO':
            minx = pos[:, 0].min()
            maxx = pos[:, 0].max()
            miny = pos[:, 1].min()
            maxy = pos[:, 1].max()
            minz = pos[:, 2].min()
            maxz = pos[:, 2].max()
            self.pxs = np.min([maxx - minx, maxy - miny, maxz - minz]) / self.npl
        if self.red > 0.:
            if simd.cosmology['omega_matter'] != 0:
                cosmo = FlatLambdaCDM(H0=simd.cosmology['h'] * 100,
                                      Om0=simd.cosmology['omega_matter'])
            else:
                print('No cosmology loaded, assume WMAP7!')
                cosmo = WMAP7
            # self.ad = cosmo.angular_diameter_distance(self.red).to("kpc").value  # in cm

            if self.ar is not None:
                self.pxs = self.ar / cosmo.arcsec_per_kpc_proper(self.red).value  # in kpc
                if self.npl == 'AUTO':
                    self.npl = np.int32(self.rr*2/self.pxs)+1

        # cut out unused data
        if self.npl != 'AUTO':
            idc = (pos[:, 0] >= -self.npl*self.pxs/2.) & (pos[:, 0] <= self.npl*self.pxs/2.) &\
                (pos[:, 1] >= -self.npl*self.pxs/2.) & (pos[:, 1] <= self.npl*self.pxs/2.)
            pos = pos[idc]
            Kszdata = Kszdata[idc]
            if hsml is not None:
                hsml = hsml[idc]

        if self.SD == 2:
            self.bdata = SPH_smoothing(Kszdata, pos[:, :2], self.pxs, hsml=hsml,
                                       neighbors=self.ngb, pxln=self.npl, Ncpu=self.ncpu,
                                       kernel_name=self.sph_kn)
        else:
            self.bdata = SPH_smoothing(Kszdata, pos, self.pxs, hsml=hsml, neighbors=self.ngb,
                                       pxln=self.npl, Ncpu=self.ncpu, kernel_name=self.sph_kn)
            self.bdata = np.sum(self.bdata, axis=2)
        self.bdata = self.bdata.T
        self.bdata /= self.pxs**2

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

        hdu = pf.PrimaryHDU(self.bdata)
        hdu.header["RCVAL1"] = float(self.cc[0])
        hdu.header["RCVAL2"] = float(self.cc[1])
        hdu.header["RCVAL3"] = float(self.cc[2])
        hdu.header["UNITS"] = "kpc"
        hdu.header["RADIUS"] = float(self.rr)
        hdu.header["REDSHIFT"] = float(self.red)
        hdu.header["PSIZE"] = float(self.pxs)
        if self.ar is None:
            hdu.header["AGLRES"] = 0
        else:
            hdu.header["AGLRES"] = float(self.ar)
        hdu.header["NOTE"] = ""
        hdu.writeto(fname, clobber=clobber)
