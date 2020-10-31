import numpy as np
from pymsz.rotate_data import rotate_data, SPH_smoothing
from astropy.cosmology import FlatLambdaCDM, WMAP7
from astropy.coordinates import SkyCoord
from astropy.time import Time
from pymsz import version
# scipy must >= 0.17 to properly use this!
# from scipy.stats import binned_statistic_2d


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
                The AR will be recalculated if z > 0.
                If z = 0, AR is set to 1. Note this makes no sense to the ICRS coordinates.
    SD       : dimensions for SPH smoothing. Type: int. Default: 2.
                Must be 2 or 3!
    SP       : Faked sky positions in [RA (longitude), DEC (latitude)] in degrees.
                Default: [194.95, 27.98], Coma' position.
    #            Need to find sometime to add lower functions
    #            If [x,y,z] (len(SP) == 3) of the Earth position in the simulation coordinate (in the same units) is given,
    #            The pos - [x,y,z] are taken as the J2000 3D coordinates and converted into RA, DEC.
    # pxsize   : physical/proper pixel size of the image. Type: float, unit: kpc.
    #             Default: None
    #             If set, this will invaided the calculation from AR.
    Memreduce: Try to reduce the memory in parallel calculation by passing the cKDTree class in SPH_smoothing. This overcomes the 
                memory cost in the cKDTree query, but should require Python>3.8 to pass the class > 4Gb.
                Default: False.
    Ncpu     : number of CPU for parallel calculation. Type: int. Default: None, all cpus from the
                computer will be used.
                Note, this parallel calculation is only for the SPH smoothing.
    Ntasks   : number of tasks for separating the calculation. Type: int. Default: None,
                the same number of Ncpu will be used. Ideally, it should be larger than or equal to the number of Ncpu.
                Set this to a much larger value if you encounter an 'IO Error: bad message length'
    # periodic : periodic condition applied for the SPH smoothing region. Tyep: bool. Default: False.
    #             periodic condition works for the too fine mesh (which means oversmoothing),
    #             you can turn this on to avoid small boundary effects. So, this is only for SPH.
    redshift : The redshift where we put the cluster for observation,.
                Default : None, we will look it from simulation data.
                Note : If redshift = 0, the returning results will be y_int, i.e. y*d^2_A,
                which takes out the angular diameter distance.
                This will also ingore the set of AR. The image pixel size =
                2 * cluster radius/npixel, so the npixel MUST NOT be 'AUTO' at redshift = 0.
                Highly recommended to *NOT* put the cluster at z = 0.
                Note the physical positions of particles are always at the simulation time.
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
    This program does not accpte redshift=0 or AR == None with npixel = "AUTO", the npixel is reset to 500.
    If npixel is "AUTO" and AR != None, then the whole cluster will project to the image with
    npixel = 2*radius/pixelsize, where pxielsize is given by AR.
    If z = 0 and AR == None, AR will be reset to 1. Note this makes no sense to the ICRS coordinates.

    Example
    -------
    mm=pymsz.TT_models(simudata, npixel=1024, "z")
    """

    def __init__(self, simudata, npixel=500, neighbours=None, axis='z', AR=None, SD=2, SP=[194.95, 27.98], 
                Memreduce=False, Ncpu=None, Ntasks=None, redshift=None, zthick=None, sph_kernel='cubic'):
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

        if self.ar is None and self.npl == 'auto':
            print("Do not accept AR == None and npixel=='AUTO' !! \n The npixel is reset to 500.!")
            self.npl = 500
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

        if self.red <= 0. and self.npl == 'auto':
            print("Do not accept redshift == 0 and npixel=='AUTO' !!\n The npixel is reset to 500.!")
            self.npl = 500

        self.cc = simd.center/simd.cosmology['h']/(1+simd.cosmology['z'])
        self.rr = simd.radius/simd.cosmology['h']/(1+simd.cosmology['z'])
        pos = rotate_data(simd.pos/simd.cosmology['h']/(1+simd.cosmology['z']), self.ax)[0]  # to proper distance
        if self.zthick is not None:
            self.zthick = self.zthick/simd.cosmology['h']/(1+simd.cosmology['z'])
            idc = (pos[:, 2] > -self.zthick) & (pos[:, 2] < self.zthick)
            pos = pos[idc]
            Tszdata = simd.Tszdata[idc]
        else:
            Tszdata = np.copy(simd.Tszdata)

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

        if self.SD == 2:
            self.ydata = SPH_smoothing(Tszdata, pos[:, :2], self.pxs, ngbinp, hsml=hsml,
                                       pxln=self.npl, Memreduce=self.Memreduce, Ncpu=self.ncpu,
                                       Ntasks=self.ntask, kernel_name=self.sph_kn)
        else:
            # be ware that zthick could cause some problems if it is larger than pxs*npl!!
            # This has been taken in care in the rotate_data function.
            self.ydata = SPH_smoothing(Tszdata, pos, self.pxs, ngbinp, hsml=hsml,
                                       pxln=self.npl, Memreduce=self.Memreduce, Ncpu=self.ncpu,
                                       Ntasks=self.ntask, kernel_name=self.sph_kn)
            self.ydata = np.sum(self.ydata, axis=2)
        self.ydata = self.ydata.T / self.pxs**2

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

    def write_fits_image(self, fname, comments="None", overwrite=False):
        r"""
        Generate a image by binning X-ray counts and write it to a FITS file.

        Parameters
        ----------
        imagefile : string
            The name of the image file to write.
        comments : The comments in str will be put into the fit file header. Defualt: 'None'
                    It accepts str or list of str or tuple of str
        overwrite : boolean, optional
            Set to True to overwrite a previous file.
        """
        # import pyfits as pf
        import astropy.io.fits as pf

        if fname[-5:] != ".fits":
            fname = fname + ".fits"

        hdu = pf.PrimaryHDU(self.ydata)
        hdu.header["SIMPLE"] = 'T'
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
    SP       : Faked sky positions in [RA (longitude), DEC (latitude)] in degrees.
                Default: [194.95, 27.98], Coma' position.
    #            Need to find sometime to add lower functions
    #            If [x,y,z] (len(SP) == 3) of the Earth position in the simulation coordinate (in the same units) is given,
    #            The pos - [x,y,z] are taken as the J2000 3D coordinates and converted into RA, DEC.
    neighbours: this parameter only works with simulation data (not yt data).
                If this is set, it will force the SPH particles smoothed into nearby N
                neighbours, HSML from the simulation will be ignored.
                If no HSML provided in the simulation, neighbours = 27
    AR       : angular resolution in arcsec. Default : None.
                Cluster's redshift with AR decides the image pixel size.
                If None, the whole cluster will be projected to the image with npixel resolution.
                The AR will be recalculated if z != 0.
                If z = 0, AR is set to 1. Note this makes no sense to the ICRS coordinates.
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
    redshift : The redshift where we put the cluster for observation.
                Default : None, we will look it from simulation data.
                Note : If redshift = 0, the returning results will be y_int, i.e. y*d^2_A,
                which takes out the angular diameter distance.
                This will also ingore the set of AR. The image pixel size =
                2 * cluster radius/npixel, so the npixel MUST NOT be 'AUTO' at redshift = 0.
                Highly recommended to put the cluster *NOT* at z = 0.
                Note the physical positions of particles are always at the simulation time.
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
    This program does not accpte redshift=0 or AR == None with npixel = "AUTO",
    the npixel will be reset to 500 for output.
    If npixel is "AUTO" and AR != None, then the whole cluster will project to the image with
    npixel = 2*radius/pixelsize, where pxielsize is given by AR.
    If redshift=0 and AR == None, AR will be reset to 1, but this does not make a sense to the ICRS system!

    Example
    -------
    mm=pymsz.TK_models(simudata, npixel=1024, "z")
    mm.bdata  # this contains the b-map
    """

    def __init__(self, simudata, npixel=500, neighbours=None, axis='z', AR=None, SD=2,
                 SP=[194.95, 27.98], Ncpu=None, redshift=None, zthick=None, sph_kernel='cubic'):
        if isinstance(npixel, type("")) or isinstance(npixel, type('')):
            self.npl = npixel.lower()
        else:
            self.npl = npixel
        self.ngb = neighbours
        self.ax = axis
        self.ar = AR
        self.ncpu = Ncpu
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
            self._cal_yt(simudata)
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
                                       kernel_name=self.sph_kn)
        else:
            self.bdata = SPH_smoothing(Kszdata, pos, self.pxs, self.ngb, hsml=hsml,
                                       pxln=self.npl, Memreduce=self.Memreduce, Ncpu=self.ncpu, 
                                       kernel_name=self.sph_kn)
            self.bdata = np.sum(self.bdata, axis=2)
        self.bdata = self.bdata.T / self.pxs**2

    def write_fits_image(self, fname, comments='None', overwrite=False):
        r"""
        Generate a image by binning X-ray counts and write it to a FITS file.

        Parameters
        ----------
        imagefile : string
            The name of the image file to write.
        radius : float, in unit of kpc
            The virial radius of the cluster. Default is the physical size of the image.
        comments : The comments in str will be put into the fit file header. Defualt: 'None'
                    It accepts str or list of str or tuple of str
        overwrite : boolean, optional
            Set to True to overwrite a previous file.
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
