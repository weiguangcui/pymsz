import numpy as np
from rotate_data import rotate_data
from astropy.cosmology import FlatLambdaCDM, WMAP7
# import pyfits
# scipy must >= 0.17 to properly use this!
# from scipy.stats import binned_statistic_2d


def SPH(x):  # 3D Cubic
    data = np.zeros(x.size, dtype=float)
    ids = (x > 0) & (x <= 0.5)
    data[ids] = 1 - 6 * x[ids]**2 + 6 * x[ids]**3
    ids = (x > 0.5) & (x < 1)
    data[ids] = 2 * (1 - x[ids])**3
    return data * 2.5464790894703255
# def SPH(x, h):  # 2D
#     if (x > 0) and (x <= 0.5):
#         return 10*(1 - 3*x**2/2. + 3*x**3/4.)/7./np.pi/h**2
#     elif (x > 1) and (x <= 2):
#         return 10*(2 - x)**3/4./7./np.pi/h**2
#     else:
#         return 0


class TH_model(object):
    r""" Theoretical calculation of y and T_sz -map for the thermal SZ effect.
    model = TH_model(model_file, npixel, axis)

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
                Default : None, which gives npixel = 2 * cluster radius
                and ignores the cluster's redshift.
                Otherwise, cluster's redshift with AR decides how large the cluster looks.
    redshift : The redshift where the cluster is at.
                Default : None, we will look it from simulation data.
                If redshift = 0, it will be automatically put into 0.02,
                unless AR is set to None.

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
    mm=pymsz.TH_models(simudata, 1024, "z")
    """

    def __init__(self, simudata, npixel=500, neighbours=None, axis='z', AR=None, redshift=None):
        self.npl = npixel
        self.ngb = neighbours
        self.ax = axis
        self.ar = AR
        self.red = redshift
        self.pxs = 0
        self.ydata = np.array([])

        if simudata.data_type == "snapshot":
            self._cal_snap(simudata)
        elif simudata.data_type == "yt_data":
            self._cal_yt(simudata)
        else:
            raise ValueError("Do not accept this data type %s"
                             "Please try to use load_data to get the data" % simudata.data_type)

    # def TH_ymap(simd, npixel=500, neighbours=None, axis='z', AR=None, redshift=None):

    def _cal_snap(self, simd):
        Kpc = 3.0856775809623245e+21  # cm
        simd.prep_snap()

        if self.red is None:
            self.red = simd.cosmology['z']

        pos = rotate_data(simd.pos, self.ax)
        if simd.radius is not None:
            idc = (pos[:, 2] > -1 * simd.radius) & (pos[:, 2] <= simd.radius) & \
                  (pos[:, 0] > -1 * simd.radius) & (pos[:, 0] <= simd.radius) & \
                  (pos[:, 1] > -1 * simd.radius) & (pos[:, 1] <= simd.radius)
            pos = pos[idc]
        minx = pos[:, 0].min()
        maxx = pos[:, 0].max()
        miny = pos[:, 1].min()
        maxy = pos[:, 1].max()
        minz = pos[:, 2].min()
        maxz = pos[:, 2].max()
        Tszdata = simd.Tszdata[idc]
        if isinstance(simd.mass, type(0.0)):
            mass = simd.mass
        else:
            mass = simd.mass[idc]
        dens = simd.rho[idc]

        if isinstance(simd.hsml, type(0)):
            self.ngb = 27
        else:
            hsml = simd.hsml[idc]

        # smearing the Tsz data using SPH with respected to the smoothing length
        from scipy.spatial import cKDTree
        if self.ar is None:
            self.pxs = np.min([maxx - minx, maxy - miny, maxz - minz]) / self.npl
            self.pxs *= (1 + 1.0e-6)
            nx = np.int32((maxx - minx) / self.pxs) + 1
            ny = np.int32((maxy - miny) / self.pxs) + 1
            nz = np.int32((maxz - minz) / self.pxs) + 1
            self.ydata = np.zeros((nx, ny, nz), dtype=np.float32)
            x = np.arange(minx, maxx, self.pxs)
            y = np.arange(miny, maxy, self.pxs)
            z = np.arange(minz, maxz, self.pxs)
        else:
            if self.red <= 0.0:
                self.red = 0.02
            if simd.cosmology['omega_matter'] != 0:
                cosmo = FlatLambdaCDM(H0=simd.cosmology['h']
                                      * 100, Om0=simd.cosmology['omega_matter'])
            else:
                cosmo = WMAP7
            self.pxs = cosmo.arcsec_per_kpc_proper(self.red) * self.ar * simd.cosmology['h']
            self.ydata = np.zeros((self.npl, self.npl, self.npl), dtype=np.float32)
            x = np.arange(minx, maxx, self.pxs)
            y = np.arange(miny, maxy, self.pxs)
            z = np.arange(minz, maxz, self.pxs)

        # if self.ngb is not None:
        #     hsml = np.sqrt(self.ngb) * self.pxs
        x, y, z = np.meshgrid(x, y, z, indexing='ij')
        xyz = np.concatenate((x.reshape(x.size, 1), y.reshape(
            y.size, 1), z.reshape(z.size, 1)), axis=1)
        mtree = cKDTree(pos)
        if self.ngb is not None:
            dist, idst = mtree.query(xyz, self.ngb)
        for i in np.arange(xyz.shape[0]):
            if self.ngb is not None:
                ids = idst[i]
                mxhsml = dist[i].max() / 2.0
                ihsml = hsml[ids]
                ihsml[ihsml < mxhsml] = mxhsml
                ihsml3 = 1. / ihsml**3
                if isinstance(mass, type(0.0)):
                    wsph = (mass / dens[ids]) * ihsml3
                else:
                    wsph = (mass[ids] / dens[ids]) * ihsml3
                wsph *= Tszdata[ids] * SPH(dist[i] / ihsml) * ihsml3
            else:
                ihsml3 = 1. / hsml[i]**3
                ids = mtree.query_ball_point(pos[i], hsml[i])
                # if isinstance(ids, type(0)):  # int object
                #     ids = np.array([ids])
                if len(ids) == 0:
                    wsph = np.array([1])
                    dist, ids = mtree.query(pos[i], k=1)
                else:
                    dist = np.sqrt((pos[i, 0] - mtree.data[ids, 0]) **
                                   2 + (pos[i, 1] - mtree.data[ids, 1])**2)
                    wsph = SPH(dist / hsml[i]) * ihsml3
            # xx = np.int32((mtree.data[ids, 0] - minx) / self.pxs)
            # yy = np.int32((mtree.data[ids, 1] - miny) / self.pxs)
            # zz = np.int32((mtree.data[ids, 2] - minz) / self.pxs)
            xx = np.int32((xyz[i, 0] - minx) / self.pxs)
            yy = np.int32((xyz[i, 1] - miny) / self.pxs)
            zz = np.int32((xyz[i, 2] - minz) / self.pxs)
            self.ydata[xx, yy, zz] += np.sum(wsph)  # / wsph.sum()

        self.ydata = np.sum(self.ydata, axis=2) * self.pxs * Kpc / simd.cosmology["h"]

    def _cal_yt(self, simd):
        # from yt.units import cm
        Ptype = simd.prep_yt()
        if self.red is None:
            self.red = simd.yt_ds.current_redshift
        if self.ar is None:
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

    # else:
    #     raise ValueError("Do not accept this data type %s"
    #                      "Please try to use load_data to get the data" % simd.data_type)

    # return ydata
