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


# class TH_model(object):
#     r""" Theoretical calculation of y and T_sz -map for the thermal SZ effect.
#     model = TH_model(model_file, npixel, axis)
#
#     Parameters
#     ----------
#     simudata : the simulation data from load_data
#
#     Returns
#     -------
#     Theoretical projected y-map in a given direction. 2D mesh data right now.
#
#     See also
#     --------
#     SZ_models for the mock SZ signal at different frequencies.
#
#     Notes
#     -----
#
#
#     Example
#     -------
#     mm=pymsz.TH_models(simudata, 1024, "z")
#     """
#
#     def __init__(self, simudata, yt_nref=None):
#         self.ydata = np.array([])
#         if simudata.data_type == "snapshot":
#             self.Tszdata = np.array([])
#             # self.ned = np.array([])        # electron number density
#             self._prep_snap(simudata)
#         elif simudata.data_type == "yt_data":
#             if yt_nref is not None:
#                 simudata.data.ds.n_ref = yt_nref
#             self._prep_yt(simudata)
#         else:
#             raise ValueError("Do not accept this data type %s"
#                              "Please try to use load_data to get the data" % simudata.data_type)

def TH_ymap(simd, npixel=500, neighbours=None, axis='z', AR=None, redshift=None):
    r"""
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
    """

    if simd.data_type == "snapshot":
        Kpc = 3.0856775809623245e+21  # cm
        simd.prep_snap()

        if redshift is None:
            redshift = simd.cosmology['z']

        pos = rotate_data(simd.pos, axis)
        minx = pos[:, 0].min()
        maxx = pos[:, 0].max()
        miny = pos[:, 1].min()
        maxy = pos[:, 1].max()

        if isinstance(simd.hsml, type(0)):
            neighbours = 27

        # smearing the Tsz data using SPH with respected to the smoothing length
        from scipy.spatial import cKDTree
        if AR is None:
            pixelsize = np.min([maxx - minx, maxy - miny]) / npixel
            pixelsize *= (1 + 1.0e-6)
            nx = np.int32((maxx - minx) / pixelsize) + 1
            ny = np.int32((maxy - miny) / pixelsize) + 1
            ydata = np.zeros((nx, ny), dtype=np.float32)
            x = np.arange(minx, maxx, pixelsize)
            y = np.arange(miny, maxy, pixelsize)
        else:
            if redshift <= 0.0:
                redshift = 0.02
            if simd.cosmology['omega_matter'] != 0:
                cosmo = FlatLambdaCDM(H0=simd.cosmology['h']
                                      * 100, Om0=simd.cosmology['omega_matter'])
            else:
                cosmo = WMAP7
            pixelsize = cosmo.arcsec_per_kpc_proper(redshift) * AR * simd.cosmology['h']
            ydata = np.zeros((npixel, npixel), dtype=np.float32)
            x = np.arange(minx, maxx, pixelsize)
            y = np.arange(miny, maxy, pixelsize)

        if neighbours is not None:
            hsml = neighbours**(1.0 / 3) * pixelsize
        x, y = np.meshgrid(x, y)
        mtree = cKDTree(np.append(x.reshape(x.size, 1), y.reshape(y.size, 1), axis=1))
        if neighbours is not None:
            dist, ids = mtree.query(pos, neighbours)
        for i in np.arange(simd.Tszdata.size):
            if neighbours is not None:
                wsph = SPH(dist[i] / hsml)
            else:
                ids = mtree.query_ball_point(pos[i], simd.hsml[i])
                # if isinstance(ids, type(0)):  # int object
                #     ids = np.array([ids])
                if len(ids) == 0:
                    wsph = np.array([1])
                    dist, ids = mtree.query(pos[i], k=1)
                else:
                    dist = np.sqrt((pos - mtree.data[ids, 0]) **
                                   2 + (pos - mtree.data[ids, 1])**2)
                    wsph = SPH(dist / simd.hsml[i])
            xx = np.int32((mtree.data[ids, 0] - minx) / pixelsize)
            yy = np.int32((mtree.data[ids, 1] - miny) / pixelsize)
            ydata[xx, yy] += simd.Tszdata[i] * wsph / wsph.sum()
        ydata *= pixelsize * Kpc / simd.cosmology["h"]
    elif simd.data_type == "yt_data":
        from yt.units import cm
        Ptype = simd.prep_yt()
        if redshift is None:
            redshift = simd.yt_ds.current_redshift
        if AR is None:
            rr = 2. * simd.radius
        else:
            if redshift <= 0.0:
                redshift = 0.02

            if simd.yt_ds.omega_matter != 0:
                cosmo = FlatLambdaCDM(H0=simd.yt_ds.hubble_constant
                                      * 100, Om0=simd.yt_ds.omega_matter)
            else:
                cosmo = WMAP7
            pixelsize = cosmo.arcsec_per_kpc_proper(redshift) * AR * simd.yt_ds.hubble_constant
            rr = npixel * pixelsize
        if isinstance(axis, type('x')):
            projection = simd.yt_ds.proj(('deposit', Ptype + '_smoothed_Tsz'), axis,
                                         center=simd.center, data_source=simd.yt_sp)
            FRB = projection.to_frb(rr, npixel)
            ydata = FRB[('deposit', Ptype + '_smoothed_Tsz')] * cm
    else:
        raise ValueError("Do not accept this data type %s"
                         "Please try to use load_data to get the data" % simd.data_type)

    return ydata
