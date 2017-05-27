import numpy as np
from rotate_data import rotate_data
from astropy.cosmology import FlatLambdaCDM, WMAP7
# import pyfits
# scipy must >= 0.17 to properly use this!
# from scipy.stats import binned_statistic_2d
try:
    from yt.utilities.physical_constants import mp, kb, cross_section_thompson_cgs, \
        solar_mass, mass_electron_cgs, speed_of_light_cgs
except ImportError:
    Mp = 1.67373522381e-24  # proton mass in g
    Kb = 1.3806488e-16      # Boltzman constants in erg/K
    cs = 6.65245854533e-25  # cross_section_thompson in cm**2
    M_sun = 1.98841586e+33  # g
    Kpc = 3.0856775809623245e+21  # cm
    me = 9.10938291e-28     # g
    c = 29979245800.0      # cm/s
else:
    Mp = mp.v
    Kb = kb.v
    cs = cross_section_thompson_cgs.v
    M_sun = solar_mass.v
    me = mass_electron_cgs.v
    c = speed_of_light_cgs.v
    from yt.utilities.physical_ratios import cm_per_kpc as Kpc
    from yt.units import cm


def Ele_num_den(field, data):
    # if ("Gas", "ElectronAbundance") in data.ds.field_info:
    return data[field.name[0], "Density"] * data[field.name[0], "ElectronAbundance"] * \
        (1 - data[field.name[0], "Z"] - 0.24) / mp
    # else:  # Assume full ionized
    # return data[field.name[0], "Density"] * 1.351 * (1 - data[field.name[0],
    # "Z"] - 0.24) / mp


def Temp_SZ(field, data):
    return data[field.name[0], "END"] * data['Gas', 'Temperature'] * kb * cross_section_thompson_cgs / \
        mass_electron_cgs / speed_of_light_cgs**2


def SPH(x, h):  # 3D
    data = np.zeros(x.size, dtype=float)
    ids = (x > 0) & (x <= 0.5)
    data[ids] = 8 * (1 - 6 * x[ids]**2 + 6 * x[ids]**3) / np.pi / h**3
    ids = (x > 0.5) & (x < 1)
    data[ids] = 16 * (1 - x[ids])**3 / np.pi / h**3
    return data
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

    def __init__(self, simudata, yt_nref=None):
        self.ydata = np.array([])
        if simudata.data_type == "snapshot":
            self.Tszdata = np.array([])
            # self.ned = np.array([])        # electron number density
            self._prep_snap(simudata)
        elif simudata.data_type == "yt_data":
            if yt_nref is not None:
                simudata.data.ds.n_ref = yt_nref
            self._prep_yt(simudata)
        else:
            raise ValueError("Do not accept this data type %s"
                             "Please try to use load_data to get the data" % simudata.data_type)

    def _prep_snap(self, simudata):  # Now everything need to be in physical
        simudata.ne = simudata.ne / Mp * \
            (1.0e10 * M_sun * simudata.cosmology["h"]**2 / Kpc**3)  # now in cm^-3
        self.Tszdata = simudata.ne * Kb * simudata.temp * cs / me / c**2  # now in cm^-1

    def _prep_yt(self, simudata):
        if 'PGas' in simudata.data.ds.particle_types:
            self.Ptype = 'PGas'
        else:
            self.Ptype = 'Gas'
        simudata.data.ds.add_field((self.Ptype, "END"), function=Ele_num_den,
                                   sampling_type="particle", units="cm**(-3)")
        simudata.data.ds.add_field((self.Ptype, "Tsz"), function=Temp_SZ,
                                   sampling_type="particle", units="1/cm")
        simudata.data.ds.add_smoothed_particle_field((self.Ptype, "Tsz"))

    def get_ymap(self, simd, npixel=500, axis='z', AR=None, redshift=None):
        r"""
            simudata : the simulation data from load_data
            npixel   : number of pixels for your image, int. Assume that x-y have the same number of pixels
            axis     : can be 'x', 'y', 'z', or a list of degrees [alpha, beta, gamma],
                       which will rotate the data points by $\alpha$ around the x-axis,
                       $\beta$ around the y-axis, and $\gamma$ around the z-axis
            AR       : angular resolution in arcsec. Default : None, which gives npixel = 2 * cluster radius
                        and ignores the cluster's redshift.
                        Otherwise, the cluster's redshift with AR decides how many pixels the cluster occupies.
            redshift : The redshift where the cluster is at. Default : None, we will look it from simulation data.
                        If redshift = 0, it will automatically put into 0.02, unless you set AR = None.
        """
        self.np = npixel
        self.ax = axis
        self.ar = AR
        self.red = redshift

        if simd.data_type == "snapshot":
            if simd.cosmology['omega_matter'] != 0:
                cosmo = FlatLambdaCDM(H0=simd.cosmology['h']
                                      * 100, Om0=simd.cosmology['omega_matter'])
            else:
                cosmo = WMAP7
            if self.red is None:
                self.red = simd.cosmology['z']

            simd.pos = rotate_data(simd.pos, self.ax)
            minx = simd.pos[:, 0].min()
            maxx = simd.pos[:, 0].max()
            miny = simd.pos[:, 1].min()
            maxy = simd.pos[:, 1].max()

            # smearing the Tsz data using SPH with respected to the smoothing length
            from scipy.spatial import cKDTree
            if self.ar is None:
                self.pixelsize = np.min([maxx - minx, maxy - miny]) / self.np
                self.pixelsize *= (1 + 1.0e-6)
                self.nx = np.int32((maxx - minx) / self.pixelsize) + 1
                self.ny = np.int32((maxy - miny) / self.pixelsize) + 1
                self.ydata = np.zeros((self.nx, self.ny), dtype=np.float32)
                x = np.arange(minx, maxx, self.pixelsize)
                y = np.arange(miny, maxy, self.pixelsize)
            else:
                if self.red <= 0.0:
                    self.red = 0.02
                self.pixelsize = cosmo.arcsec_per_kpc_proper(
                    self.red) * self.ar * simd.cosmology['h']
                self.ydata = np.zeros((self.np, self.np), dtype=np.float32)
                x = np.arange(minx, maxx, self.pixelsize)
                y = np.arange(miny, maxy, self.pixelsize)

            x, y = np.meshgrid(x, y)
            mtree = cKDTree(np.append(x.reshape(x.size, 1), y.reshape(y.size, 1), axis=1))
            for i in np.arange(self.Tszdata.size):
                ids = mtree.query_ball_point(simd.pos[i, :2], simd.hsml[i])
                if isinstance(ids, type(0)):  # int object
                    ids = np.array([ids])
                dist = np.sqrt((simd.pos[i, 0] - mtree.data[ids, 0]) **
                               2 + (simd.pos[i, 1] - mtree.data[ids, 1])**2)
                xx = np.int32((mtree.data[ids, 0] - minx) / self.pixelsize)
                yy = np.int32((mtree.data[ids, 1] - miny) / self.pixelsize)
                wsph = SPH(dist / simd.hsml[i], simd.hsml[i])
                self.ydata[xx, yy] += self.Tszdata[i] * wsph / wsph.sum()
            self.ydata *= self.pixelsize * Kpc / simd.cosmology["h"]
        elif simd.data_type == "yt_data":
            if self.red is None:
                self.red = simd.data.ds.current_redshift
            if self.ar is None:
                rr = 2. * simd.radius
            else:
                if self.red <= 0.0:
                    self.red = 0.02
                self.pixelsize = cosmo.arcsec_per_kpc_proper(
                    self.red) * self.ar * simd.cosmology['h']
                rr = self.np * self.pixelsize
            if isinstance(self.ax, type('x')):
                projection = simd.data.ds.proj(('deposit', self.Ptype + '_smoothed_Tsz'), self.ax,
                                               center=simd.center, data_source=simd.data)
                FRB = projection.to_frb(rr, self.np)
                self.ydata = FRB[('deposit', self.Ptype + '_smoothed_Tsz')] * cm
        else:
            raise ValueError("Do not accept this data type %s"
                             "Please try to use load_data to get the data" % simd.data_type)

        return self.ydata
