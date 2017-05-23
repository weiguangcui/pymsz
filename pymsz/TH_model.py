# from scipy.interpolate import interp1d
import numpy as np
from rotate_data import rotate_data
# import pyfits
# scipy must >= 0.17 to properly use this!
# from scipy.stats import binned_statistic_2d
try:
    from yt.utilities.physical_constants import mp, kb, cross_section_thompson_cgs, \
        solar_mass, mass_electron_cgs, speed_of_light_cgs
except ImportError:
    mp = 1.67373522381e-24  # proton mass in g
    kb = 1.3806488e-16      # Boltzman constants in erg/K
    cs = 6.65245854533e-25  # cross_section_thompson in cm**2
    M_sun = 1.98841586e+33  # g
    Kpc = 3.0856775809623245e+21  # cm
    me = 9.10938291e-28     # g
    c = 29979245800.0      # cm/s
else:
    mp = mp.v
    kb = kb.v
    cs = cross_section_thompson_cgs.v
    M_sun = solar_mass.v
    me = mass_electron_cgs.v
    c = speed_of_light_cgs.v
    from yt.utilities.physical_ratios import cm_per_kpc as Kpc


def SPH(x, h):
    if (x > 0) and (x <= 0.5):
        return 8*(1 - 6*x**2 + 6*x**3)/np.pi/h**3
    elif (x > 0.5) and (x <= 1):
        return 16*(1 - x)**3/np.pi/h**3
    else:
        return 0


class TH_model(object):
    r""" Theoretical calculation of y and T_sz -map for the thermal SZ effect.
    model = TH_model(model_file, npixel, axis)

    Parameters
    ----------
    simudata : the simulation data from load_data
    npixel   : number of pixels for your image, int. Assume that x-y have the same number of pixels
    axis     : can be 'x', 'y', 'z', or a list of degrees [alpha, beta, gamma],
               which will rotate the data points by $\alpha$ around the x-axis,
               $\beta$ around the y-axis, and $\gamma$ around the z-axis
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

    def __init__(self, simudata, npixel, axis):
        self.Tszdata = np.array([])
        self.ned = np.array([])        # electron number density
        self.np = npixel
        self.ydata = np.array([])

        if simudata.data_type == "snapshot":
            simudata.pos = rotate_data(simudata.pos, axis)
            self.pixelsize = np.min(np.max(simudata.pos[:, :2], axis=0) - np.min(simudata.pos[:, :2], axis=0))/self.np
            self.nx = np.int32((simudata.pos[:, 0].max() - simudata.pos[:, 0].min())/self.pixelsize+1)
            self.ny = np.int32((simudata.pos[:, 1].max() - simudata.pos[:, 1].min())/self.pixelsize+1)
            self.ydata = np.zeros((self.nx, self.ny), dtype=np.float32)
            self._ymap_snap(simudata)
        elif simudata.data_type == "yt_data":
            self._ymap_yt(simudata)
        else:
            raise ValueError("Do not accept this data type %s"
                             "Please try to use load_data to get the data" % simudata.data_type)

    def _ymap_snap(self, simudata):  # Now everything need to be in physical
        self.ned = simudata.ne/mp * (1.0e10*M_sun*simudata.cosmology["h"]**2/Kpc**3)
        self.Tszdata = self.ned*kb*simudata.temp*cs/me/c**2
        # smearing the data using SPH with respected to the smoothing length
        from scipy.spatial import cKDTree
        x = np.linspace(simudata.pos[:, 0].min(), simudata.pos[:, 0].max(), self.nx-1)
        y = np.linspace(simudata.pos[:, 1].min(), simudata.pos[:, 1].max(), self.ny-1)
        x, y = np.meshgrid(x, y)
        x = x.reshape(x.size, 1)
        y = y.reshape(y.size, 1)
        mtree = cKDTree(np.append(x, y, axis=1))
        for i in np.arange(self.Tszdata.size):
            dist, ids = mtree.query(simudata.pos[i, :2], simudata.hsml[i])
            if isinstance(ids, type(0)):  # int object
                self.ydata[ids % self.nx, ids/self.nx] += self.Tszdata[i] * \
                    SPH(dist/simudata.hsml[i], simudata.hsml[i])
            else:
                for j, n in enumerate(ids):
                    # d = np.sqrt((simudata.pos[i, 0] - x[j])**2 + (simudata.pos[i, 1] - y[j])**2)
                    self.ydata[n % self.nx, n/self.nx] += self.Tszdata[i] * \
                            SPH(dist[j]/simudata.hsml[i], simudata.hsml[i])

    def _ymap_yt(self, simudata):
        self.ned = simudata[("Gas", "NE")]

    # def ymap(self, file, IMF):
    #     r""" calculating of SZ y-map.
    #     TH_model.ymap(nsample=None)
    #
    #     Parameters
    #     ----------
    #     model_file : File name for the SSP models. Only "name" + _ + "model",
    #                  such as c09_exp, or bc03_ssp. The whole name of the model
    #                  must be name_model_z_XXX_IMF.model (or .ised, .txt, .fits)
    #
    #     Returns
    #     -------
    #     The theoretical y-map from the SZ effect.
    #
    #     See also
    #     --------
    #     T_sz function in this class
    #
    #     Notes
    #     -----
    #     ...
    #
    #     Example
    #     -------
    #     y-map = TH_model.ymap(nsample=None)
    #     """
    #
    #     return y_map

    # def T_sz(self, file):
    #     r""" calculating of SZ y-map.
    #     TH_model.ymap(nsample=None)
    #
    #     Parameters
    #     ----------
    #     model_file : File name for the SSP models. Only "name" + _ + "model",
    #                  such as c09_exp, or bc03_ssp. The whole name of the model
    #                  must be name_model_z_XXX_IMF.model (or .ised, .txt, .fits)
    #
    #     Returns
    #     -------
    #     The theoretical y-map from the SZ effect.
    #
    #     See also
    #     --------
    #     T_sz function in this class
    #
    #     Notes
    #     -----
    #     ...
    #
    #     Example
    #     -------
    #     y-map = TH_model.ymap(nsample=None)
    #     """
    #
    #     return Tsz
    #
    # def get_seds(self, simdata, dust_func=None, units='Fv'):
    #     r"""
    #     Seds = SSP_model(simdata, dust_func=None, units='Fv')
    #
    #     Parameters
    #     ----------
    #     simudata   : Simulation data read from load_data
    #     dust_func  : dust function.
    #     units      : The units for retruned SEDS. Default: 'Fv'
    #
    #     Get SEDS for the simulated star particles
    #
    #     Available output units are (case insensitive):
    #
    #     ========== ====================
    #     Name       Units
    #     ========== ====================
    #     Jy         Jansky
    #     Fv         ergs/s/cm^2/Hz
    #     Fl         ergs/s/cm^2/Angstrom
    #     Flux       ergs/s/cm^2
    #     Luminosity ergs/s
    #     ========== ====================
    #     """
    #
    #     seds = np.zeros((self.nvs[0], simdata.S_age.size), dtype=np.float64)
    #
    #     # We do not do 2D interpolation since there is only several metallicity
    #     # in the models.
    #     if self.nmets > 1:
    #         mids = np.interp(simdata.S_metal, self.metals,
    #                          np.arange(self.nmets))
    #         mids = np.int32(np.round(mids))
    #
    #     for i, metmodel in enumerate(self.met_name):
    #         f = interp1d(self.ages[metmodel], self.seds[metmodel])
    #
    #         if self.nmets > 1:
    #             ids = mids == i
    #         else:
    #             ids = np.ones(simdata.S_metal.size, dtype=np.bool)
    #
    #         if dust_func is None:
    #             seds[:, ids] = f(simdata.S_age[ids]) * simdata.S_mass[ids]
    #         else:
    #             seds[:, ids] = f(simdata.S_age[ids]) * simdata.S_mass[ids] * \
    #                 dust_func(simdata.S_age[ids], self.ls[metmodel])
    #
    #     if simdata.grid_mass is not None:
    #         seds = binned_statistic_2d(simdata.S_pos[:, 0], simdata.S_pos[:, 1],
    #                                    values=seds,
    #                                    bins=[simdata.nx, simdata.nx],
    #                                    statistic='sum')[0]
