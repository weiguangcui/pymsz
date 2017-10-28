import numpy as np
from pymsz.readsnapsgl import readsnapsgl
# from astropy.cosmology import FlatLambdaCDM
try:
    from yt.utilities.physical_constants import mp, kb, cross_section_thompson_cgs, \
        solar_mass, mass_electron_cgs, speed_of_light_cgs, Tcmb, hcgs
    from yt.utilities.physical_ratios import cm_per_kpc as Kpc
except ImportError:
    Mp = 1.67373522381e-24  # proton mass in g
    Kb = 1.3806488e-16      # Boltzman constants in erg/K
    cs = 6.65245854533e-25  # cross_section_thompson in cm**2
    M_sun = 1.98841586e+33  # g
    Kpc = 3.0856775809623245e+21  # cm
    me = 9.10938291e-28     # g
    c = 29979245800.0       # cm/s
    tcmb = 2.726            # K
    Hcgs = 6.62606957e-27   # erg*s
    I0 = 270.211495296      # MJy/steradian
else:
    Mp = mp.v
    Kb = kb.v
    cs = cross_section_thompson_cgs.v
    M_sun = solar_mass.v
    me = mass_electron_cgs.v
    c = speed_of_light_cgs.v
    tcmb = Tcmb.v
    Hcgs = hcgs.v
    from yt.units import sr
    I0 = (2 * (kb * Tcmb)**3 / ((hcgs * speed_of_light_cgs)**2) / sr).in_units("MJy/steradian")
    I0 = I0.v


class load_data(object):
    r"""load analysing data from simulation snapshots (gadget format only),
    yt, or raw data. Currently only works with snapshot=True. Will work more
    on other data sets.

    Parameters
    ----------
    filename    : The filename of simulation snapshot, or data.
                    Type: str. Default : ''
    metal       : Gas metallicity.
                    Type: float or array. Default: None, will try to read from simulation.
                    Otherwise, will use this give metallicity.
                    It must be the same number of gas particles if it is an array.
    mu          : mean_molecular_weight.
                    Type: float. Default: None.
                    It is used for calculating gas temperature, when there is no NE block
                    from the simulation snapshot.
                    with snapshot=True, it assumes full ionized gas with mu ~ 0.588.
                    with yt_load=True, it assumes zero ionization with mu ~ 1.22.
    snapshot    : Is loading snapshot or not?
                    Type: bool. Default : False
    yt_load     : Do you want to use yt to load the data?
                    Type: bool. Default : False.
                    It requries yt models.

    specified_field: If you want to specify the data fields (for Gadget snapshots) for yt.load.
                    Default: None. This only works with yt_load = True.
                    For example:
                    My_def = ("Coordinates", "Velocities", "ParticleIDs", "Mass",
                              ("InternalEnergy", "Gas"),
                              ("Density", "Gas"),
                              ("ElectronAbundance", "Gas"),
                              ("NeutralHydrogenAbundance", "Gas"),
                              ("SmoothingLength", "Gas"),
                              ("StarFomationRate", "Gas"),
                              ("Age", ("Stars", "Bndry")),
                              ("Z", ("Gas","Stars")),
                             )
                    specified_field=My_def
    n_ref       : It governs how many particles in an oct results in that oct being
                    refined into eight child octs. Only for yt_load = True
                  Type: int. Default: None (use yt setup 64).
                  It’s recommended that if you want higher-resolution, try reducing the
                    value of n_ref to 32 or 16.

    rawdata     : Is it raw data? Default : False
                  Please look at/change the reading function in this file to load the raw data.

    ---------- If you only need parts of loaded data. Specify center and radius
    center      : The center of a sphere for the data you want to get.
                  Default : None
    radius      : The radius of a sphere for the data you want to get.
                  Default : None

    Notes
    -----
    Please be extremly careful about the units!!! Currently only assume in simulation units:
        kpc/h and 10^10 M_sun
    Raw data set needs to provide the cosmology, Otherwise WMAP7 is used for later calculation...
    center and radius need to set together in the simulation units!

    Example
    -------
    simd = load_data(snapfilename="/home/weiguang/Downloads/snap_127",
                     snapshot=True, center=[500000,500000,500000], radius=800)

    """

    def __init__(self, filename='', metal=None, mu=None, snapshot=False, yt_load=False,
                 specified_field=None, n_ref=None, datafile=False, center=None, radius=None):
        self.center = center
        self.radius = radius
        self.filename = filename
        if metal is None:
            self.metal = 0
        elif isinstance(metal, type(0.1)) or isinstance(metal, type(np.ones(1))):
            self.metal = metal
        else:
            raise ValueError("Do not accept this metal %f." % metal)
        self.mu = mu
        self.n_ref = n_ref

        if snapshot:
            self.data_type = "snapshot"
            self.temp = np.array([])
            self.mass = 0.0
            self.pos = np.array([])
            self.vel = np.array([])
            self.rho = np.array([])
            self.ne = 0
            self.hsml = 0
            self.cosmology = {}  # default wmap7

            self.Tszdata = np.array([])  # prep_ss_TT
            self.Kszdata = np.array([])  # prep_ss_KT

            self.tau = np.array([])  # prep_ss_SZT
            self.Te = np.array([])
            self.bpar = np.array([])
            self.omega = np.array([])
            self.sigma = np.array([])
            self.kappa = np.array([])
            self.bperp = np.array([])
            self.mmw = None  # mean_molecular_weight
            # self.currenta = 1.0  # z = 0
            # self.z = 0.0
            # self.Uage = 0.0  # university age in Gyrs
            # self.nx = self.grid_mass = self.grid_age = self.grid_metal = None

            self._load_snap()
        elif yt_load:
            self.data_type = "yt_data"
            self.yt_sp = None
            self.yt_ds = self._load_yt(specified_field)

        # elif datafile:
        #     self.data_type = "snapshot"
        #     self._load_raw()
        else:
            raise ValueError("Please sepecify the simulation data type. ")

    def _load_snap(self):
        head = readsnapsgl(self.filename, "HEAD", quiet=True)
        self.cosmology["z"] = head[3] if head[3] > 0 else 0.0
        self.cosmology["a"] = head[2]
        self.cosmology["omega_matter"] = head[-3]
        self.cosmology["omega_lambda"] = head[-2]
        self.cosmology["h"] = head[-1]
        # self.cosmology = FlatLambdaCDM(head[-1] * 100, head[-3])
        # self.currenta = head[2]
        # self.Uage = self.cosmology.age(1. / self.currenta - 1)
        # self.z = head[3] if head[3] > 0 else 0.0

        # positions # only gas particles
        spos = readsnapsgl(self.filename, "POS ", ptype=0, quiet=True)
        if (self.center is not None) and (self.radius is not None):
            # r = np.sqrt(np.sum((spos - self.center)**2, axis=1))
            # ids = r <= self.radius  # increase to get all the projected data
            # Now using cubic box to get the data
            ids = (spos[:,0]>=self.center[0]-self.radius) & (spos[:,0]<=self.center[0]+self.radius) &
                    (spos[:,1]>=self.center[1]-self.radius) & (spos[:,1]<=self.center[1]+self.radius) &
                    (spos[:,2]>=self.center[2]-self.radius) & (spos[:,2]<=self.center[2]+self.radius)
            self.pos = spos[ids] - self.center
        else:
            ids = np.ones(head[0][0], dtype=bool)
            self.center = np.median(spos, axis=0)
            self.pos = spos - self.center
            ids = np.ones(self.pos.shape[0], dtype=np.bool)

        # velocity
        self.vel = readsnapsgl(self.filename, "VEL ", ptype=0, quiet=True)
        if self.vel is not 0:
            self.vel = self.vel[ids] * np.sqrt(self.cosmology["a"])  # to peculiar velocity
        else:
            raise ValueError("Can't get gas velocity, which is required")
        r = np.sqrt(np.sum(self.pos**2, axis=1))
        self.vel -= np.mean(self.vel[r < 30.], axis=0)  # remove halo motion

        # Temperature
        if self.mu is None:
            self.temp = readsnapsgl(self.filename, "TEMP", quiet=True)
        else:
            self.temp = readsnapsgl(self.filename, "TEMP", mu=self.mu, quiet=True)
        if self.temp is not 0:
            self.temp = self.temp[ids]
        else:
            raise ValueError("Can't get gas temperature, which is required for this code.")

        # density
        self.rho = readsnapsgl(self.filename, "RHO ", quiet=True)
        if self.rho is not 0:
            self.rho = self.rho[ids]
        else:
            raise ValueError("Can't get gas density, which is required")

        # smoothing length
        self.hsml = readsnapsgl(self.filename, "HSML", quiet=True)
        if self.hsml is not 0:
            self.hsml = self.hsml[ids]

        # mass only gas
        self.mass = readsnapsgl(self.filename, "MASS", ptype=0, quiet=True)
        if not isinstance(self.mass, type(0.0)):
            self.mass = self.mass[ids]

        # gas metal if there are
        if self.metal is 0:
            self.metal = readsnapsgl(self.filename, "Z   ", ptype=0,
                                     quiet=True)  # auto calculate Z
            if self.metal is not 0:
                self.metal = self.metal[ids]

        # Electron fraction
        self.ne = readsnapsgl(self.filename, "NE  ", quiet=True)
        yhelium = 0.07894736842105263  # ( 1. - xH ) / ( 4 * xH )hydrogen mass-fraction (xH) = n_He/n_H
        if self.ne is not 0:
            self.ne = self.ne[ids]
            self.mmw = (1. + 4. * yhelium) / (1. + yhelium + self.ne)
        else:
            self.mmw = (1. + 4. * yhelium) / (1. + 3 * yhelium + 1)  # assume full ionized
            if self.mu is None:
                # full ionized without taking metal into account. What about metal?
                self.ne = np.ones(self.rho.size) * (4.0 / self.mmw - 3.28) / 3.04
                # (4.0 / self.mmw - 3.0 * 0.76 - 1.0) / 4.0 / 0.76
            else:
                self.ne = np.ones(self.rho.size) * (4.0 / self.mu - 3.28) / 3.04

        # Change NE (electron number fraction respected to H number density in simulation)
        # M/mmw/mp gives the total nH+nHe+ne! To get the ne, which = self.ne*nH, we reexpress this as ne*Q_NE.
        # Q_NE is given in self.ne in below.
        self.ne = (1. + yhelium + self.ne)/self.ne

        # we need to remove some spurious particles.... if there is a MHI or SRF block
        # see Klaus's doc or Borgani et al. 2003 for detials.
        mhi = readsnapsgl(self.filename, "MHI ", quiet=True)
        if mhi is 0:
            # try exclude sfr gas particles
            sfr = readsnapsgl(self.filename, "SFR ", quiet=True)
            if sfr is not 0:
                sfr = sfr[ids]
                ids_ex = sfr < 0.1
                if sfr[ids_ex].size == sfr.size:
                    ids_ex = True
            else:
                ids_ex = None
        else:
            mhi = mhi[ids] / 0.76 / self.mass
            ids_ex = (self.temp < 3.0e4) & (self.rho > 6.e-7)
            ids_ex = (mhi < 0.1) & (~ids_ex)
            if mhi[ids_ex].size == mhi.size:
                ids_ex = True
            self.rho *= (1 - mhi)  # correct multi-phase baryon model by removing cold gas

        if (ids_ex is not None) and (ids_ex is not True):
            self.temp = self.temp[ids_ex]     # cgs
            if not isinstance(self.mass, type(0.0)):
                self.mass = self.mass[ids_ex]
            self.pos = self.pos[ids_ex]
            self.vel = self.vel[ids_ex]
            self.rho = self.rho[ids_ex]
            self.ne = self.ne[ids_ex]         # cgs
            if self.metal is not 0:
                self.metal = self.metal[ids_ex]
            if self.hsml is not 0:
                self.hsml = self.hsml[ids_ex]
            else:
                self.hsml = (3 * self.mass / self.pos / 4 / np.pi)**(1. / 3.)  # approximate
            if self.mmw is not None:
                self.mmw = self.mmw[ids_ex]

    def _load_yt(self, specified_field):
        try:
            import yt
            from yt.utilities.physical_constants import mp, kb
        except ImportError:
            raise ImportError("Can not find yt package, which is required to use this function!")

        def _add_GTP(field, data):
            if ("Gas", "ElectronAbundance") in data.ds.field_info:
                mu = 4.0 / (3.28 + 3.04 * data['Gas', 'ElectronAbundance'])
            else:
                if self.mu is not None:
                    mu = self.mu
                else:
                    mu = 0.5882352941176471  # full ionized gas
            ret = data['Gas', "InternalEnergy"] * (2.0 / 3.0) * mu * mp / kb
            return ret.in_units(data.ds.unit_system["temperature"])

        def _add_GEA(field, data):  # full ionized gas ElectronAbundance
            # mu = 4.0 / (3.0 * x_H + 1.0 + 4.0 * x_H * a_e)
            if self.mu is not None:
                ae = (4.0 / self.mu - 3.28) / 3.04
            else:  # full ionized
                # (4.0 / 0.5882352941176471 - 3.0 * 0.76 - 1.0) / 4.0 / 0.76
                ae = 1.157894736842105
            return data['Gas', 'particle_ones'] * ae

        def _add_GMT(field, data):  # No metallicity
            return data['Gas', 'particle_ones'] * self.metal

        if specified_field is not None:
            from yt.frontends.gadget.definitions import gadget_field_specs
            gadget_field_specs["my_def"] = specified_field
            ds = yt.load(self.filename, field_spec="my_def")
        else:
            ds = yt.load(self.filename)

        if self.n_ref is not None:
            ds.n_ref = self.n_ref

        if ("Gas", "ElectronAbundance") not in ds.field_list:
            if self.mu is None:
                print("Add electrons as full ionized gas")
            else:
                print("Add electrons for gas with given mean_mol_weight %f" % self.mu)
            ds.add_field(("Gas", "ElectronAbundance"), function=_add_GEA,
                         sampling_type="particle", units="", force_override=True)

        # we force to re-add the gas temperature. Otherwise yt cal temperature with mu ~ 1.2
        ds.add_field(("Gas", "Temperature"), function=_add_GTP, sampling_type="particle",
                     units='K', force_override=True)

        if ("Gas", "Z") not in ds.field_list:
            print("Adding given metallicity %f" % self.metal)
            ds.add_field(("Gas", "Z"), function=_add_GMT,
                         sampling_type="particle", units="", force_override=True)

        return ds

    # def _load_raw(self):
    #     if (self.center is not None) and (self.radius is not None):
    #         r = np.sqrt(np.sum((self.filename['pos'] - self.center)**2, axis=1))
    #         ids = r <= self.radius
    #     else:
    #         ids = np.ones(self.filename['age'].size, dtype=bool)
    #     self.temp = self.filename['age'][ids]
    #     self.mass = self.filename['mass'][ids]
    #     self.metal = self.filename['metal'][ids]

    # prepare for Theoretical model calculations
    def prep_ss_TT(self, force_redo=False):  # Now everything need to be in physical
        if len(self.Tszdata) is 0 or force_redo:  # only need to prepare once
            constTsz = 1.0e10*M_sun*self.cosmology["h"]*Kb*cs/me/Mp/c**2/Kpc**2
            if self.mu is None:
                self.Tszdata = constTsz*self.mass*self.temp/self.mmw/self.ne
            else:
                self.Tszdata = constTsz*self.mass*self.temp/self.mu/self.ne
            # now Tszdata is dimensionless y_i, and can divided pixel size in kpc/h directly later

    def prep_ss_KT(self, vel, force_redo=False):
        if len(self.Kszdata) is 0 or force_redo:  # only need to prepare once
            constKsz = 1.0e15*M_sun*self.cosmology["h"]*cs/Mp/c/Kpc**2  # velocity in km/s -> cm/s
            if self.mu is None:
                self.Kszdata = constKsz*self.mass*vel/self.mmw/self.ne
            else:
                self.Kszdata = constKsz*self.mass*vel/self.mu/self.ne

    # prepare for mock observation model calculations
    def prep_ss_SZ(self, force_redo=False):
        if len(self.tau) is 0 or force_redo:
            self.tau = cs * self.rho * self.mueinv / Mp

    def prep_yt_TT(self, conserved_smooth=False, force_redo=False):
        if 'PGas' in self.yt_ds.particle_types:
            Ptype = 'PGas'
        else:
            Ptype = 'Gas'

        if (self.yt_sp is None) or force_redo:  # only need to calculate once
            import yt
            if force_redo:
                print("data fields are forced to recalculated.")

            # def Ele_num_den(field, data):
            #     # if ("Gas", "ElectronAbundance") in data.ds.field_info:
            #     return data[field.name[0], "Density"] * data[field.name[0], "ElectronAbundance"] * \
            #         (1 - data[field.name[0], "Z"] - 0.24) / mp
            #     # else:  # Assume full ionized
            #     # return data[field.name[0], "Density"] * 1.351 * (1 - data[field.name[0],
            #     # "Z"] - 0.24) / mp

            def Temp_SZ(field, data):
                const = kb * cross_section_thompson_cgs / mass_electron_cgs / speed_of_light_cgs**2 / mp
                end = data[field.name[0], "Mass"] * data[field.name[0], "ElectronAbundance"] * \
                    (1 - data[field.name[0], "Z"] - 0.24)
                return end * data[field.name[0], 'Temperature'] * const

            def MTsz(field, data):
                return data[field.name[0], 'Tsz'] * data[field.name[0], 'Mass']

            def SMWTsz(field, data):
                ret = data[field.name[0], 'Gas_smoothed_MTsz']
                ids = data[field.name[0], 'Gas_smoothed_Mass'] > 0
                ret[ids] /= data[field.name[0], 'Gas_smoothed_Mass'][ids]
                return ret

            # self.yt_ds.add_field(("Gas", "END"), function=Ele_num_den,
            #                      sampling_type="particle", units="cm**(-3)")
            self.yt_ds.add_field(("Gas", "Tsz"), function=Temp_SZ,
                                 sampling_type="particle", units="cm**2", force_override=True)
            if conserved_smooth:
                print("conserved smoothing...")
                self.yt_ds.add_field(("Gas", "MTsz"), function=MTsz,
                                     sampling_type="particle", units="g/cm", force_override=True)
                self.yt_ds.add_smoothed_particle_field(("Gas", "Mass"))
                self.yt_ds.add_smoothed_particle_field(("Gas", "MTsz"))
                self.yt_ds.add_field(('deposit', "Gas" + "_smmothed_Tsz"), function=SMWTsz,
                                     sampling_type="cell", units="1/cm", force_override=True)
            else:
                print("Not conserved smoothing...")
                self.yt_ds.add_smoothed_particle_field(("Gas", "Tsz"))

            def _proper_gas(pfilter, data):
                filter = data[pfilter.filtered_type, "StarFomationRate"] < 0.1
                return filter

            if (self.center is not None) and (self.radius is not None):
                self.yt_sp = self.yt_ds.sphere(center=self.center, radius=(self.radius, "kpc/h"))
            else:
                self.yt_sp = self.yt_ds.all_data()

            if ('Gas', 'StarFomationRate') in self.yt_ds.field_info.keys():
                if len(self.yt_sp['Gas', 'StarFomationRate'][self.yt_sp['Gas', 'StarFomationRate'] >= 0.1]) > 0:
                    yt.add_particle_filter("PGas", function=_proper_gas,
                                           filtered_type='Gas', requires=["StarFomationRate"])
                    self.yt_ds.add_particle_filter('PGas')
                    Ptype = 'PGas'

        return Ptype

    def prep_yt_SZ(self, conserved_smooth=False, force_redo=False):
        def _t_squared(field, data):
            return data["gas", "density"] * data["gas", "kT"] * data["gas", "kT"]
        self.yt_ds.add_field(("gas", "t_squared"), function=_t_squared,
                             units="g*keV**2/cm**3")

        def _beta_par_squared(field, data):
            return data["gas", "beta_par"]**2 / data["gas", "density"]
        self.yt_ds.add_field(("gas", "beta_par_squared"), function=_beta_par_squared,
                             units="g/cm**3")

        def _beta_perp_squared(field, data):
            return data["gas", "density"] * data["gas", "velocity_magnitude"]**2 \
                / speed_of_light_cgs / speed_of_light_cgs - data["gas", "beta_par_squared"]
        self.yt_ds.add_field(("gas", "beta_perp_squared"), function=_beta_perp_squared,
                             units="g/cm**3")

        def _t_beta_par(field, data):
            return data["gas", "kT"] * data["gas", "beta_par"]
        self.yt_ds.add_field(("gas", "t_beta_par"), function=_t_beta_par,
                             units="keV*g/cm**3")

        def _t_sz(field, data):
            return data["gas", "density"] * data["gas", "kT"]
        self.yt_ds.add_field(("gas", "t_sz"), function=_t_sz,
                             units="keV*g/cm**3")
