import numpy as np
from pymsz.readsnapsgl import readsnapsgl
# from astropy.cosmology import FlatLambdaCDM
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
                  Please look at (or change) the reading function in this file to load the raw data.

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
            self.rho = np.array([])
            self.ne = 0
            self.hsml = 0
            self.cosmology = {}  # default wmap7
            self.Tszdata = np.array([])

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
            r = np.sqrt(np.sum((spos - self.center)**2, axis=1))
            ids = r <= self.radius
            self.pos = spos[ids] - self.center
        else:
            ids = np.ones(head[0][0], dtype=bool)
            self.pos = spos - np.mean(spos, axis=0)

        # Electron fraction
        self.ne = readsnapsgl(self.filename, "NE  ", quiet=True)
        if self.ne is not 0:
            self.ne = self.ne[ids]

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

        # Change NE (electron fraction to number density in simulation code/mp)
        Zs = readsnapsgl(self.filename, "Zs  ", quiet=True)
        if Zs is not 0:
            self.ne *= self.rho * (1 - self.metal - Zs[ids, 0])
        else:
            if self.ne is not 0:
                self.ne *= self.rho * (1 - self.metal - 0.24)  # assume constant Y = 0.24
            else:
                if self.mu is None:
                    # full ionized without taking metal into account
                    self.ne = 1.157894736842105 * self.rho * (1 - self.metal - 0.24)
                else:
                    ae = (4.0 / self.mu - 3.0 * 0.76 - 1.0) / 4.0 / 0.76
                    self.ne = ae * self.rho * (1 - self.metal - 0.24)

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
            self.rho = self.rho[ids_ex]
            self.ne = self.ne[ids_ex]         # cgs
            if self.metal is not 0:
                self.metal = self.metal[ids_ex]
            if self.hsml is not 0:
                self.hsml = self.hsml[ids_ex]
            else:
                self.hsml = (3 * self.mass / self.pos / 4 / np.pi)**(1. / 3.)  # approximate

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

    def prep_snap(self):  # Now everything need to be in physical
        if len(self.Tszdata) == 0:  # only need to prepare once
            self.Tszdata = self.ne / Mp * \
                (1.0e10 * M_sun * self.cosmology["h"]**2 / Kpc**3)  # now in cm^-3
            self.Tszdata *= Kb * self.temp * cs / me / c**2  # now in cm^-1

    def prep_yt(self):
        if 'PGas' in self.yt_ds.particle_types:
            Ptype = 'PGas'
        else:
            Ptype = 'Gas'

        if self.yt_sp is None:  # only need to calculate once
            import yt

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

            # def Ele_num_den(field, data):
            #     # if ("Gas", "ElectronAbundance") in data.ds.field_info:
            #     return data[field.name[0], "Density"] * data[field.name[0], "ElectronAbundance"] * \
            #         (1 - data[field.name[0], "Z"] - 0.24) / mp
            #     # else:  # Assume full ionized
            #     # return data[field.name[0], "Density"] * 1.351 * (1 - data[field.name[0],
            #     # "Z"] - 0.24) / mp

            def Temp_SZ(field, data):
                const = kb * cross_section_thompson_cgs / mass_electron_cgs / speed_of_light_cgs**2 / mp
                end = data[field.name[0], "Density"] * data[field.name[0], "ElectronAbundance"] * \
                    (1 - data[field.name[0], "Z"] - 0.24)
                return end * data[field.name[0], 'Temperature'] * const

            def MTsz(field, data):
                return data[field.name[0], 'Tsz'] * data[field.name[0], 'Mass']

            def SMWTsz(field, data):
                return data[field.name[0], Ptype + '_smoothed_MTsz'] / data[field.name[0], Ptype + '_smoothed_Mass']

            # self.yt_ds.add_field((Ptype, "END"), function=Ele_num_den,
            #                      sampling_type="particle", units="cm**(-3)")
            self.yt_ds.add_field((Ptype, "Tsz"), function=Temp_SZ,
                                 sampling_type="particle", units="1/cm")
            self.yt_ds.add_field((Ptype, "MTsz"), function=MTsz,
                                 sampling_type="particle", units="g/cm")
            # self.yt_ds.add_smoothed_particle_field((Ptype, "Tsz"))
            self.yt_ds.add_smoothed_particle_field((Ptype, "Mass"))
            self.yt_ds.add_smoothed_particle_field((Ptype, "MTsz"))
            self.yt_ds.add_field(('deposit', "MWSTsz"), function=SMWTsz,
                                 sampling_type="cell", units="1/cm")

        return Ptype
