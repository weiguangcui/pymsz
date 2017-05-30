import numpy as np
from pymsz.readsnapsgl import readsnapsgl
# from astropy.cosmology import FlatLambdaCDM
Mu = 0
Metal = 0


def add_GEA(field, data):  # full ionized gas ElectronAbundance
    global Mu
    # mu = 4.0 / (3.0 * x_H + 1.0 + 4.0 * x_H * a_e)
    ae = (4.0 / Mu - 3.0 * 0.76 - 1.0) / 4.0 / 0.76
    return data['Gas', 'particle_ones'] * ae


def add_GMT(field, data):  # No metallicity
    global Metal
    return data['Gas', 'particle_ones'] * Metal


def proper_gas(pfilter, data):
    filter = data[pfilter.filtered_type, "StarFomationRate"] < 0.1
    return filter


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
                 specified_field=None, datafile=False, center=None, radius=None):
        global Metal, Mu
        self.center = center
        self.radius = radius
        self.filename = filename
        if metal is None:
            self.metal = 0
        elif isinstance(metal, type(0.1)) or isinstance(metal, type(np.ones(1))):
            self.metal = metal
        else:
            raise ValueError("Do not accept this metal %f." % metal)
        Metal = self.metal
        self.mu = mu
        if self.mu is None:
            Mu = 0.5882352941176471  # full ionized
        else:
            Mu = self.mu

        if snapshot:
            self.data_type = "snapshot"
            self.temp = np.array([])
            self.mass = 0.0
            self.pos = np.array([])
            self.rho = np.array([])
            self.ne = 0
            self.hsml = 0
            self.cosmology = {}  # default wmap7
            # self.currenta = 1.0  # z = 0
            # self.z = 0.0
            # self.Uage = 0.0  # university age in Gyrs
            # self.nx = self.grid_mass = self.grid_age = self.grid_metal = None

            self._load_snap()
        elif yt_load:
            self.data_type = "yt_data"
            self.data = self._load_yt(specified_field)
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
                # full ionized without taking metal into account
                self.ne = 1.157894736842105 * self.rho * (1 - self.metal - 0.24)

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
        except ImportError:
            raise ImportError("Can not find yt package, which is required to use this function!")

        if specified_field is not None:
            from yt.frontends.gadget.definitions import gadget_field_specs
            gadget_field_specs["my_def"] = specified_field
            ds = yt.load(self.filename, field_spec="my_def")
        else:
            ds = yt.load(self.filename)

        if ("Gas", "ElectronAbundance") not in ds.field_list:
            if self.mu is None:
                print("Add electrons as full ionized gas")
            else:
                print("Add electrons for gas with given mean_mol_weight %f" % self.mu)
            ds.add_field(("Gas", "ElectronAbundance"), function=add_GEA,
                         sampling_type="particle", units="", force_override=True)

        if ("Gas", "Z") not in ds.field_list:
            print("Adding given metallicity %f" % Metal)
            ds.add_field(("Gas", "Z"), function=add_GMT,
                         sampling_type="particle", units="", force_override=True)

        if (self.center is not None) and (self.radius is not None):
            sp = ds.sphere(center=self.center, radius=(self.radius, "kpc/h"))
        else:
            sp = ds.all_data()

        if ('Gas', 'StarFomationRate') in ds.field_info.keys():
            if len(sp['Gas', 'StarFomationRate'][sp['Gas', 'StarFomationRate'] >= 0.1]) > 0:
                yt.add_particle_filter("PGas", function=proper_gas,
                                       filtered_type='Gas', requires=["StarFomationRate"])
                ds.add_particle_filter('PGas')

        # if ('Gas', 'StarFomationRate') in ds.field_info.keys():  this is only work with cell data
        #     sp = sp.cut_region(["obj[('Gas', 'StarFomationRate')] < 0.1"])

        return sp

    # def _load_raw(self):
    #     if (self.center is not None) and (self.radius is not None):
    #         r = np.sqrt(np.sum((self.filename['pos'] - self.center)**2, axis=1))
    #         ids = r <= self.radius
    #     else:
    #         ids = np.ones(self.filename['age'].size, dtype=bool)
    #     self.temp = self.filename['age'][ids]
    #     self.mass = self.filename['mass'][ids]
    #     self.metal = self.filename['metal'][ids]
