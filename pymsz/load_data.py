import numpy as np
from pymsz.readsnapsgl import readsnapsgl
# from astropy.cosmology import FlatLambdaCDM


class load_data(object):
    r"""load analysing data from simulation snapshots (gadget format only),
    yt, or raw data. Currently only works with snapshot=True. Will work more
    on other data sets.

    Parameters
    ----------
    filename    : The filename of simulation snapshot, or data. Default : ''
    snapshot    : Is loading snapshot or not? Default : True
    metal       : Gas metallicity. Default: None, will try to read from simulation.
                  Otherwise, will use this give metallicity, which can be float,
                  or array (Must be the same number of gas particles)

    yt_load     : Do you want to use yt to load the data? Default : False. Requries yt modelds.

    specified_field: If you want to specify the data fields for yt.load. Default: None.
                    Only works with yt_load = True.

    rawdata     : Is it raw data? Default : False
                  Please look at (or change) the reading function in this file to load the raw data.

    ---------- If you only need parts of loaded data. Specify center and radius
    center      : The center of a sphere for the data you want to get.
                  Default : None
    radius      : The radius of a sphere for the data you want to get.
                  Default : None

    Notes
    -----
    Need to take more care about the units!!! Currently only assume simulation units.
    kpc/h and 10^10 M_sun
    Raw data set needs to provide the cosmology, Otherwise WMAP7 is used...
    center and radius need to set together in the simulation units!
    Example
    -------
    simd = load_data(snapfilename="/home/weiguang/Downloads/snap_127",
                     snapshot=True,
                     center=[500000,500000,500000], radius=800)

    """

    def __init__(self, filename='', snapshot=True, metal=None, yt_load=False,
                 specified_field=None, datafile=False, center=None, radius=None):

        if snapshot:
            self.data_type = "snapshot"
            self.temp = np.array([])
            if metal is None:
                self.metal = 0
            elif isinstance(metal, type(0.1)) or isinstance(metal, type(np.ones(1))):
                self.metal = metal
            else:
                raise ValueError("Do not accept this metal %f." % metal)
            self.mass = np.array([])
            self.pos = np.array([])
            self.rho = np.array([])
            self.ne = 0
            self.hsml = 0
            self.cosmology = {}  # default wmap7
            # self.currenta = 1.0  # z = 0
            # self.z = 0.0
            # self.Uage = 0.0  # university age in Gyrs
            # self.nx = self.grid_mass = self.grid_age = self.grid_metal = None

            self._load_snap(filename, center, radius)
        elif yt_load:
            self.data_type = "yt_data"
            self.data = self._load_yt(filename, center, radius, specified_field)
        elif datafile:
            self.data_type = "snapshot"
            self._load_raw(datafile, center, radius)
        else:
            raise ValueError("Please sepecify the simulation data type. ")

    def _load_snap(self, filename, cc, rr):
        head = readsnapsgl(filename, "HEAD", quiet=True)
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
        spos = readsnapsgl(filename, "POS ", ptype=0, quiet=True)
        if (cc is not None) and (rr is not None):
            r = np.sqrt(np.sum((spos - cc)**2, axis=1))
            ids = r <= rr
            self.pos = spos[ids] - cc
        else:
            ids = np.ones(head[0][0], dtype=bool)
            self.pos = spos - np.mean(spos, axis=0)

        # Electron fraction
        self.ne = readsnapsgl(filename, "NE  ", quiet=True)
        if self.ne is not 0:
            self.ne = self.ne[ids]

        # Temperature
        self.temp = readsnapsgl(filename, "TEMP", quiet=True)
        if self.temp is not 0:
            self.temp = self.temp[ids]
        else:
            raise ValueError("Can't get gas temperature, which is required for this code.")

        # density
        self.rho = readsnapsgl(filename, "RHO ", quiet=True)
        if self.rho is not 0:
            self.rho = self.rho[ids]
        else:
            raise ValueError("Can't get gas density, which is required")

        # smoothing length
        self.hsml = readsnapsgl(filename, "HSML", quiet=True)
        if self.hsml is not 0:
            self.hsml = self.hsml[ids]

        # mass only gas
        self.mass = readsnapsgl(filename, "MASS", ptype=0, quiet=True)
        if self.mass is not 0:
            self.mass = self.mass[ids]
        else:
            raise ValueError("Can't get gas mass, which is required")

        # gas metal if there are
        if self.metal is 0:
            self.metal = readsnapsgl(filename, "Z   ", ptype=0, quiet=True)  # auto calculate Z
            if self.metal is not 0:
                self.metal = self.metal[ids]

        # we need to remove some spurious particles.... if there is a MHI or SRF block
        # see Klaus's doc or Borgani et al. 2003 for detials.
        mhi = readsnapsgl(filename, "MHI ", quiet=True)
        if mhi is 0:
            # try exclude sfr gas particles
            sfr = readsnapsgl(filename, "SFR ", quiet=True)
            if sfr is not 0:
                sfr = sfr[ids]
                ids_ex = sfr < 0.1
            else:
                ids_ex = None
        else:
            mhi = mhi[ids] / 0.76 / self.mass
            ids_ex = (self.temp < 3.0e4) & (self.rho > 6.e-7)
            ids_ex = (mhi < 0.1) & (~ids_ex)
            self.rho *= (1 - mhi)  # correct multi-phase baryon model by removing cold gas

        # Change NE (electron fraction to number density in simulation code/mp)
        Zs = readsnapsgl(filename, "Zs  ", quiet=True)
        if Zs is not 0:
            self.ne *= self.rho * (1 - self.metal - Zs[:, 0])
        else:
            if self.ne is not 0:
                self.ne *= self.rho * (1 - self.metal - 0.24)  # assume constant Y = 0.24
            else:
                # full ionized without taking metal into account
                self.ne = 1.315789474 * self.rho * (1 - self.metal - 0.24)

        if ids_ex is not None:
            self.temp = self.temp[ids_ex]     # cgs
            self.mass = self.mass[ids_ex]
            self.pos = self.pos[ids_ex]
            self.rho = self.rho[ids_ex]
            self.ne = self.ne[ids_ex]         # cgs
            if self.metal is not 0:
                self.metal = self.metal[ids_ex]
            if self.hsml is not 0:
                self.hsml = self.hsml[ids_ex]
            else:
                self.hsml = (3*self.mass/self.pos/4/np.pi)**(1./3.)  # approximate

    def _load_yt(self, filename, cc, rr, specified_field):
        try:
            import yt
        except ImportError:
            raise ImportError("Can not find yt package, which is required to use this function!")

        if specified_field is not None:
            from yt.frontends.gadget.definitions import gadget_field_specs
            gadget_field_specs["my_def"] = specified_field
            ds = yt.load(filename, field_spec="my_def")
        else:
            ds = yt.load(filename)

        if (cc is not None) and (rr is not None):
            sp = ds.sphere(center=cc, radius=(rr, "kpc/h"))
        else:
            sp = ds.all_data()

        if ('Gas', 'StarFomationRate') in ds.field_info.keys():
            sp = sp.cut_region(["obj[('Gas', 'StarFomationRate')] < 0.1"])

        return sp

        # if ('Gas', 'Temperature') in sp.ds.field_info.keys():
        #     self.temp = sp[('Gas', 'Temperature')].v
        # else:
        #     raise ValueError("Can't get gas temperature, which is required for this code.")
        # if ('Gas', 'Mass') in sp.ds.field_info.keys():
        #     self.mass = sp[('Gas', 'Mass')].v
        # else:
        #     raise ValueError("Can't get gas mass, which is required for this code.")
        # if ('Gas', 'Coordinates') in sp.ds.field_info.keys():
        #     self.pos = sp[('Gas', 'Coordinates')].v
        # else:
        #     raise ValueError("Can't get gas positions, which is required for this code.")
        # if ('Gas', 'Density') in sp.ds.field_info.keys():
        #     self.rho = sp[('Gas', 'Density')].v
        # else:
        #     raise ValueError("Can't get gas density, which is required for this code.")
        # if self.metal == 0:
        #     if ('Gas', 'Z') in sp.ds.field_info.keys():
        #         self.metal = sp[('Gas', 'Z')].v
        # if ('Gas', 'ElectronAbundance') in sp.ds.field_info.keys():
        #     self.ne = sp[('Gas', 'ElectronAbundance')].v
        # if ('Gas', 'SmoothingLength') in sp.ds.field_info.keys():
        #     self.hsml = sp[('Gas', 'SmoothingLength')].v

    def _load_raw(self, datafile, cc, rr):
        if (cc is not None) and (rr is not None):
            r = np.sqrt(np.sum((datafile['pos'] - cc)**2, axis=1))
            ids = r <= rr
        else:
            ids = np.ones(datafile['age'].size, dtype=bool)
        self.temp = datafile['age'][ids]
        self.mass = datafile['mass'][ids]
        self.metal = datafile['metal'][ids]
