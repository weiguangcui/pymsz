import numpy as np
from pymgal.readsnapsgl import readsnapsgl
from astropy.cosmology import FlatLambdaCDM


class load_data(object):
    r"""load analysing data from simulation snapshots (gadget format only),
    yt, or raw data. Currently only works with snapshot=True. Will work more
    on other data sets.

    Parameters
    ----------
    filename    : The filename of simulation snapshot, or data. Default : ''
    snapshot    : Is loading snapshot or not? Default : True

    yt_data     : Is it yt data? Default : False

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
    Raw data set needs to provide the cosmology ...
    center and radius need to set both!
    Example
    -------
    simd = load_data(snapfilename="/home/weiguang/Downloads/snap_127",
                     snapshot=True,
                     center=[500000,500000,500000], radius=800)

    """

    def __init__(self, filename='', snapshot=True, yt_data=False,
                 datafile=False, center=None, radius=None):

        self.temp = np.array([])
        self.metal = 0
        self.mass = np.array([])
        self.pos = np.array([])
        self.rho = np.array([])
        self.NE = 0
        self.hsml = 0
        self.cosmology = None  # default wmap7
        self.currenta = 1.0  # z = 0
        self.z = 0.0
        self.Uage = 0.0  # university age in Gyrs
        self.nx = self.grid_mass = self.grid_age = self.grid_metal = None

        if snapshot:
            self.datatype = "snapshot"
            self._load_snap(filename, center, radius)
        elif yt_data:
            self.datatype = "yt"
            self._load_yt(filename, center, radius)
        elif datafile:
            self.datatype = "rawdata"
            self._load_raw(datafile, center, radius)

    def _load_snap(self, filename, cc, rr):
        head = readsnapsgl(filename, "HEAD", quiet=True)
        self.cosmology = FlatLambdaCDM(head[-1] * 100, head[-3])
        self.currenta = head[2]
        self.Uage = self.cosmology.age(1. / self.currenta - 1)
        self.z = head[3] if head[3] > 0 else 0.0

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
        self.NE = readsnapsgl(filename, "NE  ", quiet=True)
        if self.NE != 0:
            self.NE = self.NE[ids]

        # Temperature
        self.temp = readsnapsgl(filename, "TEMP", quiet=True)
        if self.temp != 0:
            self.temp = self.temp[ids]
        else:
            raise ValueError("Can't get gas temperature, which is required for this code.")

        # density
        self.rho = readsnapsgl(filename, "RHO ", quiet=True)
        if self.rho != 0:
            self.rho = self.rho[ids]
        else:
            raise ValueError("Can't get gas density, which is required")

        # smoothing length
        self.hsml = readsnapsgl(filename, "HSML", quiet=True)
        if self.hsml != 0:
            self.hsml = self.hsml[ids]

        # mass only gas
        self.mass = readsnapsgl(filename, "MASS", ptype=0, quiet=True)
        if self.mass != 0:
            self.mass = self.mass[ids]
        else:
            raise ValueError("Can't get gas mass, which is required")

        # gas metal if there are
        self.metal = readsnapsgl(filename, "Z   ", ptype=0, quiet=True)
        if self.metal != 0:
            self.metal = self.metal[ids]

        # we need to remove some spurious particles.... if there is a MHI or SRF block
        mhi = readsnapsgl(filename, "MHI ", quiet=True)
        if mhi == 0:
            # try exclude sfr gas particles
            sfr = readsnapsgl(filename, "SFR ", quiet=True)
            if sfr != 0:
                sfr = sfr[ids]
                ids_ex = sfr < 0.1
            else:
                ids_ex = None
        else:
            mhi = mhi[ids] / 0.76 / self.mass
            ids_ex = (self.temp < 3.0e4) & (self.rho > 6.e-7)
            ids_ex = (mhi < 0.1) & (~ids_ex)
        if ids_ex is not None:
            self.temp = self.temp[ids_ex]
            self.mass = self.mass[ids_ex]
            self.pos = self.pos[ids_ex]
            self.rho = self.rho[ids_ex]
            if self.metal != 0:
                self.metal = self.metal[ids_ex]
            if self.NE != 0:
                self.NE = self.NE[ids_ex]
            if self.hsml != 0:
                self.hsml = self.hsml[ids_ex]

    def _load_yt(self, filename, cc, rr):
        import yt
        ds = yt.load(filename)
        sp = yt.sperical(ds)
        return(sp)

    def _load_raw(self, datafile, cc, rr):
        if (cc is not None) and (rr is not None):
            r = np.sqrt(np.sum((datafile['pos'] - cc)**2, axis=1))
            ids = r <= rr
        else:
            ids = np.ones(datafile['age'].size, dtype=bool)
        self.temp = datafile['age'][ids]
        self.mass = datafile['mass'][ids]
        self.metal = datafile['metal'][ids]

    def rotate_grid(self, axis, nx):
        r""" rotate the data points and project them into a 2D grid.

        Parameter:
        ----------
        axis    : can be 'x', 'y', 'z', or a list of degrees [alpha, beta, gamma],
                  which will rotate the data points by $\alpha$ around the x-axis,
                  $\beta$ around the y-axis, and $\gamma$ around the z-axis
        nx      : The pixel size of the grid. A nx x nx image can be produced later.
        """
        self.nx = nx
        # ratation data points first
        if isinstance(axis, type('')):
            if axis == 'y':  # x-z plane
                self.pos[:, 1] = self.pos[:, 2]
            elif axis == 'x':  # y - z plane
                self.pos[:, 0] = self.pos[:, 1]
                self.pos[:, 1] = self.pos[:, 2]
            else:
                if axis != 'z':  # project to xy plane
                    raise ValueError(
                        "Do not accept this value %s for projection" % axis)
        elif isinstance(axis, type([])):
            if len(axis) == 3:
                sa, ca = np.sin(axis[0] / 180. *
                                np.pi), np.cos(axis[0] / 180. * np.pi)
                sb, cb = np.sin(axis[1] / 180. *
                                np.pi), np.cos(axis[1] / 180. * np.pi)
                sg, cg = np.sin(axis[2] / 180. *
                                np.pi), np.cos(axis[2] / 180. * np.pi)
                # ratation matrix from
                # http://inside.mines.edu/fs_home/gmurray/ArbitraryAxisRotation/
                Rxyz = np.array(
                    [[cb * cg, cg * sa * sb - ca * sg, ca * cg * sb + sa * sg],
                     [cb * sg, ca * cg + sa * sb * sg, ca * sb * sg - cg * sa],
                     [-sb,     cb * sa,                ca * cb]], dtype=np.float64)
                self.pos = np.dot(self.pos, Rxyz)
            else:
                raise ValueError(
                    "Do not accept this value %s for projection" % axis)

        # Now grid the data
        # pmax, pmin = np.max(self.pos, axis=0), np.min(self.pos, axis=0)
        # grid_x, grid_y = np.mgrid[pmin[0]:pmax[0]:nx, pmin[1]:pmax[1]:nx]
        self.grid_mass = np.histogram2d(self.pos[:, 0], self.pos[:, 1], bins=[
                                        nx, nx], weights=self.mass)[0]
        ids = self.grid_mass > 0
        self.grid_age = np.histogram2d(self.pos[:, 0], self.pos[:, 1], bins=[
                                       nx, nx], weights=self.temp * self.mass)[0]
        self.grid_age[ids] /= self.grid_mass[ids]  # mass weighted age
        self.grid_metal = np.histogram2d(self.pos[:, 0], self.pos[:, 1], bins=[
                                         nx, nx], weights=self.metal * self.mass)[0]
        self.grid_metal[ids] /= self.grid_mass[ids]  # mass weighted metal
        # dx = (pmax[0] - pmin[0] + pmax[0] * 0.001) / nx
        # dy = (pmax[1] - pmin[1] + pmax[1] * 0.001) / nx
        # self.grids = np.int32(
        #     np.floor((self.pos[:, :2] - pmin[:2]) / np.array([dx, dy])))
