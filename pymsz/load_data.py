import numpy as np
from pymsz.readsnapsgl import readsnap
# from astropy.cosmology import FlatLambdaCDM
try:
    from yt.utilities.physical_constants import mp, kb, cross_section_thompson_cgs, \
        solar_mass, mass_electron_cgs, speed_of_light_cgs, Tcmb, hcgs
    from yt.utilities.physical_ratios import cm_per_kpc as Kpc
except ImportError:
    Mp = 1.672621777e-24  # proton mass in g
    Kb = 1.38064852e-16      # Boltzman constants in erg/K
    cs = 6.65245854533e-25  # cross_section_thompson in cm**2
    M_sun = 1.9891e+33  # g
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
    filename    : The filename of simulation snapshot, or data. For multiple file, should just put the file base name here!
                    Type: str. Default : ''
    metal       : Gas metallicity.
                    Type: float or array. Default: None, will try to read from simulation.
                    Otherwise, will use this give metallicity.
                    It must be the same number of gas particles if it is an array.
    Nmets       : Number of metal elements in the simulation metallicity data block. Defualt : 11
                    set to 0 for no metals.
    mu          : mean_molecular_weight.
                    Type: float. Default: None.
                    It is used for calculating gas temperature, when there is no NE block
                    from the simulation snapshot.
                    with snapshot=True, it assumes full ionized gas with mu ~ 0.588.
                    with yt_load=True, it assumes zero ionization with mu ~ 1.22.
    snapshot    : Is loading snapshot or not? 
                    Type: bool. Default : False. If loading HDF5 files, specify it with snapshot='hdf5'
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
                  It is recommended that if you want higher resolution, try reducing the
                    value of n_ref to 32 or 16.

    rawdata     : Is it raw data? Default : False
                  Please look at/change the reading function in this file to load the raw data.

    ---------- If you only need parts of loaded data. Specify center and radius
    center      : The center of a sphere for the data you want to get.
                  Default : None, whole data will load.
    radius      : The radius of a sphere for the data you want to get.
                  Default : None, whole data will be used.
    restrict_r  : Using the exact radius to cut out simulation data.
                  Default : True. Otherwise, a cubic (2 *radius) data is cut out to fill the output fits image.
    hmrad       : Radius for calculation halo motion, which is used for calculating the kSZ effect.
                  Default : None, the halo motion are given by the mean of all particles.
                  0 or nagative value for not removing halo motion.
    ---------- additional data cut to exclude spurious gas particles.
    cut_sfr     : All the data higher than this star formation rate are excluded in the calc. Default None.
                  NUnit [Msun/yr].
    cut_rhoT    : You can also do density and temperature cut simultaneously. Default None, setting to [6.e-7, 3.0e4] 
                  will exlcude particles rho < 6.e-7 and T > 3.0e4.

    Notes
    -----
    Please be extremly careful about the units!!! Currently only assume in simulation units:
        kpc/h and 10^10 M_sun
    Raw data set needs to provide the cosmology, Otherwise WMAP7 is used for later calculation...
    center and radius need to set together in the simulation units!
    No periodical boundery effect is considered yet!

    Example
    -------
    simd = load_data(snapfilename="/home/weiguang/Downloads/snap_127",
                     snapshot=True, center=[500000,500000,500000], radius=800)
    """

    def __init__(self, filename='', metal=None, Nmets=11, mu=None, snapshot=False, yt_load=False,
                 specified_field=None, n_ref=None, datafile=False, center=None, radius=None,
                 restrict_r=True, hmrad=None, cut_sfr=None, cut_rhoT=[None, None]):
        self.center = center
        self.radius = radius
        self.filename = filename

        self.metal = metal
        if (metal is not None) and (not isinstance(metal, type(0.1)) or not isinstance(metal, type(np.ones(1)))):
            raise ValueError("Do not accept this metal %f." % metal)
            
        self.Nmets = Nmets
        self.mu = mu
        self.n_ref = n_ref
        self.hmrad = hmrad
        self.cut_sfr = cut_sfr
        self.cut_rhoT = cut_rhoT
        self.restrict_r = restrict_r

        if (snapshot) or (snapshot.lower()=='hdf5'):
            self.data_type = "snapshot"
            self.temp = np.array([])
            self.mass = None
            self.pos = np.array([])
            self.vel = np.array([])
            self.rho = np.array([])
            self.ne = None
            self.X = None
            self.hsml = None
            self.cosmology = {}  # default wmap7
            self.bulkvel = np.array([])

            self.Tszdata = np.array([])  # prep_ss_TT
            self.Kszdata = np.array([])  # prep_ss_KT

            self.tau = np.array([])  # prep_ss_SZT
            self.Te = np.array([])
            self.bpar = np.array([])
            self.omega = np.array([])
            self.sigma = np.array([])
            self.kappa = np.array([])
            self.bperp = np.array([])
            self.mmw = None  # mean_molecular_weight for internal calculation
            # self.currenta = 1.0  # z = 0
            # self.z = 0.0
            # self.Uage = 0.0  # university age in Gyrs
            # self.nx = self.grid_mass = self.grid_age = self.grid_metal = None
            if self.filename[-4:].upper() == 'HDF5' or self.filename[-3:].upper() == 'HDF' or self.filename[-3:].lower()=='hdf5':
                # import h5py
                # sn = h5py.File(self.filename, 'r')
                self._load_snap_hdf()  #current desgin for GIZMO, may need tricks for other simulations
                # sn.close()
            else:
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

    def _load_snap_hdf(self):
        head = readsnap(self.filename, "Header", quiet=False)
        self.cosmology["z"] = head.Redshift  # if head[3] > 0 else 0.0
        self.cosmology["a"] = head.Time
        self.cosmology["omega_matter"] = head.Omega0
        self.cosmology["omega_lambda"] = head.OmegaLambda
        self.cosmology["h"] = head.HubbleParam
        self.cosmology["boxsize"] = head.Boxsize
        
        # check wind particles and remove them if possible
        iddt = readsnap(self.filename, "DelayTime", quiet=True, ptype=0)
        if iddt is not None:
            iddt = iddt > 0
        # if 'DelayTime' in sn['/PartType0'].keys():
        #     iddt = sn['/PartType0/DelayTime'][:]>0
        # else:
        #     iddt = None
            
        # gas pos
        # spos = sn['/PartType0/Coordinates'][:]
        spos = readsnap(self.filename, "Coordinates", quiet=True, ptype=0)
        if (self.center is not None) and (self.radius is not None):
            # periodic bound condition
            for i in range(3):
                if self.center[i]+self.radius > self.cosmology["boxsize"]:
                    ids=spos[:,i] <= self.center[i]+self.radius - self.cosmology["boxsize"]
                    spos[ids,i] += self.cosmology["boxsize"]
                if self.center[i] - self.radius < 0:
                    ids = spos[:,i] >= self.center[i] - self.radius + self.cosmology["boxsize"]
                    spos[ids,i] -= self.cosmology["boxsize"]
            if self.restrict_r:
                r = np.sqrt(np.sum((spos - self.center)**2, axis=1))
                ids = r <= self.radius  
            else:
                ids = (spos[:, 0] > self.center[0] - self.radius) & \
                    (spos[:, 0] <= self.center[0] + self.radius) & \
                    (spos[:, 1] > self.center[1] - self.radius) & \
                    (spos[:, 1] <= self.center[1] + self.radius) & \
                    (spos[:, 2] > self.center[2] - self.radius) & \
                    (spos[:, 2] <= self.center[2] + self.radius)
            if iddt is not None:
                ids = ids&(~iddt)  # exclude wind particles
            self.pos = spos[ids] - self.center
        else:
            # ids = np.ones(sn['Header'].attrs['NumPart_Total'][0], dtype=bool)
            self.center = np.median(spos, axis=0)
            if iddt is not None:
                ids = ~iddt
            else:
                ids = np.ones(self.pos.shape[0], dtype=bool)
            self.pos = spos[ids] - self.center
            self.radius = (self.pos.max()-self.pos.min())/2.
        
        # gas velocity
        self.vel = readsnap(self.filename, "Velocities", quiet=True, ptype=0)
        if self.vel is not None:
            # self.vel = sn['/PartType0/Velocities'][:] 
            self.vel = self.vel[ids] * np.sqrt(self.cosmology["a"]) # to peculiar velocity
        else:
            raise ValueError("Can't get gas velocity, which is required")
        self.bulkvel = np.mean(self.vel, axis=0)  # bulk velocity is given by mean
        if self.hmrad is None:
            self.vel -= self.bulkvel  # remove halo motion
        else:
            if self.hmrad > 0:  # Not remove the halo bulk velocity if hmrad < = 0
                r = np.sqrt(np.sum(self.pos**2, axis=1))
                self.bulkvel = np.mean(self.vel[r < self.hmrad], axis=0)
                self.vel -= self.bulkvel

        # gas metal if there are
        if self.metal is not None:
            self.X = 1 - self.metal - 0.24  # simply assume He=0.24
        else:
            self.metal = readsnap(self.filename, "Metallicity", quiet=True, ptype=0) # GIZMO
            if self.metal is None:
                self.metal = readsnap(self.filename, "GFM_Metallicity", quiet=True, ptype=0) # Illustris
                if self.metal is None:
                    raise ValueError('Metallicity can not be read from simulation!')
                else:
                    # self.metal=self.metal[ids]
                    self.X = readsnap(self.filename, "GFM_Metals", quiet=True, ptype=0) 
                    self.X = self.X[:, 0]
            else:
                self.X = 1 - self.metal[:,0] - self.metal[:,1] # Now self.X is hydrogen mass fraction, electron number = M*X/m_H*NE
                # self.metal = self.metal[ids,0]
        
        yhelium = (1. - self.X) / (4 * self.X)
        if isinstance(self.X, type(np.array([0.0]))):
            self.X = self.X[ids]
        if isinstance(self.metal, type(np.array([0.0]))):
            self.metal = self.metal[ids]
        
        # Electron fraction
        self.ne = readsnap(self.filename, "ElectronAbundance", quiet=True, ptype=0)   
        if self.ne is not None:
            self.mmw = (1. + 4. * yhelium) / (1. + yhelium + self.ne)
            self.ne = self.ne[ids]
        else:  # calculate NE from mean mol weight
            if self.mu is not None:
                self.ne = np.ones(self.rho.size) * (4.0 / self.mu - 3.28) / 3.04
            else:
                self.mmw = (1. + 4. * yhelium) / (1. + 3 * yhelium + 1)  # assume full ionized
                if isinstance(self.mmw, type(np.array([0.0]))):
                    self.ne = np.ones(self.rho.size) * (4.0 / self.mmw[ids] - 3.28) / 3.04
                else:
                    self.ne = np.ones(self.rho.size) * (4.0 / self.mmw - 3.28) / 3.04                

        # Temperature
        if self.mu is not None:
            self.temp = readsnap(self.filename, "Temperature", my=self.mu, quiet=True)
        elif self.mmw is not None:
            self.temp = readsnap(self.filename, "Temperature", mu=self.mmw, quiet=True)
            if isinstance(self.mmw, type(np.array([0.0]))):
                self.mmw = self.mmw[ids]
        if self.temp is not None:
            self.temp = self.temp[ids]
        else:
            U = readsnap(self.filename, "InternalEnergy", quiet=True, ptype=0)
            if U is None:
                raise ValueError('InternalEnergy for calculating temperature can not be read from simulation!')
            else:
                U = U[ids]
                # U = sn['/PartType0/InternalEnergy'][ids]
                v_unit = 1.0e5      # (e.g. 1.0 km/sec)
                prtn = 1.67373522381e-24  # (proton mass in g)
                bk = 1.3806488e-16        # (Boltzman constant in CGS)
                if self.mu is not None:
                    self.temp = U * (5. / 3 - 1) * v_unit**2 * prtn * self.mu / bk
                elif self.mmw is not None:
                    self.temp = U * (5. / 3 - 1) * v_unit**2 * prtn * self.mmw / bk
                else:
                    raise ValueError('mean_molecular_weight mu need to be set!')
                U = 0

        # density
        self.rho = readsnap(self.filename, "Density", quiet=True, ptype=0)
        if self.rho is not None:
            self.rho = self.rho[ids]
        else:
            raise ValueError("Can't get gas density, which is required")

        # smoothing length
        self.hsml = readsnap(self.filename, "SmoothingLength", quiet=True, ptype=0)
        if self.hsml is not None:
            self.hsml = self.hsml[ids]

        # mass only gas
        self.mass = readsnap(self.filename, "Masses", quiet=True, ptype=0)
        if self.mass is not None:
            self.mass = self.mass[ids]
        else:
            raise ValueError("Can't get gas Mass, which is required")

        # we need to remove some spurious particles....
        # try exclude sfr gas particles
        sfr = readsnap(self.filename, "StarFormationRate", quiet=True, ptype=0)
        if (self.cut_sfr is not None) and (sfr is not None):
            sfr = sfr[ids]
            ids_ex = sfr < self.cut_sfr
        else:
            ids_ex = np.ones(self.pos.shape[0], dtype=bool)
        if (self.cut_rhoT[0] is not None) and (self.cut_rhoT[1] is not None):
            ids_ex = ((self.temp > self.cut_rhoT[1]) | (self.rho < self.rhoT[0])) & ids_ex

        self.temp = self.temp[ids_ex]     # cgs
        if not isinstance(self.mass, type(0.0)):
            self.mass = self.mass[ids_ex]
        if not isinstance(self.X, type(0.0)):
            self.X = self.X[ids_ex]
        self.pos = self.pos[ids_ex]
        self.vel = self.vel[ids_ex]
        self.rho = self.rho[ids_ex]
        self.ne = self.ne[ids_ex]
        if not isinstance(self.metal, type(0.0)): # or isinstance(self.metal, type(np.ones(1))):
            self.metal = self.metal[ids_ex]
        if self.hsml is not None:
            self.hsml = self.hsml[ids_ex]
        else:
            self.hsml = (3 * self.mass / self.rho / 4 / np.pi)**(1. / 3.)  # approximate

    def _load_snap(self):
        head = readsnap(self.filename, "HEAD", quiet=True)
        self.cosmology["z"] = head.Redshift  # if head[3] > 0 else 0.0
        self.cosmology["a"] = head.Time
        self.cosmology["omega_matter"] = head.Omega0
        self.cosmology["omega_lambda"] = head.OmegaLambda
        self.cosmology["h"] = head.HubbleParam
        self.cosmology["boxsize"] = head.Boxsize
        # self.cosmology = FlatLambdaCDM(head[-1] * 100, head[-3])
        # self.currenta = head[2]
        # self.Uage = self.cosmology.age(1. / self.currenta - 1)
        # self.z = head[3] if head[3] > 0 else 0.0

        # positions # only gas particles
        spos = readsnap(self.filename, "POS ", ptype=0, quiet=True)
        if (self.center is not None) and (self.radius is not None):
            for i in range(3):
                if self.center[i]+self.radius > self.cosmology["boxsize"]:
                    ids=spos[:,i] <= self.center[i]+self.radius - self.cosmology["boxsize"]
                    spos[ids,i] += self.cosmology["boxsize"]
                if self.center[i] - self.radius < 0:
                    ids = spos[:,i] >= self.center[i] - self.radius + self.cosmology["boxsize"]
                    spos[ids,i] -= self.cosmology["boxsize"]
            if self.restrict_r:
                r = np.sqrt(np.sum((spos - self.center)**2, axis=1))
                ids = r <= self.radius  
            else:
                # ids = r <= self.radius*np.sqrt(2)  # increase to get all the projected data
                # Now using cubic box to get the data
                ids = (spos[:, 0] > self.center[0] - self.radius) & \
                    (spos[:, 0] <= self.center[0] + self.radius) & \
                    (spos[:, 1] > self.center[1] - self.radius) & \
                    (spos[:, 1] <= self.center[1] + self.radius) & \
                    (spos[:, 2] > self.center[2] - self.radius) & \
                    (spos[:, 2] <= self.center[2] + self.radius)
            self.pos = spos[ids] - self.center
        else:
            # ids = np.ones(head[0][0], dtype=bool)
            self.center = np.median(spos, axis=0)
            self.pos = spos - self.center
            ids = np.ones(self.pos.shape[0], dtype=bool)
            self.radius = (self.pos.max()-self.pos.min())/2.

        # velocity
        self.vel = readsnap(self.filename, "VEL ", ptype=0, quiet=True)
        if self.vel is not None:
            self.vel = self.vel[ids] * np.sqrt(self.cosmology["a"])  # to peculiar velocity
        else:
            raise ValueError("Can't get gas velocity, which is required")
        # Althoug the halo motion should be the mean of all particles
        # It is very close to only use gas particles. Besides little effect to the final result.
        self.bulkvel = np.mean(self.vel, axis=0)  # bulk velocity is given by mean
        if self.hmrad is None:
            self.vel -= self.bulkvel  # remove halo motion
        else:
            if self.hmrad > 0:  # Not remove the halo bulk velocity if hmrad < = 0
                r = np.sqrt(np.sum(self.pos**2, axis=1))
                self.bulkvel = np.mean(self.vel[r < self.hmrad], axis=0)
                self.vel -= self.bulkvel

        # gas metal if there are
        if self.metal is None:
            self.metal = readsnap(self.filename, "Z   ", ptype=0, nmet=self.Nmets, quiet=True)  # auto calculate Z
            if self.metal is None:
                raise ValueError("No metallicity! please give me a value as I can't find in snapshot!")

        Zs = readsnap(self.filename, "Zs  ", ptype=0, quiet=True)
        if Zs is not None:
            self.X = 1 - self.metal - Zs[:, 0]/self.mass  # hydrogen mass fraction assume Tornatore et al. 2007.
        else:
            self.X = 1 - self.metal - 0.24  # simply assume He=0.24
            # Now self.X is hydrogen mass fraction, electron number = M*X/m_H*NE
        # else:  # for no matalicity!
        #     # Change NE (electron number fraction respected to H number density in simulation)
        #     # M/mmw/mp gives the total nH+nHe+ne! To get the ne, which = self.ne*nH, we reexpress this as ne*Q_NE.
        #     # Q_NE is given in self.ne in below.
        #     self.ne = (1. + yhelium + self.ne) / self.ne
        # yhelium = 0.07894736842105263
        yhelium = (1. - self.X) / (4 * self.X)
        if isinstance(self.X, type(np.array([0.0]))):
            self.X = self.X[ids]
        if isinstance(self.metal, type(np.array([0.0]))):
            self.metal = self.metal[ids]  
        
        # Electron fraction
        self.ne = readsnap(self.filename, "NE  ", quiet=True)
        # ( 1. - xH ) / ( 4 * xH )hydrogen mass-fraction (xH) = n_He/n_H      
        if self.ne is not None:
            self.mmw = (1. + 4. * yhelium) / (1. + yhelium + self.ne)
            self.ne = self.ne[ids]
        else:  # calculate NE from mean mol weight
            if self.mu is not None:
                self.ne = np.ones(self.rho.size) * (4.0 / self.mu - 3.28) / 3.04
            else:
                self.mmw = (1. + 4. * yhelium) / (1. + 3 * yhelium + 1)  # assume full ionized
                # full ionized without taking metal into account. What about metal?
                if isinstance(self.mmw, type(np.array([0.0]))):
                    self.ne = np.ones(self.rho.size) * (4.0 / self.mmw[ids] - 3.28) / 3.04
                else:
                    self.ne = np.ones(self.rho.size) * (4.0 / self.mmw - 3.28) / 3.04
                # (4.0 / self.mmw - 3.0 * 0.76 - 1.0) / 4.0 / 0.76

        # Temperature
        if self.mu is not None:
            self.temp = readsnap(self.filename, "TEMP", mu=self.mu, quiet=True)
        elif self.mmw is not None:
            self.temp = readsnap(self.filename, "TEMP", mu=self.mmw, quiet=True)
            if isinstance(self.mmw, type(np.array([0.0]))):
                self.mmw = self.mmw[ids]
        if self.temp is not None:
            self.temp = self.temp[ids]
        else:
            raise ValueError("Can't get gas temperature, which is required for this code.")

        # density
        self.rho = readsnap(self.filename, "RHO ", quiet=True)
        if self.rho is not None:
            self.rho = self.rho[ids]
        else:
            raise ValueError("Can't get gas density, which is required")

        # smoothing length
        self.hsml = readsnap(self.filename, "HSML", quiet=True)
        if self.hsml is not None:
            self.hsml = self.hsml[ids]

        # mass only gas
        self.mass = readsnap(self.filename, "MASS", ptype=0, quiet=True)
        if not isinstance(self.mass, type(0.0)):
            self.mass = self.mass[ids]

        # we need to remove some spurious particles.... if there is a MHI or SRF block
        # see Klaus's doc or Borgani et al. 2003 for detials.
        mhi = readsnap(self.filename, "MHI ", quiet=True)
        ids_ex = None
        if mhi is None:
            # try exclude sfr gas particles
            sfr = readsnap(self.filename, "SFR ", quiet=True)
            if (sfr is not None) and (self.cut_sfr is not None):
                sfr = sfr[ids]
                ids_ex = sfr < self.cut_sfr
            else:
                ids_ex = np.ones(self.rho.size, dtype=bool)
            if (self.cut_rhoT[0] is not None) and (self.cut_rhoT[1] is not None):
                ids_ex = ids_ex & ((self.temp > self.cut_rhoT[1]) | (self.rho < self.rhoT[0]))
        else:
            mhi = mhi[ids] / 0.76 / self.mass
            if (self.cut_rhoT[0] is not None) and (self.cut_rhoT[1] is not None):
                ids_ex = (self.temp < self.cut_rhoT[1]) & (self.rho > self.rhoT[0])
                ids_ex = (mhi < 0.1) & (~ids_ex)
            else:
                ids_ex = (mhi < 0.1)
            self.rho *= (1 - mhi)  # correct multi-phase baryon model by removing cold gas

        if ids_ex is not None:
            self.temp = self.temp[ids_ex]     # cgs
            if not isinstance(self.mass, type(0.0)):
                self.mass = self.mass[ids_ex]
            if not isinstance(self.X, type(0.0)):
                self.X = self.X[ids_ex]
            self.pos = self.pos[ids_ex]
            self.vel = self.vel[ids_ex]
            self.rho = self.rho[ids_ex]
            self.ne = self.ne[ids_ex]
            if not isinstance(self.metal, type(0.0)):
                self.metal = self.metal[ids_ex]
            if self.hsml is not None:
                self.hsml = self.hsml[ids_ex]
            else:
                self.hsml = (3 * self.mass / self.rho / 4 / np.pi)**(1. / 3.)  # approximate
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
        if len(self.Tszdata) == 0 or force_redo:  # only need to prepare once
            constTsz = 1.0e10 * M_sun / self.cosmology["h"] * Kb * cs / me / Mp / c**2 / Kpc**2
            self.Tszdata = constTsz * self.mass * self.temp * self.X * self.ne
            # if self.mu is None:
            #     # self.Tszdata = constTsz * self.mass * self.temp / self.mmw / self.ne
            # else:
            #     self.Tszdata = constTsz * self.mass * self.temp / self.mu / self.ne
            # now Tszdata is dimensionless y_i, and can divided pixel size in kpc/h directly later

    def prep_ss_KT(self, vel):
        # need to calculate Kszdata for each projection, because vel chagnes!!
        constKsz = 1.0e15 * M_sun / self.cosmology["h"] * cs / Mp / c / Kpc**2  # velocity in km/s -> cm/s
        self.Kszdata = constKsz * self.mass * vel * self.X * self.ne
        # if self.mu is None:
        #     self.Kszdata = constKsz * self.mass * vel / self.mmw / self.ne
        # else:
        #     self.Kszdata = constKsz * self.mass * vel / self.mu / self.ne

    # prepare for mock observation model calculations
    def prep_ss_SZ(self, force_redo=False):
        if len(self.tau) == 0 or force_redo:
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
