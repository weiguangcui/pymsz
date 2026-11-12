# Loading simulation data

All pymsz models start from a {class}`~pymsz.load_data` object, which holds the gas particles of one
region of a snapshot.

```python
import numpy as np
import pymsz

simd = pymsz.load_data('snap_128.hdf5', snapshot=True,
                       center=np.array([500000., 500000., 500000.]),   # kpc/h
                       radius=2000.,                                   # kpc/h
                       restrict_r=False, cut_sfr=0.1)
```

## Supported formats

Gadget binary snapshots (`snapshot=True`)
: Formats 1 and 2 (with block names), single or multiple files. For multiple files give the base
  name (`snap_128` for `snap_128.0`, `snap_128.1`, …). The blocks read are `POS`, `VEL`, `MASS`,
  `RHO`, `HSML`, `NE`, `TEMP` (or the internal energy), `Z`/`Zs` (metals), and `SFR` or `MHI` (to
  remove star-forming or multiphase gas).

Gadget / GIZMO HDF5 snapshots (`snapshot=True`, file name ending in `.hdf5` or `.hdf`)
: Reads `PartType0`: `Coordinates`, `Velocities`, `Masses`, `Density`, `SmoothingLength`,
  `ElectronAbundance`, `Temperature` or `InternalEnergy`, `Metallicity` (GIZMO) or
  `GFM_Metallicity`/`GFM_Metals` (Illustris-like), `StarFormationRate`, and `DelayTime`. Particles with
  `DelayTime > 0` (decoupled wind particles) are removed.

yt (`yt_load=True`, experimental)
: Loads the data with [yt](https://yt-project.org/). yt is an optional dependency, not installed with
  pymsz: `pip install yt` (or `pip install ".[yt]"`). Only {class}`~pymsz.TT_model` accepts yt data;
  {class}`~pymsz.TK_model` and {class}`~pymsz.SZpack_model` need `snapshot=True`. This path is not
  maintained. Use the snapshot readers when you can.

## Units

pymsz assumes the usual Gadget units:

| Quantity | Unit |
|---|---|
| positions, `center`, `radius`, smoothing lengths | comoving kpc/h |
| masses | $10^{10}\,M_\odot/h$ |
| velocities in the snapshot | km/s × $\sqrt{a}$ (Gadget convention); converted to peculiar km/s |
| temperatures | K |

The cosmology (redshift, $h$, $\Omega_m$, $\Omega_\Lambda$, box size) is read from the snapshot header
into `simd.cosmology`.

## Selecting the region

`center` and `radius`
: Give both (comoving kpc/h) to select a region. Give neither to load the whole snapshot; the centre is
  then the median gas position.

`restrict_r`
: `True` (default) keeps a sphere of `radius`. `False` keeps a cube of side `2 radius`, which fills a
  square image. For maps use `restrict_r=False`.

Periodic boundaries
: If the region crosses a face of the simulation box, the particle positions are wrapped so that the
  region is contiguous.

The positions are stored relative to `center` (`simd.pos`).

## Removing gas that does not emit SZ

`cut_sfr`
: Exclude gas particles whose star formation rate is at or above this value (M☉/yr). Star-forming gas
  in sub-grid models has an effective temperature and density that are not physical. Typical value:
  `0.1`.

`cut_rhoT = [rho, T]`
: Exclude gas that is both denser than `rho` and colder than `T` (in simulation units and K).

Multiphase gas (Gadget binary, `MHI` block)
: If the snapshot contains the `MHI` block of the multiphase star-formation model, particles with a
  cold-gas fraction ≥ 0.1 are removed and the density of the others is corrected for the cold phase.

## Gas properties

Electron abundance
: From the `NE` / `ElectronAbundance` block. Without it, the gas is assumed to be fully ionised,
  unless a mean molecular weight `mu` is given.

Hydrogen mass fraction
: From the metallicity block: $X = 1 - Z - Y$, with the helium fraction $Y$ from the snapshot when it
  is there (Gadget `Zs` block, GIZMO `Metallicity[:, 1]`), else $Y = 0.24$. If the snapshot has no
  metals, give `metal` (a number or one value per particle). `Nmets` is the number of elements in the
  Gadget binary metal block.

Temperature
: From the `TEMP` / `Temperature` block, or from the internal energy with the mean molecular weight
  ($\gamma = 5/3$).

Smoothing lengths
: From the snapshot. If missing, $h = (3m / 4\pi\rho)^{1/3}$.

## The halo motion

The kinetic SZ effect depends on the gas velocity along the line of sight. By default the bulk
velocity of the object is removed from all velocities, so the kSZ map shows the internal motions:

| `hmrad` | Bulk velocity removed |
|---|---|
| `None` (default) | mean velocity of all loaded gas |
| `> 0` | mean velocity of the gas within `hmrad` (kpc/h) of the centre |
| `<= 0` | none: the velocities are those of the simulation frame |

The removed velocity is kept in `simd.bulkvel`. {class}`~pymsz.SZpack_model` can add it back, or add
any other velocity, with `bulk_velocity` (see {doc}`szpack_model`).

## What is stored

| Attribute | Content |
|---|---|
| `pos` | positions relative to `center`, comoving kpc/h, shape (N, 3) |
| `vel` | peculiar velocities, km/s, bulk motion removed |
| `temp` | temperature, K |
| `rho`, `mass`, `hsml` | density, mass, smoothing length (simulation units) |
| `ne`, `X`, `metal` | electron abundance per hydrogen atom, hydrogen mass fraction, metallicity |
| `cosmology` | dict: `z`, `a`, `h`, `omega_matter`, `omega_lambda`, `boxsize` |
| `center`, `radius`, `bulkvel` | the region and the removed bulk velocity |

The models call `simd.prep_ss_TT()`, which adds the number of electrons of every particle
(`electron_number`) and its Compton-*y* contribution per kpc² (`Tszdata`).

The models convert positions to physical kpc with the **snapshot** redshift,
$x_\mathrm{phys} = x / h / (1 + z_\mathrm{snap})$, whatever redshift the object is placed at for the
observation.
