# Quickstart

This page shows the two ways of using pymsz: from Python, and with a parameter file. Both use the
test cluster `test/snap_CL288.hdf5` that comes with the source code. It is a cluster from The Three
Hundred project, simulated with GIZMO-Simba. Its centre and $R_{200}$ are taken from the AHF halo
catalogue.

## From Python

### 1. Load the gas around the object

```python
import numpy as np
import pymsz

center = np.array([500095.89011853, 500286.04465527, 500481.30095660])   # kpc/h, comoving
simd = pymsz.load_data('test/snap_CL288.hdf5', snapshot=True,
                       center=center, radius=1446.17,     # kpc/h
                       restrict_r=False,                  # a cube of side 2 radius
                       cut_sfr=0.1)                       # drop star-forming gas
```

{class}`~pymsz.load_data` reads the gas positions, velocities, temperatures, densities, electron
abundances, metallicities and smoothing lengths. Lengths are in simulation units (comoving kpc/h).
See {doc}`user_guide/loading_data`.

### 2. Make SZ maps

```python
tt = pymsz.TT_model(simd, npixel=256, AR=10, redshift=0.1)   # thermal SZ
tk = pymsz.TK_model(simd, npixel=256, AR=10, redshift=0.1)   # kinetic SZ

tt.ydata     # Compton-y map, (256, 256)
tt.Tydata    # y-weighted electron temperature (keV)
tk.bdata     # b = tau v_los / c, > 0 for gas moving towards the observer
```

The cluster is placed at redshift 0.1 and imaged with 10 arcsec pixels (`AR`). To get the SZ signal
at observed frequencies:

```python
tt.to_sz([90., 150., 220.], unit='uK')                     # non-relativistic tSZ
tt.to_sz([90., 150., 220.], unit='uK', relativistic=True)  # with relativistic corrections (SZpack)
tk.to_sz([90., 150., 220.], unit='uK')
total = tt.szdata + tk.szdata                              # (3, 256, 256) in micro-K
tt.write_sz_fits('CL288_tSZ.fits')
```

For the exact relativistic thermal plus kinetic signal, use {class}`~pymsz.SZpack_model`, which needs
the compiled SZpack extension:

```python
mm = pymsz.SZpack_model(simd, [90., 150., 220., 353.], method='combo',
                        npixel=256, AR=10, redshift=0.1)
mm.szdata          # (4, 256, 256) in MJy/sr
mm.write_fits_image('CL288_SZpack.fits')
```

See {doc}`user_guide/theoretical_models` and {doc}`user_guide/szpack_model`.

### 3. Observe with a telescope

```python
tel = pymsz.Telescope.preset('SPT-3G')
obs = tel.observe([tt, tk], cmb='camb', seed=1)   # tSZ + kSZ + primary CMB + noise
obs.maps            # (3, ny, nx) in uK_CMB, on 0.25 arcmin pixels
obs.sz, obs.cmb, obs.noise                        # the separate components
obs.write_fits('CL288_SPT3G.fits')
```

`cmb='camb'` needs `pip install camb`. Leave it out for SZ plus noise only. With SZpack, compute the
model at the frequencies needed for the band integration:

```python
mm = pymsz.SZpack_model(simd, tel.frequencies(), npixel=256, AR=10, redshift=0.1)
obs = tel.observe(mm, seed=1)
```

See {doc}`user_guide/telescopes`.

## With a parameter file

The command `pymsz-mock` (also `python -m pymsz`) runs the whole chain from one parameter file. Start
from the annotated template, which lists every parameter:

```bash
pymsz-mock --template my_mock.yaml     # write the template; edit it
pymsz-mock my_mock.yaml --dry-run      # check the parameters, print what will be done
pymsz-mock my_mock.yaml                # run
```

The example `test/mock_CL288.yaml` observes the test cluster with SPT-3G and ACT DR6:

```yaml
simulation:
  snapshot: snap_CL288.hdf5          # relative to this file
  name: CL288
  center: [500095.89011853, 500286.04465527, 500481.30095660]   # kpc/h
  radius: 1446.17                                               # kpc/h
  restrict_r: false
  cut_sfr: 0.1

image:
  redshift: 0.1
  AR: 10.
  npixel: 256
  axis: z

sz:
  model: szpack          # relativistic tSZ + kSZ; or tt+tk
  method: combo

telescopes:
  - SPT-3G
  - ACT-DR6

observation:
  unit: uK_CMB
  noise: true
  cmb: null
  seed: 1

output:
  directory: mock_CL288
  overwrite: true
```

```bash
pymsz-mock test/mock_CL288.yaml
ls test/mock_CL288/
# CL288_z0.1_z_ACT-DR6.fits  CL288_z0.1_z_SPT-3G.fits  CL288_z0.1_z_SZpack.fits  mock_CL288.resolved.yaml
```

See {doc}`user_guide/pipeline` and {doc}`reference/parameters`.

## Next steps

- {doc}`tutorials/quick_tour`: the same steps with figures.
- {doc}`user_guide/conventions`: units, sign conventions and the layout of the FITS files.
- {doc}`api/index`: every class and function.
