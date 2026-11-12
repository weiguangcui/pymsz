# pymsz: mock Sunyaev-Zel'dovich observations of simulations

**pymsz** makes Sunyaev-Zel'dovich (SZ) maps of galaxy clusters and other objects in hydrodynamical
simulations, and turns them into mock observations by (sub-)millimetre telescopes. This is the
documentation of version {{version}}.

:::{note}
Version 1.0 is the first public release of pymsz; see the {doc}`changelog`. Please report problems
and ask questions on the [issue tracker](https://github.com/weiguangcui/pymsz/issues).
:::

## What it does

```text
simulation snapshot ──load_data──▶ gas particles around an object
                    ──SZ model───▶ SZ images (y, τ, Te, v_los maps; ΔI or ΔT at any frequency)
                    ──Telescope──▶ bandpass, CMB, beam, filtering, pixels, gains, noise
                    ──▶ FITS files
```

- **Theoretical maps.** {class}`~pymsz.TT_model` projects the Compton-*y* map of the thermal SZ effect
  (and the *y*-weighted electron temperature). {class}`~pymsz.TK_model` projects the
  $b = \tau\, v_\mathrm{los}/c$ map of the kinetic SZ effect. Both can be converted into SZ images at
  any observed frequency.
- **Relativistic SZ with SZpack.** {class}`~pymsz.SZpack_model` computes the thermal plus kinetic SZ
  signal including relativistic corrections with [SZpack v2](https://github.com/CMBSPEC/SZpack),
  summed exactly particle by particle along the line of sight.
- **Mock telescopes.** {class}`~pymsz.Telescope` observes any of these models. It integrates over
  measured bandpasses, then adds the primary CMB, the beam, map filtering, pixelisation, calibration
  errors and white plus 1/f noise. Presets cover Planck, ACT DR6, SPT-SZ, SPT-3G, Simons Observatory,
  CMB-S4 and FYST (CCAT).
- **Measured bandpasses** of Planck, ACT DR6, SPT-SZ and SPT-3G ship with the package
  ({mod}`pymsz.bandpasses`).
- **Parameter-file pipeline.** The command `pymsz-mock` runs the whole chain for many objects,
  redshifts, lines of sight and telescopes from one YAML, TOML or JSON file ({mod}`pymsz.pipeline`).
- **Simulation formats.** Gadget binary snapshots (formats 1 and 2), Gadget/GIZMO HDF5 snapshots, and
  (experimental) any format readable by [yt](https://yt-project.org/).

## A first example

```python
import numpy as np
import pymsz

simd = pymsz.load_data('test/snap_CL288.hdf5', snapshot=True,
                       center=np.array([500095.9, 500286.0, 500481.3]), radius=1446.17,
                       cut_sfr=0.1, restrict_r=False)

tt = pymsz.TT_model(simd, npixel=256, AR=10, redshift=0.1)   # Compton-y map: tt.ydata
tk = pymsz.TK_model(simd, npixel=256, AR=10, redshift=0.1)   # kSZ b map:     tk.bdata

tel = pymsz.Telescope.preset('SPT-3G')
obs = tel.observe([tt, tk], seed=1)        # tSZ + kSZ seen by SPT-3G, in uK_CMB
obs.write_fits('CL288_SPT3G.fits')
```

or, without writing any code:

```bash
pymsz-mock test/mock_CL288.yaml
```

Start with {doc}`installation` and the {doc}`quickstart`. The {doc}`tutorials/index` show complete
worked examples with figures.

```{toctree}
:maxdepth: 2
:caption: Getting started
:hidden:

installation
quickstart
tutorials/index
```

```{toctree}
:maxdepth: 2
:caption: User guide
:hidden:

user_guide/loading_data
user_guide/theoretical_models
user_guide/szpack_model
user_guide/telescopes
user_guide/bandpasses
user_guide/pipeline
user_guide/conventions
```

```{toctree}
:maxdepth: 2
:caption: Reference
:hidden:

reference/parameters
reference/cli
reference/presets
api/index
```

```{toctree}
:maxdepth: 1
:caption: About
:hidden:

citing
contributing
testing
changelog
```
