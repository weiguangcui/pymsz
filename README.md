# pymsz: mock Sunyaev-Zel'dovich observations of simulations

[![tests](https://github.com/weiguangcui/pymsz/actions/workflows/tests.yml/badge.svg)](https://github.com/weiguangcui/pymsz/actions/workflows/tests.yml)
[![docs](https://readthedocs.org/projects/pymsz/badge/?version=latest)](https://pymsz.readthedocs.io)

pymsz makes Sunyaev-Zel'dovich (SZ) maps of galaxy clusters and other objects in hydrodynamical
simulations, and turns them into mock observations by (sub-)millimetre telescopes.

**Documentation: <https://pymsz.readthedocs.io>**: installation, user guide, tutorial, parameter
reference and the full API.

Version 1.0 is the first public release. The changes are listed in the
[changelog](https://pymsz.readthedocs.io/en/latest/changelog.html).

## Features

- **Theoretical maps:** Compton-*y* (`TT_model`, with the *y*-weighted temperature) and kinetic SZ
  *b* = τ v<sub>los</sub>/c (`TK_model`), SPH-smoothed from the gas particles. Both can be converted
  into SZ images at any frequency, the tSZ optionally with relativistic corrections.
- **Relativistic SZ with [SZpack v2](https://github.com/CMBSPEC/SZpack):** `SZpack_model` computes
  thermal + kinetic SZ, summed exactly particle by particle along the line of sight.
- **Mock telescopes:** `Telescope.observe` integrates over the bandpasses, then adds the primary CMB,
  the beam, map filtering, pixelisation, calibration errors and white + 1/f noise. Presets: Planck,
  ACT DR6, SPT-SZ, SPT-3G, Simons Observatory, CMB-S4 and FYST (CCAT).
- **Measured bandpasses** of Planck, ACT DR6, SPT-SZ and SPT-3G are included.
- **One parameter file, many maps:** `pymsz-mock` runs the whole chain for many objects, redshifts,
  lines of sight and telescopes, from a YAML, TOML or JSON file.
- **Simulation formats:** Gadget binary (formats 1 and 2) and Gadget/GIZMO HDF5 snapshots.

## Installation

```bash
pip install "pymsz[yaml]"
```

This installs pymsz with numpy, scipy, astropy, h5py, psutil and PyYAML. Ready-made packages
(wheels) exist for Linux (x86_64, aarch64) and macOS (Apple silicon) with Python 3.10-3.14. On other
systems pip compiles pymsz, which needs a C++ compiler and [GSL](https://www.gnu.org/software/gsl/)
(e.g. `sudo apt install libgsl-dev`, `brew install gsl` or `conda install -c conda-forge gsl`).

Optional extras: `pymsz[cmb]` (CAMB, for CMB realisations), `pymsz[yt]` (loading data with yt,
experimental) and `pymsz[test]` (pytest). To install the development version from GitHub:

```bash
git clone --recurse-submodules https://github.com/weiguangcui/pymsz.git
cd pymsz
pip install -e ".[yaml,test]"
```

See the [installation guide](https://pymsz.readthedocs.io/en/latest/installation.html) for details
and troubleshooting.

## Quick example

The test cluster `test/snap_CL288.hdf5` is in the [GitHub repository](https://github.com/weiguangcui/pymsz)
(it is not part of the pip package).

```python
import numpy as np
import pymsz

simd = pymsz.load_data('test/snap_CL288.hdf5', snapshot=True,
                       center=np.array([500095.89, 500286.04, 500481.30]), radius=1446.17,   # kpc/h
                       restrict_r=False, cut_sfr=0.1)

tt = pymsz.TT_model(simd, npixel=256, AR=10, redshift=0.1)    # Compton-y map: tt.ydata
tk = pymsz.TK_model(simd, npixel=256, AR=10, redshift=0.1)    # kinetic SZ map: tk.bdata

tel = pymsz.Telescope.preset('SPT-3G')
obs = tel.observe([tt, tk], seed=1)                           # SPT-3G maps in uK_CMB
obs.write_fits('CL288_SPT3G.fits')

mm = pymsz.SZpack_model(simd, tel.frequencies(), npixel=256, AR=10, redshift=0.1)   # relativistic
obs = tel.observe(mm)
```

Or from a parameter file:

```bash
pymsz-mock --template my_mock.yaml       # annotated template with every parameter; edit it
pymsz-mock my_mock.yaml --dry-run        # check the parameters and print what will be done
pymsz-mock my_mock.yaml                  # run
pymsz-mock test/mock_CL288.yaml          # example: the test cluster seen by SPT-3G and ACT DR6
```

## Tests

```bash
pip install -e ".[yaml,test]"
pytest
```

The tests also run automatically on GitHub for every push and pull request. See
[Testing](https://pymsz.readthedocs.io/en/latest/testing.html).

## Citing

Please cite the papers for the parts of pymsz you use:

- thermal SZ maps: [Cui et al. 2018](https://doi.org/10.1093/mnras/sty2111);
- kinetic SZ maps: [Baldi et al. 2018](https://ui.adsabs.harvard.edu/abs/2018MNRAS.479.4028B);
- SZ signal with SZpack: [Chluba et al. 2012](https://ui.adsabs.harvard.edu/abs/2012MNRAS.426..510C),
  [Chluba et al. 2013](https://ui.adsabs.harvard.edu/abs/2013MNRAS.430.3054C) and, for SZpack v2,
  [Lee, Chluba & Holder 2022](https://arxiv.org/abs/2112.10666);
- measured bandpasses and telescope presets: the instrument papers listed in the documentation.

## License

Copyright (C) 2017-2026 Weiguang Cui and contributors.

pymsz is free software: you can redistribute it and/or modify it under the terms of the
[GNU General Public License](https://github.com/weiguangcui/pymsz/blob/master/LICENSE) as published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version. It is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
A PARTICULAR PURPOSE. See the [LICENSE](https://github.com/weiguangcui/pymsz/blob/master/LICENSE) file for details.

The SZpack sources (git submodule `SZpack/`) and the measured bandpasses in `pymsz/data/bandpasses/`
come from other authors and keep their own terms: see `SZpack/README.md` and
`pymsz/data/bandpasses/README.md`.

## Contact

Please report problems and ask questions on the
[issue tracker](https://github.com/weiguangcui/pymsz/issues), or write to Weiguang Cui
(cuiweiguang@gmail.com). Contributions are welcome as pull requests.
