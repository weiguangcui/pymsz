# Testing

pymsz has a test suite in `test/`, run with [pytest](https://docs.pytest.org/). The same tests run
automatically on GitHub for every push and pull request.

## Running the tests

From the root of the repository:

```bash
pip install -e ".[test]"        # pymsz in development mode, plus pytest and pyyaml
pytest                          # all tests (about 150; a minute or two)
pytest test/test_spectra.py     # one file
pytest -k telescope             # the tests whose names contain 'telescope'
pytest -x                       # stop at the first failure
```

pytest prints which pymsz it imported and whether the SZpack extension is built:

```text
pymsz 1.0 imported from /home/me/pymsz/pymsz
SZpack extension: built
```

The tests that need the compiled SZpack extension are **skipped** if it is not built. The tests on
simulation data use the test cluster `test/snap_CL288.hdf5` (and its Gadget binary version
`test/snap_CL288`), and are skipped if the file is missing.

Environment variables:

| Variable | Effect |
|---|---|
| `PYMSZ_REQUIRE_SZPACK=1` | fail, instead of skip, the SZpack tests if the extension is not built (used on GitHub) |
| `PYMSZ_TEST_NCPU` | number of CPUs for the SPH smoothing in the tests (default: up to 4) |

## What is tested

The tests check results against analytic limits and independent code paths, not against stored
numbers.

| File | Content |
|---|---|
| `test_spectra.py` | {mod}`pymsz.sz_spectra`: the tSZ null at 217 GHz, the Rayleigh-Jeans limit, kSZ as a temperature change, $\partial B_\nu/\partial T$, round trips of all units, `Jy/pixel`, interpolation. No simulation data. |
| `test_szpack.py` | the SZpack extension: non-relativistic limit at low temperature, `'combo'` against the exact `'3D'` integration, the relativistic shift of the null, the smaller decrement, the kinematic limit, `thermal_table`, the frequency range check. |
| `test_models.py` | {class}`~pymsz.load_data` (HDF5 and Gadget binary versions of the test cluster agree), {class}`~pymsz.TT_model` and {class}`~pymsz.TK_model` maps and spectra, the integrated *y* independent of the pixels, `SZpack_model` with `'nonrel'` against TT + TK, the relativistic SZpack result, FITS round trip. |
| `test_telescope.py` | every shipped bandpass and preset, band calibration, relativistic band integration, noise units and white-noise level, flux conservation by beam and pixelisation, seeds and the shared CMB, calibration errors, FITS output. Synthetic skies. |
| `test_pipeline.py` | the parameter template and the example file are valid, parameter and command-line errors, `--set`, lines of sight, YAML/TOML/JSON give the same run, catalogues, seeds, a full run against the same steps done by hand, and the shipped example `test/mock_CL288.yaml` through the command line. |

Shared settings and fixtures (the test cluster, small image geometry for speed) are in
`test/conftest.py`.

## Continuous integration on GitHub

The workflow `.github/workflows/tests.yml` runs on every push and pull request (and with the
**Run workflow** button in the **Actions** tab of the repository). It has two jobs:

**lint**
: `ruff` checks `pymsz/`, `test/` and `setup.py` for syntax errors and undefined names.

**tests**
: On Python 3.11, 3.13 and 3.14, GitHub checks out the repository with the SZpack submodule,
  installs GSL (`libgsl-dev`), installs pymsz with `pip install ".[test]"` (which compiles the SZpack
  extension), checks that the extension was built, and runs `pytest` with
  `PYMSZ_REQUIRE_SZPACK=1`.

The result is shown as a green tick or a red cross next to each commit and pull request. Click it, or
open the **Actions** tab, to see the log of each step.

## Adding tests

- Put new tests in the `test_*.py` file of the module they test, as functions named `test_...`.
- Use the fixtures of `test/conftest.py`: `cluster` (the loaded test cluster), `tt` and `tk` (its
  TT/TK models, a fresh copy for every test), and `szpack` (the compiled SZpack module; requesting it
  skips the test if the extension is not built).
- Keep images small (the shared geometry `GEO` uses 48 pixels) so that the suite stays fast.
- Prefer checks against physics (limits, conservation laws, two independent code paths) over
  comparisons with stored numbers.
