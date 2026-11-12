# Installation

## With pip

```bash
pip install "pymsz[yaml]"
```

This is all most users need. Ready-made packages (*wheels*) exist for Linux (x86_64 and aarch64) and
macOS (Apple silicon and Intel) with Python 3.10 to 3.14. They include the compiled SZpack extension and GSL,
so nothing else is needed. On other systems (e.g. Windows), pip downloads the source package and
compiles it, which needs a C++ compiler and GSL: see [the requirements](#requirements) below.

pymsz contains a C++ extension (`pymsz.szpack.SZpack`), compiled from the
[SZpack v2](https://github.com/CMBSPEC/SZpack) sources. {class}`~pymsz.SZpack_model` and the
relativistic option of {class}`~pymsz.TT_model` need it; everything else works without it.

## Requirements

**Python 3.10 or newer.** The tests run with Python 3.11, 3.13 and 3.14 (see {doc}`testing`), and
every wheel is tested before it is published.

**Python packages.** pip installs the required ones. The optional ones are installed on request, as
*extras*:

| Package | Needed for | Installed by |
|---|---|---|
| `numpy`, `scipy`, `astropy` | everything | `pip install pymsz` |
| `h5py` | reading HDF5 snapshots; the ACT bandpass uncertainties and PA7 bands | `pip install pymsz` |
| `psutil` | the SPH smoothing (memory logging) | `pip install pymsz` |
| `pyyaml` | YAML parameter files of `pymsz-mock` (TOML and JSON need nothing extra) | `pip install "pymsz[yaml]"` |
| `camb` | primary CMB realisations with `cmb='camb'` | `pip install "pymsz[cmb]"` |
| `yt` | loading data with yt, `load_data(..., yt_load=True)` (experimental) | `pip install "pymsz[yt]"` |
| `pytest` | running the tests | `pip install "pymsz[test]"` |
| `matplotlib` | the plots in the tutorials | `pip install matplotlib` |

Extras can be combined, e.g. `pip install "pymsz[yaml,cmb]"`; from a source checkout, write `.`
instead of `pymsz` (`pip install ".[yaml,cmb]"`). You can also install an optional package later with
plain `pip install pyyaml` (or `camb`, `yt`): pymsz imports these packages only when a feature that
needs them is used, and the error message then says what to install.

**To compile the SZpack extension** (only without a wheel, or from source): a C++ compiler, the
Python headers (`python3-dev` on Debian/Ubuntu), and the
[GNU Scientific Library](https://www.gnu.org/software/gsl/) (GSL) with its headers. `pybind11` is
installed automatically for the build.

| System | GSL |
|---|---|
| Debian / Ubuntu | `sudo apt install libgsl-dev` |
| Fedora / RHEL | `sudo dnf install gsl-devel` |
| macOS (Homebrew) | `brew install gsl` |
| conda | `conda install -c conda-forge gsl` |
| none of these | `./build_szpack.sh --build-gsl` builds a private static copy (see below) |

## From source

For the development version, or to change the code:

```bash
git clone --recurse-submodules https://github.com/weiguangcui/pymsz.git
cd pymsz
pip install ".[yaml]"         # or, for development: pip install -e ".[yaml,test]"
```

`--recurse-submodules` also downloads the SZpack sources into `./SZpack` (a git submodule). If you
cloned without it, run

```bash
git submodule update --init
```

`pip install .` compiles the SZpack extension if it finds the SZpack sources, `pybind11` and GSL. If
one of them is missing, it prints a message (for example
`pymsz: GSL not found -> SZpack models are NOT built.`) and installs pymsz without the extension.

### Where GSL is searched for

`setup.py` takes the first match of:

1. `GSL_INC` and `GSL_LIB` (include and library directories), or `GSL_DIR` (an installation prefix);
2. the prefix reported by `gsl-config --prefix`;
3. `./third_party/gsl`, made by `./build_szpack.sh --build-gsl`;
4. `$CONDA_PREFIX`;
5. `/usr/local`, `/opt/local`, `/opt/homebrew`, `/usr`.

For example, for a GSL installed in your home directory:

```bash
GSL_DIR=$HOME/local pip install .
```

### Without a system GSL: `build_szpack.sh`

The script `build_szpack.sh` fetches the SZpack submodule if needed and builds the extension in place
(in `pymsz/szpack/`). With `--build-gsl`, it first downloads GSL 2.8, checks its SHA-256 checksum and
builds a static copy in `./third_party/gsl`:

```bash
./build_szpack.sh --build-gsl                       # private GSL + SZpack extension
PYTHON=/path/to/python ./build_szpack.sh            # use another Python
GSL_DIR=/opt/gsl ./build_szpack.sh                  # use a GSL installed under /opt/gsl
```

An in-place build is used when pymsz is run from the source directory (or installed with
`pip install -e .`). For a regular installation, run `pip install .` after `--build-gsl`: it then
finds `./third_party/gsl`.

## Check the installation

```python
import pymsz
from pymsz import szpack

print(pymsz.version.version)
print("SZpack extension available:", szpack.available)
```

If the extension is missing, {func}`pymsz.szpack.require` raises an `ImportError` that explains how to
build it. A quick end-to-end test is the example parameter file shipped in `test/`. It observes the
test cluster with SPT-3G and ACT DR6 and writes the maps to `test/mock_CL288/`:

```bash
pymsz-mock test/mock_CL288.yaml --dry-run      # check the parameters
pymsz-mock test/mock_CL288.yaml                # run (a few minutes)
```

The full test suite (`pip install -e ".[test]"`, then `pytest`) checks everything, see
{doc}`testing`.

## Troubleshooting

`ImportError: libgsl.so...: cannot open shared object file`
: The extension was linked against a shared GSL that the dynamic loader cannot find at run time.
  Add its library directory to `LD_LIBRARY_PATH` (`DYLD_LIBRARY_PATH` on macOS), or rebuild with a
  static GSL: `./build_szpack.sh --build-gsl`.

`The SZpack extension (pymsz.szpack.SZpack) is not built`
: Install GSL and pybind11, fetch the submodule (`git submodule update --init`) and reinstall
  (`pip install .`). The messages printed by `pip install -v .` say which part was not found.

`YAML parameter files need PyYAML`
: `pip install pyyaml`, or write the parameter file in TOML (`.toml`) or JSON (`.json`).

`load_data(yt_load=True) needs the yt package`
: `pip install yt`. Loading with yt is experimental; only `TT_model` accepts yt data.

The old SZpack v1.1.1 copy
: Earlier versions of pymsz compiled SZpack v1.1.1 from `pymsz/SZpacklib` with SWIG. This copy is no
  longer used; SWIG is not needed any more.

## Building this documentation

```bash
pip install -r docs/requirements.txt
make -C docs html          # then open docs/_build/html/index.html
```
