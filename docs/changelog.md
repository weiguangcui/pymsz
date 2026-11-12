# Changes

## Version 1.0

The first public release. Compared with the earlier, unreleased versions:

New

- {class}`~pymsz.SZpack_model` with SZpack v2: relativistic thermal + kinetic SZ, computed with any
  SZpack method and summed particle by particle along the line of sight.
- `to_sz` for {class}`~pymsz.TT_model` and {class}`~pymsz.TK_model`: SZ images at any frequency from
  the *y* and *b* maps, the tSZ optionally with relativistic corrections at the *y*-weighted
  temperature (`TT_model.Tydata`).
- Mock telescope observations ({class}`~pymsz.Telescope`, {class}`~pymsz.Band`), with presets for
  Planck, ACT DR6, SPT-SZ, SPT-3G, Simons Observatory, CMB-S4 and FYST.
- The measured bandpasses of Planck, ACT DR6, SPT-SZ and SPT-3G ({mod}`pymsz.bandpasses`).
- Mock observations from a parameter file: `pymsz-mock` and {mod}`pymsz.pipeline`.
- A test suite and automatic tests on GitHub ({doc}`testing`), and this documentation.

Changed

- SZpack v2 replaces the old copy of SZpack v1.1.1 in `pymsz/SZpacklib`, which is no longer used. It
  is compiled with pybind11 from the git submodule `./SZpack`; SWIG is not needed any more.
- `pip install .` installs the required Python packages. PyYAML, CAMB and yt are optional extras
  (`.[yaml]`, `.[cmb]`, `.[yt]`).
- {class}`~pymsz.TK_model` uses the same pixel grid as {class}`~pymsz.TT_model` (the extent of the
  projected plane), so their maps can be added.
- All models use one set of physical constants ({mod}`pymsz.sz_spectra`, CODATA 2018), whether or not
  yt is installed, and compute the particle weights in double precision. The *y* and
  $b = \tau v_\mathrm{los}/c$ maps of TT_model / TK_model are therefore identical to the *y* and
  $\tau\beta_\parallel$ maps of SZpack_model, to $2\times10^{-14}$ per pixel. Compared with earlier
  versions, the TT_model *y* maps increase by $2.6\times10^{-7}$ and the TK_model *b* maps by
  $2.8\times10^{-8}$. With yt installed, earlier versions used yt's solar mass, so all maps change by
  $+3.4\times10^{-4}$.
