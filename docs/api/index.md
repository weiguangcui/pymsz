# API reference

The complete Python interface, generated from the docstrings.

## Main classes

These are available directly from the `pymsz` package (`pymsz.TT_model`, …).

```{eval-rst}
.. currentmodule:: pymsz

.. autosummary::
   :toctree: generated
   :nosignatures:

   load_data
   TT_model
   TK_model
   SZpack_model
   Telescope
   Band
   SkyMaps
   MockObservation
```

## Modules

| Module | Content |
|---|---|
| {doc}`pymsz.pipeline <pipeline>` | mock observations from a parameter file (`pymsz-mock`) |
| {doc}`pymsz.telescope <telescope>` | telescopes, bands, presets, CMB and noise realisations |
| {doc}`pymsz.bandpasses <bandpasses>` | the measured bandpasses shipped with pymsz |
| {doc}`pymsz.sz_spectra <sz_spectra>` | SZ spectra, unit conversions, physical constants |
| {doc}`pymsz.SZpack_models <szpack_models>` | `SZpack_model` and the SZpack spectra |
| {doc}`pymsz.szpack <szpack>` | the compiled SZpack v2 extension |
| {doc}`pymsz.rotate_data <rotate_data>` | rotations and SPH smoothing |
| {doc}`pymsz.readsnapsgl <readsnapsgl>` | Gadget and HDF5 snapshot reader |

```{toctree}
:hidden:
:maxdepth: 1

pipeline
telescope
bandpasses
sz_spectra
szpack_models
szpack
rotate_data
readsnapsgl
```
