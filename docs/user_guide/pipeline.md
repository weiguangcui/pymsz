# Mock observations from a parameter file

The pipeline ({mod}`pymsz.pipeline`) runs the whole chain from one parameter file:

```text
for every object:                     load_data (gas around the object)
  for every redshift:
    for every line of sight:          SZ model (SZpack_model, or TT_model + TK_model)
      for every telescope:            Telescope.observe
                                      -> one FITS file
```

The file is written in YAML (needs PyYAML), TOML or JSON. All parameters are listed in
{doc}`../reference/parameters`.

## Command line

```bash
pymsz-mock --template my_mock.yaml       # annotated template with every parameter
pymsz-mock my_mock.yaml --dry-run        # check the parameters and print what will be done
pymsz-mock my_mock.yaml                  # run
pymsz-mock my_mock.yaml --set image.npixel=256 --set 'telescopes=[SPT-3G, Planck]'
python -m pymsz my_mock.yaml             # the same as pymsz-mock
```

Always do a `--dry-run` first. It checks every parameter and stops at an unknown or misspelt one,
with a suggestion. It then prints the objects (and whether their snapshots exist), the images, lines
of sight, SZ model, frequencies, telescopes and output files. The options are listed in
{doc}`../reference/cli`.

## A parameter file

```yaml
simulation:
  snapshot: snap_128.hdf5
  name: CL0001
  center: [500000., 500000., 500000.]     # kpc/h
  radius: 2000.                           # kpc/h
  restrict_r: false
  cut_sfr: 0.1

image:
  redshift: [0.1, 0.3]                    # one set of maps per redshift
  AR: 5.                                  # arcsec
  npixel: 512
  axis: [x, y, z]                         # three lines of sight

sz:
  model: szpack                           # or tt+tk
  method: combo

telescopes:
  - SPT-3G
  - {preset: ACT-DR6, transfer: [scan, 300]}
  - {preset: Planck, bands: [HFI143, HFI353]}

observation:
  unit: uK_CMB
  noise: true
  cmb: camb
  seed: 1

output:
  directory: mock_maps
```

This makes 2 redshifts × 3 lines of sight = 6 SZ models, each observed by 3 telescopes: 18 mock
maps, plus the 6 model cubes. Relative paths are relative to the directory of the parameter file.

The sections are:

`simulation`
: The snapshot and the object(s): see {doc}`loading_data`.

`image`
: The SZ images of the model: redshift, pixel size, number of pixels, lines of sight, SPH settings.
  See {doc}`theoretical_models`.

`sz`
: The SZ model. `szpack` is {class}`~pymsz.SZpack_model` (relativistic tSZ + kSZ, exact). `tt+tk` is
  {class}`~pymsz.TT_model` + {class}`~pymsz.TK_model` (with `relativistic: true`, the tSZ includes
  the relativistic corrections at the *y*-weighted temperature). `tt` or `tk` is one of them.

`telescopes`
: A list of presets, presets with changes, or your own instruments. See {doc}`telescopes`.

`observation`
: Unit, noise, CMB, foregrounds, calibration errors, random seed.

`output`
: Directory, file names, and what to save.

## Several objects

List them in `simulation.objects`. Each object can set its own snapshot and any `image` parameter:

```yaml
simulation:
  cut_sfr: 0.1
  restrict_r: false
  objects:
    - {name: CL0001, snapshot: NewMDCLUSTER_0001/snap_128, center: [500000., 500000., 500000.], radius: 1500.}
    - {name: CL0002, snapshot: NewMDCLUSTER_0002/snap_128, center: [500100., 499800., 500050.], radius: 1400.,
       redshift: 0.3}
```

Or give a text catalogue. Its first line names the columns: `x y z radius` are required; `name`,
`snapshot` and `redshift` are optional. Other columns are ignored, and snapshot paths are relative to
the catalogue.

```yaml
simulation:
  snapshot: snap_128            # default snapshot for rows without one
  catalogue: clusters.txt
```

```text
# name    snapshot                     x          y          z          radius
CL0001    NewMDCLUSTER_0001/snap_128   500000.0   500000.0   500000.0   1500.0
CL0002    NewMDCLUSTER_0002/snap_128   500100.0   499800.0   500050.0   1400.0
```

The snapshot of an object is loaded once for all its redshifts and lines of sight.

## Lines of sight

`image.axis` takes

| Value | Lines of sight (label in the file names) |
|---|---|
| `z` | along a simulation axis (`z`) |
| `[0., 45., 30.]` | one rotation by these angles (deg) around x, y, z (`a0_45_30`) |
| `{angles: [0., 45., 30.], label: view1}` | the same with a label (`view1`) |
| `{matrix: [[...], [...], [...]], label: m1}` | a rotation matrix |
| `{random: 10, seed: 0}` | 10 orientations drawn uniformly on the sphere (`rand000` … `rand009`) |
| `[x, z, {random: 4}]` | a list of these |

A list of three numbers is always one rotation, not three lines of sight.

## Telescopes

```yaml
telescopes:
  - SPT-3G                                   # a preset
  - preset: ACT-DR6                          # a preset with changes
    measured_bandpass: true
    bands: [f090, f150]                      # keep only these bands
    band_options:
      f150: {bandpass: ACT-DR6/pa5_f150}
      f090: {beam: beams/act_f090_bl.txt}    # text file: ell, B_ell
    transfer: [scan, 300]
    label: ACT-pa5                           # name in the output files
    cmb: camb                                # any observation parameter except seed
  - name: MyCamera                           # your own instrument
    pixel: 0.25
    bands:
      - {name: f150, freq: 150., fwhm: 1.0, noise: 3., frac_bandwidth: 0.25, ell_knee: 2000.}
      - {name: f220, freq: 220., fwhm: 0.8, noise: 10., bandpass: my_220GHz_band.txt}
```

The SZ model is computed once, at the band-integration frequencies of all telescopes
({meth}`Telescope.frequencies <pymsz.Telescope.frequencies>`), plus `sz.frequencies` if given. Each
telescope then observes it. Labels must be unique, so add `label` if the same preset appears twice.

## Output files

Each mock observation is written to `output.directory` as

```text
{object}_z{redshift}_{projection}_{telescope}.fits     e.g. CL0001_z0.1_x_SPT-3G.fits
```

(`output.name_format`). With `save_model: true` (default), the SZ model is also written, as
`{object}_z{redshift}_{projection}_{model}.fits`, with `model` = `SZpack`, `tSZ` or `kSZ`
(`output.model_format`). It includes the *y*, τ, $T_e$ and *b* maps. The FITS layout is described in
{doc}`conventions`. The mock maps also get the header keywords `OBJECT`, `PROJ`, `SZMODEL`,
`SNAPSHOT` and `PARFILE`.

Existing files stop the run, unless `overwrite: true` (replace them) or `skip_existing: true` (keep
them and compute only the missing ones, to resume an interrupted run).

## Reproducibility

Seeds
: Every map gets its own seed, derived from `observation.seed` and the object, redshift, line of
  sight and telescope names. It is written to the FITS keyword `SEED`. The result of a map therefore
  does not depend on the order of the loops or on which other maps are made. All telescopes see the
  same CMB sky for a given image (keyword `CMBSEED`), with independent noise. With `seed: null`, a
  random seed is drawn and saved.

Provenance
: With `save_params: true` (default), the parameters actually used are written next to the maps
  (`<parameter file>.resolved.yaml`), with all defaults, absolute paths and the drawn seed. Running
  this file again reproduces the maps.

## From Python

```python
from pymsz import pipeline

recs = pipeline.run('my_mock.yaml')                                  # one record per map file
recs = pipeline.run('my_mock.yaml', overrides={'image.npixel': 256})
recs = pipeline.run(params_dict)                                     # a dict instead of a file

pipe = pipeline.Pipeline('my_mock.yaml')
print(pipe.describe())                                               # what --dry-run prints
recs = pipe.run(keep_results=True)                                   # also keep the MockObservations
```

Every record is a dict with `object`, `redshift`, `projection`, `telescope`, `file`, `model_files`,
`seed`, `cmb_seed` and `observation` (the {class}`~pymsz.MockObservation` if `keep_results=True`).

`callback=f` calls `f(obs, info)` after every observation, with `info` the record plus `models` (the
SZ models). Use it to analyse the maps while the run goes on:

```python
def measure(obs, info):
    print(info['object'], info['telescope'], obs.sz.min(axis=(1, 2)))

pipeline.run('my_mock.yaml', callback=measure)
```
