# Parameter file reference

The parameters of `pymsz-mock` and {func}`pymsz.pipeline.run`, section by section. Any parameter that
is left out takes its default value (`null` = not used). Lengths are in simulation units (comoving
kpc/h). Relative paths are relative to the directory of the parameter file. The full annotated
template is at the {ref}`end of this page <parameter-template>`.

## `simulation`

| Parameter | Default | Meaning |
|---|---|---|
| `snapshot` | — | Gadget binary snapshot (for multiple files the base name, e.g. `snap_128`) or Gadget/GIZMO HDF5 file |
| `name` | snapshot name | object name in the output files |
| `center` | `null` | centre `[x, y, z]` (kpc/h). `null` together with `radius: null`: the whole snapshot |
| `radius` | `null` | half the side of the region (kpc/h) |
| `restrict_r` | `true` | `false`: a cube of side 2 `radius` (fills the square image); `true`: a sphere |
| `cut_sfr` | `null` | exclude gas with star formation rate ≥ `cut_sfr` (M☉/yr) |
| `cut_rhoT` | `null` | `[rho, T]`: exclude cold dense gas (denser than `rho` and colder than `T`) |
| `hmrad` | `null` | radius (kpc/h) for the bulk velocity that is removed; `null`: all gas; ≤ 0: not removed |
| `metal` | `null` | gas metallicity, if the snapshot has none |
| `Nmets` | `11` | number of elements in the Gadget binary metallicity block |
| `mu` | `null` | mean molecular weight, if the snapshot has no electron abundance |
| `objects` | `null` | list of objects: `{name, snapshot, center, radius}` plus any `image` parameter |
| `catalogue` | `null` | text file with columns `x y z radius` (required) and `name snapshot redshift` (optional) |

See {doc}`../user_guide/loading_data`.

## `image`

| Parameter | Default | Meaning |
|---|---|---|
| `redshift` | `null` | redshift of the object, or a list (one set of maps each). `null`: the snapshot redshift (must be > 0) |
| `AR` | `null` | pixel size in arcsec. Use pixels several times smaller than the beams |
| `npixel` | `500` | pixels per side; `auto`: the whole region at `AR` |
| `axis` | `z` | lines of sight: `x`, `y`, `z`, `[alpha, beta, gamma]`, `{angles: [...], label: ...}`, `{matrix: ..., label: ...}`, `{random: N, seed: 0}`, or a list of these |
| `SP` | `[194.95, 27.98]` | `[RA, Dec]` (deg) of the image centre, for the FITS WCS |
| `sph_kernel` | `cubic` | `cubic`, `quartic`, `quintic`, `wendland2`, `wendland4`, `wendland6` |
| `SD` | `2` | SPH smoothing in 2 or 3 dimensions |
| `neighbours` | `null` | smooth every particle over this many neighbours instead of its smoothing length |
| `zthick` | `null` | use only gas within ±`zthick` (kpc/h) of the centre along the line of sight |
| `Ncpu` | `null` | processes for the SPH smoothing (`null`: all CPUs) |
| `Ntasks` | `null` | number of tasks for the SPH smoothing (`null`: `Ncpu`) |
| `Memreduce` | `false` | lower-memory parallel smoothing |

See {doc}`../user_guide/theoretical_models` and {doc}`../user_guide/pipeline`.

## `sz`

| Parameter | Default | Meaning |
|---|---|---|
| `model` | `szpack` | `szpack` (SZpack_model), `tt+tk` (TT_model + TK_model), `tt` or `tk` |
| `method` | `combo` | SZpack method: `combo`, `asymptotic`, `CNSN`, `CNSN_opt`, `nonrel`, `means`, `3D`, `5D`, `precise` |
| `projection` | `particle` | `szpack`: `particle` (exact sum along the line of sight) or `pixel` |
| `kinetic` | `true` | include the kSZ effect |
| `relativistic` | `true` | `tt`: relativistic tSZ at the *y*-weighted temperature (needs SZpack); `false`: non-relativistic |
| `bulk_velocity` | `null` | `szpack`: velocity added to the gas: `null`, `true` (the removed halo motion) or `[vx, vy, vz]` km/s |
| `szpack_options` | `null` | SZpack accuracy settings, e.g. `{T_order: 20, beta_order: 2}` |
| `Te_grid` | `null` | `szpack`: temperature nodes (keV) of the spectral table |
| `frequencies` | `null` | extra frequencies (GHz) for the saved model cube |
| `unit` | `MJy/sr` | unit of the saved model cube: `MJy/sr`, `Jy/pixel`, `K`, `uK`, `Dn` |

See {doc}`../user_guide/szpack_model`.

## `telescopes`

A list. Each entry is a preset name (`SPT-3G`) or a mapping:

| Key | Meaning |
|---|---|
| `preset` | preset name: `SPT-SZ`, `SPT-3G`, `ACT-DR6`, `Planck`, `SO-LAT`, `SO-LAT-goal`, `CMB-S4`, `FYST` (= `CCAT`) |
| `bands` | with `preset`: band names or nominal frequencies to keep. Without: a list of Band parameters |
| `band_options` | Band parameters for all bands (`{nnodes: 13}`) or per band (`{f150: {bandpass: ACT-DR6/pa5_f150}}`) |
| `measured_bandpass` | `false`: top-hat bands instead of the measured ones (default `true`) |
| `name` | telescope name |
| `label` | name in the output files (needed if a telescope appears twice) |
| `pixel` | map pixel (arcmin); default: the preset's |
| `transfer` | map filtering: `[scan, ell]`, `[highpass, ell]`, `[lowpass, ell]`, `{kind: scan, ell: 300, order: 4}`, a list of these, or a file with columns `ell F_ell` |
| `red_noise_corr` | correlation of the 1/f noise between bands |
| `pointing_rms` | pointing jitter (arcsec) |
| `description`, `reference` | free text |
| `unit`, `noise`, `cmb`, `foregrounds`, `calibration`, `pad` | per-telescope values of the `observation` parameters |

Band parameters: `freq` (GHz), `fwhm` (arcmin), `noise`, `noise_unit` (`uK-arcmin`, `uK-deg`,
`MJy/sr-arcmin`, `kJy/sr-deg`), `name`, `bandwidth` (GHz) or `frac_bandwidth`, `bandpass` (shipped
name or file), `aperture` (`flat` or `RJ`), `shift` (GHz), `ell_knee`, `alpha_knee`, `calib_error`,
`beam` (file with columns `ell B_ell`), `nnodes`, `node_cut`. See {doc}`../user_guide/telescopes`.

## `observation`

| Parameter | Default | Meaning |
|---|---|---|
| `unit` | `uK_CMB` | `uK_CMB`, `K_CMB` or `MJy/sr` (IRAS convention) |
| `noise` | `true` | instrument and atmospheric noise |
| `cmb` | `null` | primary CMB: `null`, `camb` (needs camb), or a CAMB-style file (`ell`, `D_ell` in µK²) |
| `foregrounds` | `null` | FITS cube `(nband, ny, nx)` or `{band: FITS file}`, in `unit`, band-averaged, at the model resolution |
| `calibration` | `true` | random gain errors (Band `calib_error`) |
| `seed` | `null` | random seed; every map gets a seed derived from it. `null`: drawn and saved |
| `pad` | `null` | zero padding for the convolutions (fraction of the image size) |

## `output`

| Parameter | Default | Meaning |
|---|---|---|
| `directory` | `pymsz_mock` | output directory |
| `name_format` | `{object}_z{redshift}_{projection}_{telescope}` | file names of the mock maps (`.fits` is added) |
| `model_format` | `{object}_z{redshift}_{projection}_{model}` | file names of the model cubes; `model` is `SZpack`, `tSZ` or `kSZ` |
| `save_model` | `true` | also write the SZ model cubes |
| `components` | `true` | also write the SZ, CMB, foreground and noise maps (FITS extensions) |
| `overwrite` | `false` | replace existing files |
| `skip_existing` | `false` | skip maps whose files exist (to resume a run) |
| `save_params` | `true` | write the parameters actually used to `<directory>/<file>.resolved.yaml` |

(parameter-template)=
## The annotated template

`pymsz-mock --template my_mock.yaml` writes this file:

```{literalinclude} ../../pymsz/data/mock_params_template.yaml
:language: yaml
```
