# Mock telescope observations

{class}`~pymsz.Telescope` turns the SZ images of a model into the maps that a (sub-)millimetre
telescope would make in each of its frequency bands.

```python
tel = pymsz.Telescope.preset('SPT-3G')
obs = tel.observe([tt, tk], cmb='camb', seed=1)
obs.maps          # (nband, ny, nx), uK_CMB
```

## What `observe` does

{meth}`Telescope.observe <pymsz.Telescope.observe>` applies, in order:

1. **Bandpass integration** of the SZ spectrum over every band, and calibration to CMB temperature
   (`uK_CMB`, `K_CMB`) or to `MJy/sr` (IRAS convention, $\nu I_\nu$ = constant).
2. **Other sky signals** (optional): a Gaussian realisation of the primary CMB, and foreground maps
   that you provide.
3. **Beam** (Gaussian FWHM or a measured $B_\ell$) and the **pointing jitter**.
4. **Map transfer function**, e.g. the high-pass filtering along the scan direction.
5. **Pixelisation** onto the telescope pixels (area-weighted, which includes the pixel window).
6. **Calibration errors**: a random gain per band.
7. **Noise**: white noise plus a $1/f$ (atmospheric) component, optionally correlated between bands.

All steps use the flat-sky approximation on the simulated patch.

The sky can be

- a {class}`~pymsz.TT_model` and/or {class}`~pymsz.TK_model`: `[tt, tk]` is tSZ + kSZ. They use the
  spectrum chosen in `to_sz` (non-relativistic by default);
- a {class}`~pymsz.SZpack_model`;
- a {class}`~pymsz.SkyMaps`: any image cube at a set of frequencies, from arrays or a FITS file;
- a list of these, which are added. They must have the same image and pixel sizes.

The SZ images need to be well sampled. Use model pixels (`AR`) several times smaller than the beam.

## Telescope presets

```python
pymsz.Telescope.list_presets()
# ['ACT-DR6', 'CCAT', 'CMB-S4', 'FYST', 'Planck', 'SO-LAT', 'SO-LAT-goal', 'SPT-3G', 'SPT-SZ']
tel = pymsz.Telescope.preset('ACT-DR6')
print(tel)
print(tel.reference)       # where the numbers come from
```

| Preset | Instrument |
|---|---|
| `Planck` | Planck full-mission LFI + HFI maps (9 bands) |
| `SPT-SZ` | SPT-SZ 2500 deg² survey |
| `SPT-3G` | SPT-3G Main field, 2019–2020 data |
| `ACT-DR6` | ACT DR6 wide survey |
| `SO-LAT`, `SO-LAT-goal` | Simons Observatory LAT wide survey, baseline and goal sensitivity |
| `CMB-S4` | CMB-S4 Chile LATs wide survey (2020 design) |
| `FYST` (= `CCAT`) | Fred Young Submillimeter Telescope, Prime-Cam wide survey |

The beam, noise and bandpass of every band, and the papers they come from, are listed in
{doc}`../reference/presets`. Beams and white-noise levels are published values. Some atmospheric
knees are approximations. Planck, SPT-SZ, SPT-3G and ACT-DR6 use the measured bandpasses shipped with
pymsz. SO, CMB-S4 and FYST use top-hat bands, because their bandpasses are not public.

### Changing a preset

```python
# keep some bands (by name or nominal frequency)
tel = pymsz.Telescope.preset('Planck', bands=['HFI143', 'HFI217', 'HFI353'])

# change all bands, or individual bands
tel = pymsz.Telescope.preset('SPT-3G', band_options={'nnodes': 13})
tel = pymsz.Telescope.preset('ACT-DR6', band_options={
    'f150': {'bandpass': 'ACT-DR6/pa5_f150'},          # one detector array
    'f090': {'beam': (ell, B_ell)},                     # measured beam
    'f220': {'shift': 1.5},                             # bandpass shifted by +1.5 GHz
})

# top-hat bands instead of the measured bandpasses
tel = pymsz.Telescope.preset('ACT-DR6', measured_bandpass=False)

# telescope properties: pixel, filtering, noise correlation, pointing
tel = pymsz.Telescope.preset('SPT-3G', transfer=('scan', 300), pixel=0.5)
```

`band_options` takes any {class}`~pymsz.Band` argument. Giving `bandwidth` or `frac_bandwidth`
replaces the measured bandpass by a top-hat; giving `bandpass` replaces a top-hat.

## Your own instrument

A {class}`~pymsz.Telescope` is a list of {class}`~pymsz.Band` objects (or dicts of their arguments)
plus map-level properties:

```python
tel = pymsz.Telescope(
    [pymsz.Band(150., fwhm=1.0, noise=3., frac_bandwidth=0.25, ell_knee=2000, name='f150'),
     pymsz.Band(220., fwhm=0.8, noise=10., bandpass='my_220GHz_band.txt', calib_error=0.01)],
    name='MyCamera', pixel=0.25, transfer=('scan', 300), red_noise_corr=0.5)
```

### Band parameters

| Parameter | Meaning |
|---|---|
| `freq` | nominal band centre (GHz); also the reference frequency of `MJy/sr` maps |
| `name` | band name (default `'f%03d' % freq`) |
| `fwhm` | Gaussian beam FWHM (arcmin); ignored if `beam` is given |
| `beam` | measured beam: `(ell, B_ell)` arrays or a function `B(ell)` |
| `noise`, `noise_unit` | white-noise level, in `'uK-arcmin'` (default), `'uK-deg'`, `'MJy/sr-arcmin'` or `'kJy/sr-deg'` |
| `ell_knee`, `alpha_knee` | $1/f$ noise: $N_\ell = N_\mathrm{white}\,[1 + (\ell/\ell_\mathrm{knee})^{\alpha_\mathrm{knee}}]$; default no $1/f$ noise, $\alpha_\mathrm{knee} = -3.5$ |
| `calib_error` | fractional (1σ) calibration uncertainty |
| `bandwidth` / `frac_bandwidth` | width of a top-hat bandpass, in GHz or as a fraction of `freq` |
| `bandpass` | measured bandpass: a shipped name (`'SPT-3G/f150'`), a {class}`~pymsz.bandpasses.Bandpass`, `(nu, response)` arrays, or a two-column text file |
| `aperture` | convention of the bandpass curve: `'flat'` (response per unit $I_\nu$, default) or `'RJ'` (response to a Rayleigh-Jeans source) |
| `shift` | shift of the whole bandpass (GHz) |
| `nnodes`, `node_cut` | sampling of the band integration (see below) |

Without `bandwidth`, `frac_bandwidth` or `bandpass` the band is monochromatic at `freq`. The
bandpass conventions are explained in {doc}`bandpasses`.

### Telescope parameters

`pixel`
: Map pixel size (arcmin). `None`: keep the pixels of the input images. The maps are rebinned with
  area weights. For input images of $n$ pixels of `AR` arcsec, the output has
  $\lfloor n\,\mathrm{AR} / (60\,\mathrm{pixel}) \rfloor$ pixels per side.

`transfer`
: The transfer function of the map making, applied to the sky signal (not to the noise, whose
  spectrum is that of the final maps):

  | Specification | Filter |
  |---|---|
  | `('scan', ell_c)` | high-pass in $\lvert\ell_x\rvert$, as for time streams filtered along scans in RA (the image x axis): $F = 1/[1 + (\ell_c/\lvert\ell_x\rvert)^{8}]$ |
  | `('highpass', ell_c)` | isotropic high-pass, $F = 1/[1 + (\ell_c/\ell)^{8}]$ |
  | `('lowpass', ell_c)` | isotropic low-pass, $F = 1/[1 + (\ell/\ell_c)^{8}]$ |
  | `{'kind': 'scan', 'ell': 300, 'order': 4, 'axis': 'x'}` | the same, with the filter order (exponent $2\times$`order`, default 4) and, for `'scan'`, the axis (`'x'` or `'y'`) |
  | `(ell, F_ell)` arrays | tabulated isotropic transfer function |
  | a function `F(lx, ly)` | anything else |
  | a list of these | their product |

`red_noise_corr`
: Correlation coefficient of the $1/f$ noise between bands: a number (same for all pairs) or an
  `(nband, nband)` matrix. Default 0.

`pointing_rms`
: RMS pointing jitter (arcsec), which broadens the effective beam.

## Observation options

`unit`
: `'uK_CMB'` (default), `'K_CMB'` or `'MJy/sr'`.

`noise`
: Add the instrument and atmospheric noise (default `True`).

`cmb`
: Add a primary CMB realisation: `'camb'` (computed with CAMB for the Planck 2018 best fit; needs
  `pip install camb`), the name of a CAMB-style text file (columns $\ell$, $D_\ell$ in µK²), or
  `(ell, C_ell)` arrays with $C_\ell$ in µK². Default `None`.

`foregrounds`
: Extra sky maps (CIB, radio sources, dust, …), already band-averaged, in `unit`, at the resolution
  of the input images: an array `(nband, ny, nx)` or a dict `{band_name: map}`. They get the same
  beam, filtering, pixelisation and gain as the SZ signal.

`calibration`
: Apply random gain errors drawn from each band's `calib_error` (default `True`).

`seed`, `cmb_seed`
: Seeds of the random numbers. With the same `cmb_seed`, different telescopes observing the same
  images see the same CMB sky, while their noise (from `seed`) stays independent.

`pad`
: Zero padding for the Fourier convolutions, as a fraction of the image size. Default: three beam
  FWHM (at most half the image), or half the image when a CMB is added.

## The result: `MockObservation`

```python
obs = tel.observe(mm, seed=1)
obs                     # MockObservation(SPT-3G, 3 bands ['f095', 'f150', 'f220'], ...)
obs['f150']             # the f150 map
obs.maps                # gains * (sz + cmb + foreground) + noise
obs.sz                  # SZ signal after bandpass, beam, filtering and pixelisation
obs.cmb, obs.foreground, obs.noise     # None if not simulated
obs.sz_true             # band-averaged SZ at the input resolution (no beam, no filtering)
obs.gains               # the applied calibration factors
obs.write_fits('mock.fits')
```

{meth}`MockObservation.write_fits <pymsz.MockObservation.write_fits>` writes the maps as the primary
HDU and, with `components=True` (default), the `SZ`, `CMB`, `FOREGROUND`, `NOISE` and `SZTRUE` cubes
as extensions. The header describes every band (`BANDn`, `FREQn`, `BPASSn`, `FWHMn`, `NOISEn`,
`GAINn`) and the seeds.

## Band integration and the model frequencies

The telescope needs the sky only at a few frequencies across every band, the integration **nodes**.
{meth}`Telescope.frequencies <pymsz.Telescope.frequencies>` returns all of them. Between the nodes,
the temperature spectrum $\Delta n / G(x)$ is interpolated with a natural cubic spline. This spectrum
is constant for the kSZ and the CMB, and nearly linear for the tSZ.

- `nnodes` nodes span the part of the band where the weight exceeds `node_cut` (default $10^{-4}$) of
  its maximum. The default of 9 nodes gives errors below $2\times10^{-4}$ of the band-averaged
  signal for all shipped bandpasses, even for 20 keV relativistic tSZ. The Planck HFI presets use 13.
- {class}`~pymsz.TT_model` and {class}`~pymsz.TK_model` compute any frequency exactly.
- For {class}`~pymsz.SZpack_model` and {class}`~pymsz.SkyMaps`, compute the images at
  `tel.frequencies()`. Otherwise they are interpolated between the frequencies you computed, which
  must bracket the bands:

```python
mm = pymsz.SZpack_model(simd, tel.frequencies(), npixel=512, AR=5, redshift=0.1)
obs = tel.observe(mm)
```

Measured bands need more nodes than top-hats. `tel.frequencies()` has 26 entries for SPT-3G and 105
for Planck.

The band-averaged signal of a sky intensity $\Delta I_\nu$ is calibrated as

$$
\Delta T = \frac{\int w(\nu)\,\Delta I_\nu\,\mathrm{d}\nu}{\int w(\nu)\,\partial B_\nu/\partial T\,\mathrm{d}\nu},
\qquad
\Delta I_{\nu_0} = \frac{\int w(\nu)\,\Delta I_\nu\,\mathrm{d}\nu}{\int w(\nu)\,(\nu_0/\nu)\,\mathrm{d}\nu},
$$

with $w(\nu)$ the bandpass (times $\nu^{-2}$ for `aperture='RJ'`) and $\nu_0$ = `freq`.

Useful band quantities:

```python
tel.tsz_response()               # non-relativistic tSZ per unit y, uK_CMB, for every band
band = tel['f150']
band.response(pymsz.sz_spectra.tsz_nonrel, 'MJy/sr')
band.cmb_to('MJy/sr')            # value of 1 K_CMB in MJy/sr in this band
band.noise_spectrum(ell)         # N_ell (unit^2 sr)
band.beam_ell(ell)               # B_ell
```

## Images from elsewhere: `SkyMaps`

{class}`~pymsz.SkyMaps` wraps any image cube so that it can be observed:

```python
sky = pymsz.SkyMaps(maps, freqs=[90., 150., 220.], unit='MJy/sr', pixel_arcsec=5.)
sky = pymsz.SkyMaps.from_fits('CL0001_SZpack.fits')    # a cube written by pymsz
obs = tel.observe(sky)
```
