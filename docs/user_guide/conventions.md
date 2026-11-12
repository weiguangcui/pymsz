# Units and conventions

## Frequencies and SZ spectra

All models carry the SZ signal internally as the change of the photon occupation number
$\Delta n(x)$, with the dimensionless frequency

$$x = \frac{h\nu}{k T_\mathrm{CMB}}, \qquad T_\mathrm{CMB} = 2.7255\ \mathrm{K}.$$

$\Delta n(x)$ does not depend on redshift, so all frequencies are **observed** frequencies (GHz). The
non-relativistic spectra ({mod}`pymsz.sz_spectra`) are

$$
\Delta n_\mathrm{tSZ} = y\, G(x) \left(x \coth\frac{x}{2} - 4\right), \qquad
\Delta n_\mathrm{kSZ} = \tau \beta_\mathrm{los}\, G(x), \qquad
G(x) = \frac{x e^x}{(e^x - 1)^2}.
$$

The thermal SZ effect is negative below $x = 3.83$ (217.5 GHz) and positive above.

## Units of the SZ images

`unit` of {class}`~pymsz.SZpack_model`, `TT_model.to_sz`, `TK_model.to_sz` and {class}`~pymsz.SkyMaps`:

| `unit` | Quantity |
|---|---|
| `'MJy/sr'` | $\Delta I_\nu = I_0\, x^3\, \Delta n$, with $I_0 = 2(kT_\mathrm{CMB})^3/(hc)^2 \approx 270.07$ MJy/sr |
| `'Jy/pixel'` | $\Delta I_\nu$ times the solid angle of a pixel (needs redshift > 0) |
| `'K'`, `'uK'` | linearised CMB thermodynamic temperature, $\Delta T = T_\mathrm{CMB}\,\Delta n / G(x)$ |
| `'Dn'` | $\Delta n$ |

Units of the mock observations ({meth}`Telescope.observe <pymsz.Telescope.observe>`):

| `unit` | Quantity |
|---|---|
| `'uK_CMB'`, `'K_CMB'` | CMB temperature: the band-averaged signal divided by the band-averaged $\partial B_\nu/\partial T$ |
| `'MJy/sr'` | intensity at the nominal band frequency, IRAS convention ($\nu I_\nu$ = constant) |

{func}`pymsz.sz_spectra.dn_to_unit` and {func}`~pymsz.sz_spectra.unit_to_dn` convert between them.

## Physical constants

All models use one set of constants (CODATA 2018), defined in {mod}`pymsz.sz_spectra`. They do not
depend on whether yt is installed. The *y* and $b = \tau v_\mathrm{los}/c$ maps of
{class}`~pymsz.TT_model` / {class}`~pymsz.TK_model` are therefore identical to the *y* and
$\tau\beta_\parallel$ maps of {class}`~pymsz.SZpack_model`, to $2\times10^{-14}$ per pixel.

## Simulation units

| Quantity | Unit |
|---|---|
| positions, `center`, `radius`, `zthick` | comoving kpc/h |
| masses | $10^{10} M_\odot/h$ |
| velocities | km/s (peculiar, after loading) |
| pixel size `pxs` of the models | physical kpc, at the snapshot redshift |
| `AR` | arcsec |
| telescope pixels, beams | arcmin |
| temperatures in the models | keV ($T_e$ maps), K (`load_data.temp`) |

## Lines of sight and image orientation

The observer is at $+\infty$ along the line of sight. The image plane and the line of sight for the
named axes are:

| `axis` | Image x | Image y | Line of sight |
|---|---|---|---|
| `'z'` | x | y | z |
| `'y'` | x | z | y |
| `'x'` | y | z | x |

`axis=[alpha, beta, gamma]` rotates the positions by $\alpha$ around x, $\beta$ around y and $\gamma$
around z (degrees). The new coordinates are `pos @ R`, with the rotation matrix of
{func}`pymsz.rotate_data.rotate_data`, and the line of sight is the new third axis. A 3×3 matrix is
used as `R` directly.

The image arrays are indexed `[iy, ix]`. Plot them with `origin='lower'` to get x to the right and y
upwards:

```python
import matplotlib.pyplot as plt
plt.imshow(tt.ydata, origin='lower')
```

In the FITS files, RA increases to the left (`CD1_1 < 0`), as on the sky.

## Sign of the kinetic SZ effect

$\beta_\mathrm{los} = v_\mathrm{los}/c > 0$ means gas moving **towards** the observer. This gas
produces a kSZ **increment**: $b > 0$ and $\Delta T > 0$.

## FITS files

### SZ model cubes

Written by `SZpack_model.write_fits_image`, `TT_model.write_sz_fits` and `TK_model.write_sz_fits`.

| Part | Content |
|---|---|
| primary HDU | `szdata`, shape `[nfreq, npixel, npixel]`, in the unit given by `PIXVAL`; the frequency of plane *k* is `FREQk` (GHz) |
| extensions | `TAU`, `TE` (keV), `YSZ`, `BPAR` (SZpack_model); `YSZ`, `TEY` (TT_model); `BSZ` (TK_model) |
| WCS | `CTYPEi`, `CUNITi`, `CRPIXi`, `CRVALi` (= `SP`), `CDi_j` |
| geometry | `REDSHIFT`; `AGLRES` (pixel, arcsec); `PSIZE` (pixel, physical kpc); `ORAD` (half image size, physical kpc); `RCVAL1`–`3` (centre, physical kpc) |
| model | `TCMB`; `SZMETHOD`, `SZPROJ`, `KINETIC` (SZpack_model) or `SZMODEL`, `RELATIV` (TT/TK_model); `ORIGIN`, `VERSION`, `DATE-OBS`, `COMMENT` |

### Mock observations

Written by `MockObservation.write_fits`.

| Part | Content |
|---|---|
| primary HDU | `maps`, shape `[nband, ny, nx]`, in `BUNIT` |
| extensions | `SZ`, `CMB`, `FOREGROUND`, `NOISE` (telescope pixels) and `SZTRUE` (input pixels), when simulated |
| per band *k* | `BANDk` (name), `FREQk` (GHz), `BPASSk` (bandpass), `FWHMk` (arcmin), `NOISEk` (µK-arcmin), `GAINk` (applied gain) |
| other | WCS; `TELESCOP`; `TCMB`; `REDSHIFT`; `INPIX` (input pixel, arcmin); `SEED`, `CMBSEED`; from the pipeline also `OBJECT`, `PROJ`, `SZMODEL`, `SNAPSHOT`, `PARFILE` |

### *y* or *b* maps alone

`TT_model.write_fits_image` and `TK_model.write_fits_image` write a 2D primary HDU with the same WCS
and geometry keywords. `PIXVAL` is `y parameter` or `omega` (the *b* map).

## Cosmology

Angular sizes use a flat ΛCDM cosmology with $H_0 = 100\,h$ and $\Omega_m$ from the snapshot header
(astropy `FlatLambdaCDM`). WMAP7 is used if the snapshot has no cosmology. The CMB realisations with
`cmb='camb'` use the Planck 2018 best-fit parameters.
