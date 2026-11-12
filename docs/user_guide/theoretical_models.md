# Thermal and kinetic SZ maps: `TT_model` and `TK_model`

{class}`~pymsz.TT_model` and {class}`~pymsz.TK_model` project the two quantities that set the SZ
effect in the non-relativistic limit:

| Model | Map | Definition |
|---|---|---|
| `TT_model` | `ydata`: Compton-*y* | $y = \dfrac{\sigma_T}{m_e c^2}\displaystyle\int n_e\, k T_e\, \mathrm{d}l$ |
| `TT_model` | `Tydata`: *y*-weighted temperature (keV) | $T_y = \displaystyle\int n_e T_e\, kT_e\,\mathrm{d}l \Big/ \int n_e\, kT_e\,\mathrm{d}l$ |
| `TK_model` | `bdata`: kinetic SZ *b* | $b = \sigma_T \displaystyle\int n_e \dfrac{v_\mathrm{los}}{c}\,\mathrm{d}l = \tau\,\beta_\mathrm{los}$ |

Every gas particle contributes $\sigma_T N_e k T_e / m_e c^2$ (or $\sigma_T N_e v_\mathrm{los}/c$),
where $N_e$ is its number of electrons. These contributions are SPH-smoothed onto the image and
divided by the pixel area.

```python
tt = pymsz.TT_model(simd, npixel=512, AR=5, redshift=0.1, axis='z')
tk = pymsz.TK_model(simd, npixel=512, AR=5, redshift=0.1, axis='z')
```

Both models take the same image parameters, and use the same pixel grid as
{class}`~pymsz.SZpack_model`, so their maps can be added and compared pixel by pixel.

## Image geometry

`redshift`
: The redshift at which the object is observed. It sets the angular size of the pixels. Default:
  the redshift of the snapshot. Place the object at $z > 0$. At $z = 0$ the angular units are not
  defined and `AR` is set to a nominal 1 arcsec.

`AR`
: The pixel size in arcsec. With `redshift`, it sets the physical pixel size.

`npixel`
: Pixels per side (the images are square). `'auto'`: enough pixels to cover the loaded region
  (`2 radius`) at the pixel size `AR`.

The three combinations are:

| `AR` | `npixel` | Pixel size | Image |
|---|---|---|---|
| given | number | from `AR` and `redshift` | `npixel` pixels around the centre |
| given | `'auto'` | from `AR` and `redshift` | covers the whole region |
| `None` | number | region size / `npixel` | covers the region; `AR` is computed |

`npixel='auto'` with `AR=None` (or at $z = 0$) is not possible; `npixel` is then reset to 500.
Distances use a flat ΛCDM cosmology with the $h$ and $\Omega_m$ of the snapshot (WMAP7 if the
snapshot has none). The pixel size in physical kpc is stored in `pxs`, the angular size in `ar`.

`axis`
: The line of sight: `'x'`, `'y'`, `'z'`, three rotation angles `[alpha, beta, gamma]` in degrees
  (around the x, y and z axes), or a 3×3 rotation matrix. See {doc}`conventions`.

`zthick`
: Use only the gas within ±`zthick` (comoving kpc/h) of the centre along the line of sight. Default:
  all loaded gas.

`SP`
: The sky position `[RA, Dec]` (degrees) of the image centre, written to the FITS WCS. Default: the
  position of Coma, `[194.95, 27.98]`.

## SPH smoothing

The particles are spread over the pixels with an SPH kernel. The kernel weights of each particle are
normalised to 1, so the total signal is conserved.

`sph_kernel`
: `'cubic'` (default), `'quartic'`, `'quintic'`, `'wendland2'`, `'wendland4'`, `'wendland6'`.

`neighbours`
: Default `None`: use the smoothing lengths of the snapshot. If set, every particle is smoothed over
  this number of neighbours instead.

`SD`
: Smoothing in 2 dimensions (default: on the image plane) or 3 (on a 3D grid, then summed along the
  line of sight). `SD=3` is much slower and needs more memory.

`Ncpu`, `Ntasks`, `Memreduce`
: The smoothing runs in parallel on `Ncpu` processes (default: all CPUs), split into `Ntasks` tasks
  (default `Ncpu`). If you see `IO Error: bad message length`, increase `Ntasks`. `Memreduce=True`
  passes the k-d tree to the workers instead of the neighbour lists, which needs less memory.

## From maps to SZ images: `to_sz`

`to_sz(freqs, unit)` turns the maps into SZ images at the observed frequencies `freqs` (GHz). The
result has shape `(len(freqs), npixel, npixel)`. It is returned and kept in `szdata`, with `freqs`
and `unit`:

```python
tt.to_sz([90., 150., 220., 353.], unit='uK')                      # non-relativistic tSZ
tt.to_sz([90., 150., 220., 353.], unit='uK', relativistic=True)   # with relativistic corrections
tk.to_sz([90., 150., 220., 353.], unit='uK')                      # kSZ
total = tt.szdata + tk.szdata
```

With $x = h\nu / k T_\mathrm{CMB}$ and $G(x) = x e^x / (e^x - 1)^2$, the change of the photon
occupation number is

| | Spectrum |
|---|---|
| tSZ, non-relativistic | $\Delta n = y\, G(x)\,\left(x \coth(x/2) - 4\right)$ |
| tSZ, `relativistic=True` | $\Delta n = y\, g(x, T_y)$, with $g$ from SZpack at the *y*-weighted temperature of each pixel |
| kSZ | $\Delta n = b\, G(x)$, i.e. $\Delta T = T_\mathrm{CMB}\, b$ |

The available units are listed in {doc}`conventions`: `'MJy/sr'` (default), `'Jy/pixel'`, `'K'`,
`'uK'` and `'Dn'`.

### Relativistic tSZ from the *y* map

With `relativistic=True`, the SZpack spectrum per unit *y* is tabulated on a grid of temperatures and
evaluated in every pixel at $T_y$ (`Tydata`). This is exact to first order in the temperature
spread along the line of sight. For a 14 keV cluster it agrees with the exact particle-by-particle sum
of {class}`~pymsz.SZpack_model` to about 0.5 % of the peak signal. Options:

- `Te`: use this temperature (keV) instead of `Tydata`: a number (isothermal) or a map.
- `method`, `szpack_options`: the SZpack method and its accuracy settings (see {doc}`szpack_model`).

This needs the SZpack extension and `TT_model(..., yweighted_Te=True)` (the default). For
relativistic kSZ corrections, or the exact mix of temperatures along the line of sight, use
{class}`~pymsz.SZpack_model`.

## Saving the results

`write_sz_fits(fname)`
: Writes the SZ images of `to_sz` as a FITS cube (primary HDU, one plane per frequency), plus the
  *y* (and $T_y$) or *b* map as image extensions. This is the same layout as
  {meth}`SZpack_model.write_fits_image <pymsz.SZpack_model.write_fits_image>`, and the files can be
  read back with {meth}`SkyMaps.from_fits <pymsz.SkyMaps.from_fits>`.

`write_fits_image(fname)`
: Writes only the 2D *y* (or *b*) map, with a WCS header.

```python
tt.write_sz_fits('CL0001_tSZ.fits', comments='my first cluster', overwrite=True)
tk.write_fits_image('CL0001_bmap.fits', overwrite=True)
```

The header keywords are described in {doc}`conventions`.

## Other maps from `TT_model`

`TT_model(simd, label='ne', ...)` projects the number of electrons instead of *y*. Then `ydata` is
the electron column density in electrons per physical kpc². The ratio of the *y* and electron
column maps is a column-averaged temperature:

```python
tt = pymsz.TT_model(simd, label='tSZ', npixel='auto', AR=5, redshift=0.1)
ne = pymsz.TT_model(simd, label='ne', npixel='auto', AR=5, redshift=0.1)
```

`to_sz` and `write_sz_fits` need `label='tSZ'`.

## Using the models with a telescope

Both models can be observed directly. {meth}`Telescope.observe <pymsz.Telescope.observe>` calls their
`sz_maps(freqs, unit)` method with the frequencies it needs, using the spectrum chosen in `to_sz`
(non-relativistic if `to_sz` was not called):

```python
tel = pymsz.Telescope.preset('ACT-DR6')
obs = tel.observe([tt, tk], seed=1)
```
