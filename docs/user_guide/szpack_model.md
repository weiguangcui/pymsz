# Relativistic SZ with SZpack: `SZpack_model`

{class}`~pymsz.SZpack_model` computes the thermal plus kinetic SZ signal, including relativistic
corrections, with [SZpack v2](https://github.com/CMBSPEC/SZpack) (Chluba et al. 2012, 2013; Lee,
Chluba & Holder 2022). It needs the compiled SZpack extension (see {doc}`../installation`).

```python
mm = pymsz.SZpack_model(simd, [90., 150., 220., 353.], method='combo',
                        npixel=512, AR=5, redshift=0.1)
mm.szdata        # (4, 512, 512) SZ images, in MJy/sr by default
mm.tau, mm.Te, mm.ydata, mm.bpar
mm.write_fits_image('CL0001_SZpack.fits')
```

The image parameters (`npixel`, `AR`, `redshift`, `axis`, `zthick`, `sph_kernel`, `neighbours`,
`SD`, `SP`, `Ncpu`, `Ntasks`, `Memreduce`) are the same as for {class}`~pymsz.TT_model`, see
{doc}`theoretical_models`.

## Frequencies

`freqs` are observed frequencies in GHz. Written in terms of $x = h\nu / k T_\mathrm{CMB}$, the SZ
spectrum does not depend on redshift. SZpack is valid for $0.1 \le x \le 50$, i.e.
5.7 GHz ≤ ν ≤ 2840 GHz.

For a mock observation, compute the model at the frequencies the telescope needs for its band
integrations, `freqs=telescope.frequencies()`. See {doc}`telescopes`.

## SZpack methods

`method` selects how SZpack computes the spectrum:

| `method` | Kind | Valid for $T_e$ (keV) | Description |
|---|---|---|---|
| `'combo'` | analytic | 0–75 | `'asymptotic'` below 2 keV, `'CNSN'` above. **Recommended.** |
| `'asymptotic'` | analytic | 0–20 | Asymptotic temperature expansion; best below ~2 keV |
| `'CNSN'` | analytic | 2–75 | Basis-function interpolation (Chluba et al. 2012) |
| `'CNSN_opt'` | analytic | 0–60 | Optimised CNSN basis (Chluba et al. 2013); see `kmax`, `accuracy_level` |
| `'means'` | analytic | 0–75 | `'combo'` plus the temperature and velocity dispersion along each line of sight (moment method). Only with `projection='pixel'` |
| `'nonrel'` | analytic | any | Non-relativistic tSZ + kSZ, for comparison |
| `'3D'` | numerical | any | Numerical integration of the collision term (two angles done analytically) |
| `'5D'` | numerical | any | Full 5D numerical integration; very slow, for tests |
| `'precise'` | numerical | any | `'combo'` below 75 keV, `'3D'` above |

Gas outside the valid temperature range of the method is computed with the spectrum at the closest
valid temperature, and a warning gives the fraction of the optical depth affected.

`szpack_options` sets the accuracy. The defaults are:

| Option | Default | Used by |
|---|---|---|
| `T_order` | 10 (`'asymptotic'`), 20 (others) | `'asymptotic'`, `'CNSN'`, `'combo'` |
| `beta_order` | 2 | analytic methods |
| `relative_accuracy` | 1e-4 | `'3D'`, `'5D'`, `'precise'` |
| `kmax` | 5 | `'CNSN_opt'` |
| `accuracy_level` | 1 | `'CNSN_opt'` |

## Line-of-sight integration: `projection`

`'particle'` (default)
: The spectrum per unit optical depth is tabulated with the chosen method on a grid of electron
  temperatures. The grid includes its dependence on the line-of-sight velocity (to second order) and
  the transverse velocity:

  $$\frac{\Delta n}{\tau} = \frac{kT_e}{m_e c^2}\,G_0(T_e) + \beta_\parallel F_1(T_e)
    + \beta_\parallel^2 F_2(T_e) + \beta_\perp^2 F_3(T_e).$$

  The table is evaluated for every gas particle, and the particle spectra are SPH-smoothed onto the
  image. This is the exact single-scattering sum along the line of sight. It is fast even for the
  numerical methods, because SZpack is called only at the table nodes.
  `Te_grid` sets the temperature nodes (keV). The default is 64 nodes, log-spaced from 0.01 keV to the
  hottest gas (within the valid range of the method).

`'pixel'`
: The optical-depth-weighted temperature and velocities (and, for `'means'`, their dispersions) are
  projected first. SZpack is then called once per pixel. This is slow for `'3D'` and `'5D'`.

## Velocities

`kinetic`
: Include the gas velocities: kinetic SZ and its relativistic corrections (default `True`). With
  `False` only the thermal SZ effect is computed.

`bulk_velocity`
: {class}`~pymsz.load_data` removes the bulk velocity of the object (see
  {doc}`loading_data`). `bulk_velocity` adds a velocity back to all particles (km/s, simulation axes,
  before the rotation):

  - `None` (default): velocities relative to the object, as in {class}`~pymsz.TK_model`;
  - `True`: add back the removed bulk velocity `simd.bulkvel` (peculiar velocities);
  - `[vx, vy, vz]`: add this velocity, e.g. the velocity of the halo relative to the CMB.

The observer is at rest in the CMB frame, at $+\infty$ along the line of sight. A positive
line-of-sight velocity is motion towards the observer. Velocities above $0.1c$ are clipped in the
`'pixel'` projection (an SZpack limit).

## Results

| Attribute | Content |
|---|---|
| `freqs`, `x` | the frequencies (GHz, input order) and $x = h\nu/kT_\mathrm{CMB}$ |
| `szdata` | SZ images, `(len(freqs), npixel, npixel)`, in `unit` |
| `dndata` | the same images as $\Delta n$ |
| `tau` | Thomson optical depth |
| `Te` | optical-depth-weighted electron temperature (keV) |
| `ydata` | Compton *y* $= \tau\, kT_e / m_e c^2$, identical to `TT_model.ydata` |
| `bpar` | optical-depth-weighted $\beta_\parallel = v_\mathrm{los}/c$ (if `kinetic`) |
| `table` | the spectral table of the `'particle'` projection |

`unit` is `'MJy/sr'` (default), `'Jy/pixel'`, `'K'`, `'uK'` or `'Dn'` (see {doc}`conventions`).
`Tcmb` (default 2.7255 K) sets $x$ and the unit conversions.

`write_fits_image(fname)` writes the image cube (primary HDU) and the τ, $T_e$, *y* and $\beta_\parallel$
maps (extensions `TAU`, `TE`, `YSZ`, `BPAR`).

`sz_maps(freqs, unit)` returns images at other frequencies. With `projection='particle'` they are
interpolated with a cubic spline in *x* between the computed frequencies, which must bracket them.
With `projection='pixel'` SZpack is called again.

## Spectra and direct access to SZpack

`mm.spectrum(Te, betac, muc)` returns $\Delta n(x)/\tau$ for one temperature (keV), speed $\beta_c$
and direction cosine $\mu_c$ with the model's method and settings.
{func}`pymsz.SZpack_models.thermal_table` tabulates the thermal spectrum per unit *y* on a temperature
grid. The complete upstream Python interface of SZpack is available as

```python
from pymsz.szpack import SZpack
SZpack.Integral3D.compute_from_variables(x, The, betac, muc, 1e-4, "all", False)
```

See the SZpack notebook `SZpack/python/PySZpack.ipynb` for its functions.

## Please cite

When you use SZpack_model, please cite the SZpack papers: Chluba, Nagai, Sazonov & Nelson 2012
(MNRAS 426, 510); Chluba, Switzer, Nagai & Nelson 2013 (MNRAS 430, 3054); and, for SZpack v2, Lee,
Chluba & Holder 2022 (arXiv:2112.10666). See {doc}`../citing`.
