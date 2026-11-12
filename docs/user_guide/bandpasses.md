# Measured bandpasses

The published spectral responses of Planck, ACT DR6, SPT-SZ and SPT-3G ship with pymsz, in
`pymsz/data/bandpasses`. {mod}`pymsz.bandpasses` reads them. The Planck, SPT-SZ, SPT-3G and ACT-DR6
telescope presets use them by default.

```python
from pymsz import bandpasses

bandpasses.available()                      # all names, e.g. 'Planck/HFI143', 'SPT-3G/f150'
bp = bandpasses.load('SPT-3G/f150')
bp.nu, bp.response                          # GHz; response per unit intensity, peak 1
bp.name, bp.reference
```

```{include} ../reference/_generated/bandpass_names.md
```

The ACT names `ACT-DR6/f090`, `f150` and `f220` are the frequency coadds of the detector arrays
(PA5+PA6, PA4+PA5+PA6 and PA4), with equal weights. `ACT-DR6/pa5_f150` etc. are single arrays.

## Using a bandpass

Give the name, a {class}`~pymsz.bandpasses.Bandpass`, `(nu, response)` arrays, or a text file with
two columns (ν in GHz, response) to a {class}`~pymsz.Band`:

```python
pymsz.Band(150., fwhm=1.2, noise=4.4, bandpass='SPT-3G/f150')
pymsz.Band(150., bandpass=bandpasses.load('ACT-DR6/pa5_f150', variant='upper'))
pymsz.Band(150., bandpass='ACT-DR6/pa5_f150', shift=1.3)       # shifted by +1.3 GHz
pymsz.Band(150., bandpass='my_band.txt', aperture='RJ')        # your own measurement
```

or change a preset:

```python
tel = pymsz.Telescope.preset('ACT-DR6', band_options={'f150': {'bandpass': 'ACT-DR6/pa5_f150'}})
tel = pymsz.Telescope.preset('ACT-DR6', measured_bandpass=False)     # top-hat approximations
```

## Conventions: `aperture`

A bandpass curve can mean two different things, and using the wrong one shifts the band by a few GHz:

`aperture='flat'` (default)
: The curve is the response per unit sky intensity $I_\nu$ (transmission-like). The detector power
  is $P \propto \int R(\nu)\, I_\nu\, \mathrm{d}\nu$. **All curves returned by
  {func}`~pymsz.bandpasses.load` follow this convention**, whatever the convention of the original
  file.

`aperture='RJ'` (alias `'single-mode'`)
: The curve is the response to a Rayleigh-Jeans ($\propto \nu^2$) source, as measured by a Fourier
  transform spectrometer, or the transmission of a single-moded detector ($A\Omega = \lambda^2$). It
  is weighted by $\nu^{-2}$.

The convention of every shipped file was checked against published numbers:

- Planck: the PLA unit conversions, to < 0.1 %;
- SPT-SZ: the tSZ effective band centres;
- SPT-3G: the band centres and widths;
- ACT: the `.dat` files equal the h5 bands times $\nu^{-2}$.

## Bandpass uncertainties

The ACT DR6 files include the bandpass uncertainties (this needs `h5py`):

```python
bandpasses.load('ACT-DR6/pa5_f150', variant='lower')    # systematic bounds: 'lower', 'upper'
bandpasses.load('ACT-DR6/pa5_f150', variant=17)         # statistical resample 0-99
bandpasses.frequency_shift_uncertainty('ACT-DR6/pa5_f150')   # 1-sigma shift of the band centre (GHz)
```

For any band, `Band(shift=...)` moves the whole bandpass.

## Does it matter?

Yes, at the percent level. With the measured instead of the top-hat bands, the observed tSZ signal
of the test cluster changes by 2.8 % (ACT f090), 4.3 % (ACT f150) and 2.8 % (SPT-3G f150).

## Combining bandpasses

{func}`~pymsz.bandpasses.combine` gives the effective bandpass of a coadd of CMB-calibrated maps:

```python
nu, resp = bandpasses.combine([bandpasses.load('ACT-DR6/pa5_f090'),
                               bandpasses.load('ACT-DR6/pa6_f090')], weights=[1., 2.])
pymsz.Band(90., bandpass=(nu, resp))
```

## Sources, checksums and citations

The text below is the `README.md` of `pymsz/data/bandpasses`. Please cite the papers listed there
when you use the bandpasses.

```{include} ../../pymsz/data/bandpasses/README.md
:heading-offset: 2
```
