# pymsz.sz_spectra

```{eval-rst}
.. automodule:: pymsz.sz_spectra
   :no-members:
```

## Spectra

```{eval-rst}
.. currentmodule:: pymsz.sz_spectra

.. autosummary::
   :toctree: generated
   :nosignatures:

   tsz_nonrel
   ksz_nonrel
   G
   dB_dT
   I0
   freq_to_x
   x_to_freq
```

## Units and files

```{eval-rst}
.. currentmodule:: pymsz.sz_spectra

.. autosummary::
   :toctree: generated
   :nosignatures:

   dn_to_unit
   unit_to_dn
   check_unit
   interpolate_dn
   write_sz_cube
```

## Constants

CODATA 2018 values, used by all models.

| Name | Value | Meaning |
|---|---|---|
| `KB` | 1.380649e-23 J/K | Boltzmann constant |
| `H` | 6.62607015e-34 J s | Planck constant |
| `C` | 299792458 m/s | speed of light (`C_KMS` in km/s) |
| `KB_KEV` | 8.617333262e-8 keV/K | Boltzmann constant |
| `ME_KEV` | 510.99895 keV | electron rest energy |
| `SIGMA_T` | 6.6524587321e-25 cm² | Thomson cross-section |
| `MP` | 1.67262192369e-24 g | proton mass |
| `KPC` | 3.0856775809623245e21 cm | kiloparsec |
| `TCMB` | 2.7255 K | CMB temperature today (Fixsen 2009) |
| `ARCSEC`, `ARCMIN` | | arcsec and arcmin in radians |
| `UNITS` | `('MJy/sr', 'Jy/pixel', 'K', 'uK', 'Dn')` | units of the SZ images |
