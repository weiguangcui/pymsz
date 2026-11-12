# pymsz.telescope

```{eval-rst}
.. automodule:: pymsz.telescope
   :no-members:
```

## Classes

These are also available from the `pymsz` package.

```{eval-rst}
.. currentmodule:: pymsz

.. autosummary::
   :nosignatures:

   Telescope
   Band
   SkyMaps
   MockObservation
```

## Functions

```{eval-rst}
.. currentmodule:: pymsz.telescope

.. autosummary::
   :toctree: generated
   :nosignatures:

   cmb_power_spectrum
   gaussian_beam
   gaussian_field
```

## Data

```{eval-rst}
.. currentmodule:: pymsz.telescope

.. data:: PRESETS

   The telescope presets, ``{name: dict(description=..., reference=..., pixel=..., bands=[Band arguments])}``.
   They are listed in :doc:`/reference/presets`.

.. data:: OBS_UNITS

   The units of the mock observations: ``('uK_CMB', 'K_CMB', 'MJy/sr')``.
```
