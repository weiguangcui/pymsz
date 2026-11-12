# pymsz.bandpasses

```{eval-rst}
.. automodule:: pymsz.bandpasses
   :no-members:
```

## Functions

```{eval-rst}
.. currentmodule:: pymsz.bandpasses

.. autosummary::
   :toctree: generated
   :nosignatures:

   available
   load
   is_available
   combine
   frequency_shift_uncertainty
```

## Data

```{eval-rst}
.. currentmodule:: pymsz.bandpasses

.. class:: Bandpass(nu, response, name, reference)

   A measured bandpass (a named tuple), as returned by :func:`load`.

   ``nu``: frequencies in GHz; ``response``: response per unit sky intensity :math:`I_\nu`, peak 1;
   ``name``: the name of the bandpass; ``reference``: where it comes from.

.. data:: DATA_DIR

   Directory of the bandpass files, ``pymsz/data/bandpasses``.
```
