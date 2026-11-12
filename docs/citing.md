# Citing and license

If you use pymsz in your work, please cite the papers for the parts you use:

| Part | Please cite |
|---|---|
| thermal SZ maps (`TT_model`) | Cui et al. 2018, MNRAS ([doi:10.1093/mnras/sty2111](https://doi.org/10.1093/mnras/sty2111)) |
| kinetic SZ maps (`TK_model`) | Baldi et al. 2018, MNRAS 479, 4028 ([ADS](https://ui.adsabs.harvard.edu/abs/2018MNRAS.479.4028B)) |
| SZ signal at any frequency with SZpack (`SZpack_model`, `relativistic=True`) | Chluba, Nagai, Sazonov & Nelson 2012, MNRAS 426, 510 ([ADS](https://ui.adsabs.harvard.edu/abs/2012MNRAS.426..510C)); Chluba, Switzer, Nagai & Nelson 2013, MNRAS 430, 3054 ([ADS](https://ui.adsabs.harvard.edu/abs/2013MNRAS.430.3054C)); for SZpack v2, Lee, Chluba & Holder 2022 ([arXiv:2112.10666](https://arxiv.org/abs/2112.10666)) |
| measured bandpasses | the instrument papers listed in {doc}`user_guide/bandpasses` |
| telescope presets | the sources listed in {doc}`reference/presets` |

Please also give the URL of the code, <https://github.com/weiguangcui/pymsz>, and its version
(`pymsz.version.version`; the FITS keyword `VERSION`).

## License

pymsz is free software, distributed under the
[GNU General Public License](https://github.com/weiguangcui/pymsz/blob/master/LICENSE), version 3 or
(at your option) any later version. You may use, modify and redistribute it; if you distribute
modified versions or software that includes pymsz, they must be released under the same license,
with their source code. pymsz comes without any warranty.

The SZpack sources (git submodule `SZpack/`) and the measured bandpasses in
`pymsz/data/bandpasses/` come from other authors and keep their own terms: SZpack asks that its use
is acknowledged and its papers cited, and the bandpasses should be cited as listed in
{doc}`user_guide/bandpasses`.
