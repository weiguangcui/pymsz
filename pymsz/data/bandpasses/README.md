# Measured bandpasses shipped with pymsz

These are the public spectral responses of Planck, ACT, SPT-SZ and SPT-3G, as published by the
collaborations. The files are unmodified and were downloaded on 2026-11-11. Read them with
`pymsz.bandpasses.load(name)`, or give the name to `pymsz.Band(bandpass=name)` or the telescope
presets. `pymsz.bandpasses.available()` lists the names.

`load()` always returns the **response per unit sky intensity** R(ν), normalised to a peak of 1.
The power on a detector is P = η ∫ R(ν) I_ν dν, which is the weight of `Band(aperture='flat')`.
The table below gives the convention of each original file and how it was checked against
published numbers.

| Directory / file | `load()` names | Source | Convention of the file | Check |
|---|---|---|---|---|
| `planck/HFI_RIMO_R3.00.fits`: `BANDPASS_F100` … `F857`; columns WAVENUMBER [cm⁻¹], TRANSMISSION | `Planck/HFI100` … `Planck/HFI857` | [Planck Legacy Archive RIMO](https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/The_RIMO), via [IRSA](https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/) | transmission per unit I_ν | Same code on the 2015 RIMO (R2.00) reproduces the [PLA unit conversions](https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/UC_CC_Tables) K_CMB→MJy/sr and K_CMB→y to < 0.1% at all six frequencies. The R3.00 bands give 0.2–0.8% different K_CMB→MJy/sr. |
| `planck/LFI_RIMO_R3.31.fits`: `BANDPASS_030`, `044`, `070`; columns WAVENUMBER [GHz], TRANSMISSION | `Planck/LFI030`, `LFI044`, `LFI070` | as above | transmission per unit I_ν | — |
| `act_dr6/passband_dr6_{pa4,pa5,pa6}_{f090,f150,f220}.dat`: columns ν [GHz], passband | `ACT-DR6/pa5_f090`, …; coadds `ACT-DR6/f090`, `f150`, `f220` | [LAMBDA ACT DR6.02 passbands](https://lambda.gsfc.nasa.gov/product/act/act_dr6.02/act_dr6.02_passbands_info.html) | per unit I_ν: these are the passbands of the DR6 likelihood | Equal to the h5 `mean_band` × ν⁻² inside `integration_bounds`, to machine precision. |
| `act_dr6/AdvACT_Passbands.h5`: `mean_band`, `lower_systematic`, `upper_systematic`, `statistical_sample` (100), `frequency_shift_uncertainty`, `integration_bounds` | `variant=` of the ACT names; `ACT-DR6/pa7_f030`, `pa7_f040` | as above | response to a Rayleigh-Jeans source (FTS not divided by ν²); `load()` multiplies by ν⁻² | needs `h5py` |
| `spt3g/spt3g_bandpass_{95,150,220}GHz_2019.txt`: ν [GHz], response (peak 1) | `SPT-3G/f095`, `f150`, `f220` | [LAMBDA SPT-3G bandpasses](https://lambda.gsfc.nasa.gov/product/spt/spt3g_bandpasses_info.html) | response to a beam-filling flat-spectrum source, including AΩ (Sobrin et al. 2022, Sect. 7.3) | Centroids and widths 93.7/26.5, 147.1/32.5, 219.9/53.5 GHz; Table 6 gives 93.8/26.4, 147.0/32.5, 219.9/53.6. |
| `spt_sz/SPTSZ_Bandpass_{90,150,220}GHz_{2009_through_2011_seasons,2008_season}.fits`: ext 1, columns NU [GHz], TRANS | `SPT-SZ/f095`, `f150`, `f220` (2009–2011); `SPT-SZ/f095_2008`, … | [LAMBDA SPT-SZ supporting products](https://lambda.gsfc.nasa.gov/product/spt/spt_sz_suport_prod_get.html) (`bandpasses.tar.gz`) | per unit I_ν | tSZ effective band centres 99.6, 155.5, 220.0 GHz, exactly the values of Bleem et al. 2022 (Sect. 2.1.4). |

The ACT coadds `ACT-DR6/f090` (PA5+PA6), `f150` (PA4+PA5+PA6) and `f220` (PA4) combine the arrays'
CMB-calibrated passbands with equal weights (`pymsz.bandpasses.combine`). The weights of the real
DR6 coadd vary with position, so these are approximate.

The h5 file also gives the bandpass uncertainties. To propagate them, use `variant='lower'`,
`'upper'` or a sample index, or shift the band with `Band(shift=...)` by
`pymsz.bandpasses.frequency_shift_uncertainty(name)`, which is 0.3–3.6 GHz (1σ).

## SHA-256 checksums

```
86a2e756021f4ef0d56bd3ff33e9aa86f85250c09f387f36f7c0db8d14aebe42  act_dr6/AdvACT_Passbands.h5
7bcd607142a46ea61fb1a6717ea0aad38eb8879c71f53c9fd4c7262ff3d37592  act_dr6/passband_dr6_pa4_f150.dat
3865c266149c099bf7bf5f324e855aeefdc0fcf56baeb04e5a3950900fda4cb8  act_dr6/passband_dr6_pa4_f220.dat
9f327d78fd7cd517b6a4143b6639316bcf5d44e889f6be9c40a0fd7516539d9b  act_dr6/passband_dr6_pa5_f090.dat
26de867b2719d3a20ebc175bf209c98b986075fbe7c9189df0a0fbe04a732cb5  act_dr6/passband_dr6_pa5_f150.dat
6d54f138a397ed671bd8e9c3a325b4ba05e3028a2ca0b9fc796db7f47772b0a9  act_dr6/passband_dr6_pa6_f090.dat
afddcacf94ee0ed4bd90bba81f294d4218f9af3c71046d3b1d05dc8b1aab9349  act_dr6/passband_dr6_pa6_f150.dat
deb55a5c6f76a8578e86c122f56c3f61998b803818b09c4f3e709897881fe222  planck/HFI_RIMO_R3.00.fits
a4794d80b378e00aef1b5e324e2759c3fa791873f2a3edcb53c93678a48dd94b  planck/LFI_RIMO_R3.31.fits
cbef9a130745089f1e378471567498377102d7bbfc136db8ffbd4d437ede3282  spt3g/spt3g_bandpass_150GHz_2019.txt
864ecf60f3a5d7a85a84e49b2dbd79a6e5f410ce25997c7ea62cf341089856ce  spt3g/spt3g_bandpass_220GHz_2019.txt
63dd5411ed354a66d3eedc82d6ce48b27fbb6136da1344eaadd4a2c1ce4ab2f4  spt3g/spt3g_bandpass_95GHz_2019.txt
668b228141ef295afc28f6ae483477ee6395a093d15eb62a2e7b401125e0cfbd  spt_sz/SPTSZ_Bandpass_150GHz_2008_season.fits
f72e358aee8f64f1cd439d729046edb2249d3ecd1407cb9bef9c294a246d90cb  spt_sz/SPTSZ_Bandpass_150GHz_2009_through_2011_seasons.fits
13734d7e5248a7f26a06397d598da31fa72ce34db5d4a5b5ab4ccf61b1d69430  spt_sz/SPTSZ_Bandpass_220GHz_2008_season.fits
16b1dec0e2f996bf1fd37ae6ae31ae06f76e879879be596f52199a625385a98d  spt_sz/SPTSZ_Bandpass_220GHz_2009_through_2011_seasons.fits
e889033f2ecfc3757bf7bc0ac474546d2357954b366b68af19e444d805197b31  spt_sz/SPTSZ_Bandpass_90GHz_2008_season.fits
f61ada040466821670e8cfa3e8fb13eb0875b03b1c4b8247427f0cdad90bc7b7  spt_sz/SPTSZ_Bandpass_90GHz_2009_through_2011_seasons.fits
```

## Please cite

When you use these bandpasses, please cite the papers that produced them:

- **Planck:** Planck Collaboration 2014, A&A 571, A9 (HFI spectral response); Zonca et al. 2009,
  JINST 4, T12010 (LFI); Planck Collaboration 2020, A&A 641, A1 (2018 release).
- **ACT:** Louis et al. 2025 (arXiv:2503.14452), the DR6 power spectra and likelihood, which use these passbands.
- **SPT-3G:** Sobrin et al. 2022, ApJS 258, 42.
- **SPT-SZ:** Bleem et al. 2022, ApJS 258, 36.

No measured bandpasses of SO, CMB-S4 or FYST/CCAT were public at download time. Their presets
use top-hats.
