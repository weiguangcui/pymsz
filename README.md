# README

<span style="color:red"> **Please note that PyMSZ is still in developing mode.** </span>

### What is this repository for?

-   This package provides theoretical views of SZ y-map from both thermal and kinetic effects for hydrodynamical simulations. This package also uses the SZpack to generate observed SZ signals at any frequency from hydrodynamical simulations.
-   It requires yt installed if you want to use yt to load and analysis the simulation data.
-   current version beta

### How do I get set up?

-   Summary of set up

    This package includes SZPack v1.1.1 library. It requires gsl (with headers files) and swig installed in your system for building this library. It is very easy with `sudo apt-get install libgsl23 libgsl-dev swig python-dev` You may need manually link the libgsl.so.0 (`sudo ln -s  /usr/lib/x86_64-linux-gnu/libgsl.so.23.0.0 /usr/lib/x86_64-linux-gnu/libgsl.so.0`) to solve the `ImportError: libgsl.so.0: cannot open shared object file: No such file or directory`

    **Note for python3**: You may first need to install `python3-dev`. If you don't have the symbol link python to python3, go to the folder `cd pymsz/SZpacklib/python/` and modify the 'Makefile' by replacing 'python' with 'python3'. Then go back to the pymsz folder `cd ../../../` and run `python3 setup.py install --user`.

-   Configuration

    look at the wiki page for a simple test script. The functions inside this package are always documented.

-   Dependencies

    numpy, scipy, astropy. SZpack requires gsl and swig

-   How to run tests
-   Deployment instructions

### Contribution guidelines

-   Writing tests
-   Code review
-   Other guidelines

### Who do I talk to?

-   Please report any issue to Weiguang Cui: cuiweiguang@gmail.com.
-   Or report a bug through issues.

### acknowledgement

  Please cite the relative papers when you use the code: for theoretical kinetic SZ effects, ([please cite Baldi et al. 2018](http://adsabs.harvard.edu/abs/2018MNRAS.479.4028B); for theoretical thermal SZ effects, ([please cite Cui et al. 2018](http://adsabs.harvard.edu/doi/10.1093/mnras/sty2111); for mock SZ-signals at any frequency, ([please cite Chluba et al. 2012](http://adsabs.harvard.edu/abs/2012MNRAS.426..510C) & [2013](http://adsabs.harvard.edu/abs/2013MNRAS.430.3054C))
<!-- -   The theoretical calculation is based on -->
<!-- -   The author owe a great debt to John ZuHone, who write the model for integrating SZpack to [yt-project](http://yt-project.org/). -->

### Temporary Notes

-   The loaded simulation data has a radius of 2 x set radius, because the data has to be projected in a cylinder. Thus, I make a cut in line of sight direction later before the calculation of theoretical y-map.
-   The TK_model needs updates with the latest modifications. to be done.
