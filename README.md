# README

<span style="color:red"> **Please note that PyMSZ is still in developing mode.** </span>

### What is this repository for?

-   This package provides theoretical views of SZ y map and temperatures Tsz. This package also uses the SZpack ([please cite Chluba et al. 2012](http://adsabs.harvard.edu/abs/2012MNRAS.426..510C) & [2013](http://adsabs.harvard.edu/abs/2013MNRAS.430.3054C)) to generate observed SZ signals in any frequency from hydro-dynamical simulations.
-   It requires yt installed if you want to use yt to load and analysis the data.
-   version beta

### How do I get set up?

-   This package includes SZPack v1.1.1 library. It requires gsl (with headers files) and swig installed in your system for building this library. It is very easy with `sudo apt-get install libgsl23 libgsl-dev swig` You may need manually link the libgsl.so.0 (`sudo ln -s  /usr/lib/x86_64-linux-gnu/libgsl.so.23.0.0 /usr/lib/x86_64-linux-gnu/libgsl.so.0`) to solve the `ImportError: libgsl.so.0: cannot open shared object file: No such file or directory
`
-   Configuration
-   Dependencies
-   Database configuration
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

-   Please cite the relative papers when you use the code.
-   The theoretical calculation owe a great debt to Federico Sembolini and his work ([Sembolini et al. 2013, MNRAS, 429, 323S](http://adsabs.harvard.edu/abs/2013MNRAS.429..323S))
-   The author owe a great debt to John ZoHone, who write the model for integrating SZpack to [yt-project](http://yt-project.org/). I shamelessly stole his code to make this package. Please also give your thanks to him.

### Temporary Notes

-   The loaded simulation data has a radius of 2 x set radius, because the data has to be projected in a cylinder. Thus, I make a cut in line of sight direction later before the calculation of theoretical y-map.
