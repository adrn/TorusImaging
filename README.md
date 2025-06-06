# Orbital Torus Imaging 🍩

[![Actions Status][actions-badge]][actions-link]

<img src="https://github.com/adrn/TorusImaging/blob/main/docs/_static/torus.jpg?raw=true" width=250>

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/adrn/torusimaging/workflows/CI/badge.svg
[actions-link]:             https://github.com/adrn/torusimaging/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/torusimaging
[conda-link]:               https://github.com/conda-forge/torusimaging-feedstock
[pypi-link]:                https://pypi.org/project/torusimaging/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/torusimaging
[pypi-version]:             https://img.shields.io/pypi/v/torusimaging
[rtd-badge]:                https://readthedocs.org/projects/torusimaging/badge/?version=latest
[rtd-link]:                 https://torusimaging.readthedocs.io/en/latest/?badge=latest
[zenodo-badge]:             https://zenodo.org/badge/DOI/10.5281/zenodo.10498411.svg
[zenodo-link]:              https://doi.org/10.5281/zenodo.10498411

<!-- prettier-ignore-end -->

An implementation of the method of
[Orbital Torus Imaging](https://arxiv.org/abs/2012.00015), which exploits
gradients in stellar labels (e.g., element abundances or ages) or stellar label
moments to infer the orbit structure and mass distribution of the Galaxy.

## Documentation

[![Documentation Status][rtd-badge]][rtd-link]

The documentation for `torusimaging` is hosted on
[Read the Docs](http://torusimaging.rtfd.io).

## Installation and Dependencies

[![PyPI version][pypi-version]][pypi-link]

The recommended way to install `torusimaging` is using `pip` to install the
latest development version:

    pip install git+https://github.com/adrn/torusimaging

or, to install the latest stable version:

    pip install torusimaging

## Attribution

[![Zenodo record][zenodo-badge]][zenodo-link]

If you find this package useful, please cite the latest Orbital Torus Imaging
paper and the Zenodo record above.

```
@ARTICLE{2025ApJ...979..115P,
       author = {{Price-Whelan}, Adrian M. and {Hunt}, Jason A.~S. and {Horta}, Danny and {Oeur}, Micah and {Hogg}, David W. and {Johnston}, Kathryn and {Widrow}, Lawrence},
        title = "{Data-driven Dynamics with Orbital Torus Imaging: A Flexible Model of the Vertical Phase Space of the Galaxy}",
      journal = {\apj},
     keywords = {Orbits, Celestial mechanics, Galaxy dynamics, Milky Way dynamics, Dark matter density, Astronomical methods, 1184, 211, 591, 1051, 354, 1043, Astrophysics - Astrophysics of Galaxies, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2025,
        month = feb,
       volume = {979},
       number = {2},
          eid = {115},
        pages = {115},
          doi = {10.3847/1538-4357/ad969a},
archivePrefix = {arXiv},
       eprint = {2401.07903},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025ApJ...979..115P},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## License

Copyright 2021-2025 Adrian Price-Whelan and contributors.

`torusimaging` is free software made available under the MIT License. For
details see the
[LICENSE](https://github.com/adrn/torusimaging/blob/main/LICENSE) file.
