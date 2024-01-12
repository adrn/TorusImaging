TorusImaging
============

TorusImaging is a Python package that implements *Orbital Torus Imaging* (OTI), a
flexible framework for foliating projections of phase space with orbits and modeling
stellar label data.
This can be used in data-driven studies of dynamics to infer the mass distribution for
the Milky Way without requiring a global model for the Milky Way or detailed modeling of
the selection function of the input survey data.
For example, see `Horta et al. 2024 <https://arxiv.org/abs/2312.07664>`_ for a set of
applications to data, and `Price-Whelan et al. 2024 <tbd>`_ for a detailed description
of the method.

|

.. image:: _static/torus.jpg
   :width: 200
   :alt: mmm donut


Tutorials
---------

.. toctree::
    :maxdepth: 1
    :glob:

    tutorials/*


API
---

.. automodapi:: torusimaging
    :no-inheritance-diagram:
    :headings: "*^"
    :no-main-docstr:
    :inherited-members:

|

.. automodapi:: torusimaging.data
    :no-inheritance-diagram:
    :headings: "*^"

|

.. automodapi:: torusimaging.plot
    :no-inheritance-diagram:
    :headings: "*^"
