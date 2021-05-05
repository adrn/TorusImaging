"""
Helper functions for interfacing Gala and Galpy
"""

# Third-party
import astropy.units as u
import numpy as np

# This project
from .config import rsun, vcirc


def gala_to_galpy_orbit(w, ro=None, vo=None):
    from galpy.orbit import Orbit

    if ro is None:
        ro = rsun

    if vo is None:
        vo = vcirc

    # PhaseSpacePosition or Orbit:
    cyl = w.cylindrical

    R = cyl.rho.to_value(ro).T
    phi = cyl.phi.to_value(u.rad).T
    z = cyl.z.to_value(ro).T

    vR = cyl.v_rho.to_value(vo).T
    vT = (cyl.rho * cyl.pm_phi).to_value(vo, u.dimensionless_angles()).T
    vz = cyl.v_z.to_value(vo).T

    o = Orbit(np.array([R, vR, vT, z, vz, phi]).T, ro=ro, vo=vo)

    if hasattr(w, 't'):
        o.t = w.t.to_value(u.Myr)

    return o
