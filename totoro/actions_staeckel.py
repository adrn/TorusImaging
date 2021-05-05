"""
Helper functions for computing actions using the axisymmetric Stäckel Fudge
method introduced in Binney 20XX, implemented in Galpy.
"""

# Third-party
import astropy.coordinates as coord
import astropy.table as at
import astropy.units as u
import numpy as np
import gala.dynamics as gd

# This project
from .config import rsun as ro, vcirc as vo
from .galpy_helpers import gala_to_galpy_orbit


def get_staeckel_aaf(galpy_potential, w, delta=None, gala_potential=None):
    from galpy.actionAngle import actionAngleStaeckel

    if delta is None:
        if gala_potential is None:
            raise ValueError("If deltas not specified, you must specify "
                             "gala_potential.")
        delta = gd.get_staeckel_fudge_delta(gala_potential, w)

    o = gala_to_galpy_orbit(w)
    aAS = actionAngleStaeckel(pot=galpy_potential, delta=delta)

    aaf = aAS.actionsFreqsAngles(o)
    aaf = {'actions': np.squeeze(aaf[:3]) * ro * vo,
           'freqs': np.squeeze(aaf[3:6]) * vo / ro,
           'angles': coord.Angle(np.squeeze(aaf[6:]) * u.rad)}
    aaf = at.Table({k: v.T for k, v in aaf.items()})
    return aaf
