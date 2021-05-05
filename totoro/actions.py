"""
Helper functions for computing actions using the O2GF method introduced in
Sanders & Binney 2014.
"""

# Third-party
import astropy.coordinates as coord
import astropy.table as at
import astropy.units as u
import numpy as np
import gala.dynamics as gd
import gala.integrate as gi

# This project
from .config import rsun as ro, vcirc as vo


def get_o2gf_aaf(potential, w0, N_max=10, dt=1 * u.Myr, n_periods=128):
    """Wrapper around the O2GF action solver in Gala that fails more gracefully

    This returns actions, angles, and frequencies for the input phase-space
    position estimated for the specified potential.
    """
    aaf_units = {
        "actions": u.km / u.s * u.kpc,
        "angles": u.degree,
        "freqs": 1 / u.Gyr,
    }

    # First integrate a little bit of the orbit with Leapfrog to estimate
    # the orbital period
    test_orbit = potential.integrate_orbit(w0, dt=2 * u.Myr, t1=0, t2=1 * u.Gyr)
    P_guess = test_orbit.estimate_period()

    if np.isnan(P_guess):
        return {k: np.full(3, np.nan) * aaf_units[k] for k in aaf_units}

    # Integrate the orbit with a high-order integrator for many periods
    orbit = potential.integrate_orbit(
        w0,
        dt=dt,
        t1=0,
        t2=n_periods * P_guess,
        Integrator=gi.DOPRI853Integrator,
    )

    # Use the Sanders & Binney action solver:
    try:
        aaf = gd.find_actions(orbit, N_max=N_max)
    except Exception:
        aaf = {k: np.full(3, np.nan) * aaf_units[k] for k in aaf_units}

    aaf = {k: aaf[k].to(aaf_units[k]) for k in aaf_units.keys()}
    return aaf


def get_staeckel_aaf(galpy_potential, w, delta=None, gala_potential=None):
    from galpy.actionAngle import actionAngleStaeckel

    if delta is None:
        if gala_potential is None:
            raise ValueError(
                "If deltas not specified, you must specify " "gala_potential."
            )
        delta = gd.get_staeckel_fudge_delta(gala_potential, w)

    o = w.to_galpy_orbit(ro, vo)
    aAS = actionAngleStaeckel(pot=galpy_potential, delta=delta)

    aaf = aAS.actionsFreqsAngles(o)
    aaf = {
        "actions": np.squeeze(aaf[:3]) * ro * vo,
        "freqs": np.squeeze(aaf[3:6]) * vo / ro,
        "angles": coord.Angle(np.squeeze(aaf[6:]) * u.rad),
    }
    aaf = at.Table({k: v.T for k, v in aaf.items()})
    return aaf
