"""
Helper functions for computing actions using different action solving methods.
"""

# Third-party
import astropy.coordinates as coord
import astropy.table as at
import astropy.units as u
import gala.dynamics as gd
import gala.integrate as gi
import gala.potential as gp
import numpy as np
from gala.units import galactic

# This project
from .config import rsun as ro
from .config import vcirc as vo


def get_o2gf_aaf(potential, w0, N_max=10, dt=1 * u.Myr, n_periods=128):
    """Wrapper around the O2GF action solver in Gala that fails more gracefully

    This returns actions, angles, and frequencies for the input phase-space position
    estimated for the specified potential.
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


def get_staeckel_aaf(potential, w, delta=None):
    from galpy.actionAngle import actionAngleStaeckel

    if delta is None:
        delta = gd.get_staeckel_fudge_delta(potential, w)

    galpy_potential = potential.to_galpy_potential()

    o = w.to_galpy_orbit(ro, vo)
    aAS = actionAngleStaeckel(pot=galpy_potential, delta=delta)

    aaf = aAS.actionsFreqsAngles(o)
    aaf = {
        "actions": np.squeeze(aaf[:3]) * ro * vo,
        "freqs": np.squeeze(aaf[3:6]) * vo / ro,
        "angles": coord.Angle(np.squeeze(aaf[6:]) * u.rad),
    }
    aaf = at.QTable({k: v.T for k, v in aaf.items()})
    return aaf


def get_agama_aaf(potential, w, **kwargs):
    import agama
    from totoro.potential_helpers import gala_to_agama_potential

    if isinstance(potential, gp.PotentialBase):
        agama_pot = gala_to_agama_potential(potential)
    else:
        agama_pot = potential

    kwargs.setdefault("interp", False)
    actFinder = agama.ActionFinder(agama_pot, **kwargs)
    agama_w = np.vstack((w.xyz.to_value(u.kpc), w.v_xyz.decompose(galactic).value)).T

    if not kwargs["interp"]:
        actions, angles, freqs = actFinder(agama_w, angles=True)
        # order in each output is: R, z, phi

        aaf = {
            "actions": np.stack(
                (actions[:, 0], actions[:, 2], actions[:, 1])  # R  # phi  # z
            ).T
            * u.kpc**2
            / u.Myr,
            "angles": np.stack((angles[:, 0], angles[:, 2], angles[:, 1])).T  # R phi z
            * u.radian,
            "freqs": np.stack((freqs[:, 0], freqs[:, 2], freqs[:, 1])).T  # R phi z
            * u.rad
            / u.Myr,
        }

    else:
        actions = actFinder(agama_w)

        # order in each is: R, z, phi
        aaf = {
            "actions": np.stack(
                (actions[:, 0], actions[:, 2], actions[:, 1])  # R  # phi  # z
            ).T
            * u.kpc
            * u.kpc
            / u.Myr
        }

    return at.QTable(aaf)
