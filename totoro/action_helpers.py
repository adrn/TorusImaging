import astropy.units as u
import numpy as np
from scipy.optimize import minimize
import gala.integrate as gi
import gala.dynamics as gd

from .potentials import potentials, galpy_potentials
from .config import vcirc
from .actions_o2gf import get_o2gf_aaf
from .actions_staeckel import get_staeckel_aaf


def _same_actions_objfunc_staeckel(p, pos, vy, potential_name, match_actions):
    vx, vz = p
    w0 = gd.PhaseSpacePosition(pos=pos,
                               vel=[vx, vy, vz] * u.km/u.s)

    o = potentials[potential_name].integrate_orbit(
        w0, dt=1.*u.Myr, t1=0, t2=1 * u.Gyr,
        Integrator=gi.DOPRI853Integrator)
    aaf = get_staeckel_aaf(o[::2],
                           galpy_potentials[potential_name])
    actions = aaf['actions'].mean(axis=-1)

    _unit = (u.km/u.s * u.kpc)**2
    val = ((actions[0] - match_actions[0])**2 +
           (actions[2] - match_actions[2])**2).to_value(_unit)

    return val


def _same_actions_objfunc_sanders(p, pos, vy, potential_name, match_actions):
    vx, vz = p
    w0 = gd.PhaseSpacePosition(pos=pos,
                               vel=[vx, vy, vz] * u.km/u.s)
    aaf = get_o2gf_aaf(potentials[potential_name], w0, N_max=8)
    actions = aaf['actions']

    _unit = (u.km/u.s * u.kpc)**2
    val = ((actions[0] - match_actions[0])**2 +
           (actions[2] - match_actions[2])**2).to_value(_unit)

    return val


def get_w0s_with_same_actions(fiducial_w0, vy=None, staeckel=False):
    if staeckel:
        _same_actions_objfunc = _same_actions_objfunc_staeckel
    else:
        _same_actions_objfunc = _same_actions_objfunc_sanders

    if vy is None:
        # Default: set to circular velocity
        vy = vcirc

    # First, determine actions for the input orbit in the
    # fiducial potential model. These will be the target action values
    fiducial_actions = []
    for n in range(fiducial_w0.shape[0]):
        if staeckel:
            o = potentials['1.0'].integrate_orbit(
                fiducial_w0[n], dt=0.5, t1=0, t2=2*u.Gyr)  # MAGIC NUMBERS
            fiducial_actions.append(
                get_staeckel_aaf(o, galpy_potentials['1.0'])['actions'])
        else:
            fiducial_actions.append(
                get_o2gf_aaf(potentials['1.0'],
                             fiducial_w0[n], N_max=8)['actions'])

    fiducial_actions = u.Quantity(fiducial_actions).to(u.km/u.s * u.kpc)

    if staeckel:
        fiducial_actions = np.mean(fiducial_actions, axis=-1)

    w0s = {}
    for name in potentials:
        if name == '1.0':
            w0s[name] = fiducial_w0
            continue

        w0s[name] = []
        for n in range(fiducial_w0.shape[0]):
            res = minimize(_same_actions_objfunc,
                           x0=fiducial_w0.v_xyz.value[[0, 2], n],
                           args=(fiducial_w0.pos[n],
                                 vy.to_value(u.km/u.s),
                                 name,
                                 fiducial_actions[n]),
                           method='nelder-mead',
                           options=dict(maxfev=64))

            if res.fun > 1e-3:
                print(f"{name}, {n}: func val = {res.fun} -- "
                      "Failed to converge")

            w0s[name].append(gd.PhaseSpacePosition(
                pos=fiducial_w0.pos[n],
                vel=[res.x[0], vy.to_value(u.km/u.s), res.x[1]] * u.km/u.s))

        w0s[name] = gd.combine(w0s[name])

    return w0s
