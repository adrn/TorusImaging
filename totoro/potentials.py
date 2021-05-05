from astropy.constants import G
import astropy.units as u
import numpy as np
from scipy.optimize import minimize
import gala.potential as gp

from .config import fiducial_mdisk, rsun, vcirc


def get_mw_potential(mdisk):
    """
    Retrieve a MW potential model with fixed vcirc=229
    """

    def objfunc(ln_mhalo):
        mhalo = np.exp(ln_mhalo)
        tmp_mw = gp.MilkyWayPotential(disk=dict(m=mdisk),
                                      halo=dict(m=mhalo))
        test_v = tmp_mw.circular_velocity(
            [-rsun.to_value(u.kpc), 0, 0] * u.kpc).to_value(u.km/u.s)[0]
        return (vcirc.to_value(u.km/u.s) - test_v) ** 2

    minit = potentials['1.0']['halo'].parameters['m'].to_value(u.Msun)
    res = minimize(objfunc, x0=np.log(minit),
                   method='powell')

    if not res.success:
        return np.nan

    mhalo = np.exp(res.x)
    return gp.MilkyWayPotential(disk=dict(m=mdisk),
                                halo=dict(m=mhalo))


def get_equivalent_galpy(potential):
    from galpy.potential import (
        MiyamotoNagaiPotential as BovyMiyamotoNagaiPotential,
        HernquistPotential as BovyHernquistPotential,
        NFWPotential as BovyNFWPotential)

    ro = rsun
    vo = vcirc

    bovy_pot = {}

    amp = (G * potential['disk'].parameters['m']).to_value(vo**2 * ro)
    a = potential['disk'].parameters['a'].to_value(ro)
    b = potential['disk'].parameters['b'].to_value(ro)
    bovy_pot['disk'] = BovyMiyamotoNagaiPotential(amp=amp, a=a, b=b,
                                                  ro=ro, vo=vo)

    amp = (G * potential['bulge'].parameters['m']).to_value(vo**2 * ro) * 2
    c = potential['bulge'].parameters['c'].to_value(ro)
    bovy_pot['bulge'] = BovyHernquistPotential(amp=amp, a=c, ro=ro, vo=vo)

    amp = (G * potential['nucleus'].parameters['m']).to_value(vo**2 * ro) * 2
    c = potential['nucleus'].parameters['c'].to_value(ro)
    bovy_pot['nucleus'] = BovyHernquistPotential(amp=amp, a=c, ro=ro, vo=vo)

    _m = potential['halo'].parameters['m']
    amp = (G * _m).to_value(vo**2 * ro)
    rs = potential['halo'].parameters['r_s'].to_value(ro)
    bovy_pot['halo'] = BovyNFWPotential(amp=amp, a=rs, ro=ro, vo=vo)

    return list(bovy_pot.values())


# Set up Milky Way models
potentials = dict()
potentials['1.0'] = gp.MilkyWayPotential(disk=dict(m=fiducial_mdisk))
facs = np.arange(0.4, 1.8+1e-3, 0.1)  # limit 1.8 to get vcirc=229
for fac in facs:
    name = f'{fac:.1f}'
    if name in potentials:
        continue
    potentials[name] = get_mw_potential(fac * fiducial_mdisk)

# Define equivalent galpy potentials:
galpy_potentials = dict()
for k, p in potentials.items():
    galpy_potentials[k] = get_equivalent_galpy(p)
