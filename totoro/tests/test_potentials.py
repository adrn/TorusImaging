import astropy.units as u
import numpy as np

from ..config import rsun, vcirc
from ..potentials import potentials, galpy_potentials


def test_vcirc():
    for k, p in potentials.items():
        assert u.isclose(
            p.circular_velocity([-rsun.to_value(u.kpc), 0, 0]*u.kpc)[0],
            vcirc, rtol=1e-5)


def galpy_test_helper(gala_pot, galpy_pot):
    from galpy.potential import (evaluateDensities,
                                 evaluatePotentials,
                                 evaluatezforces)

    ntest = 16
    Rs = np.random.uniform(1, 15, size=ntest) * u.kpc
    zs = np.random.uniform(1, 15, size=ntest) * u.kpc

    xyz = np.zeros((3, Rs.size)) * u.kpc
    xyz[0] = Rs
    xyz[2] = zs

    assert np.allclose(
        gala_pot.density(xyz).to_value(u.Msun/u.pc**3),
        evaluateDensities(galpy_pot, R=Rs.to_value(rsun), z=zs.to_value(rsun)))

    assert np.allclose(
        gala_pot.energy(xyz).to_value((u.km / u.s)**2),
        evaluatePotentials(galpy_pot, R=Rs.to_value(rsun), z=zs.to_value(rsun)))

    assert np.allclose(
        gala_pot.gradient(xyz).to_value((u.km/u.s) * u.pc/u.Myr / u.pc)[2],
        -evaluatezforces(galpy_pot, R=Rs.to_value(rsun), z=zs.to_value(rsun)))


def test_against_galpy():
    for k in potentials:
        galpy_test_helper(potentials[k], galpy_potentials[k])
