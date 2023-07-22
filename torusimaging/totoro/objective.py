import inspect

# Third-party
import astropy.coordinates as coord
import astropy.units as u
import gala.dynamics as gd
import numpy as np
from scipy.optimize import minimize
from totoro.abundance_anomaly import AbundanceAnomalyMaschine

# This project
from totoro.actions import get_agama_aaf


class BaseOrbitalTorusImaging:
    pass


class ClassicalOrbitalTorusImaging(BaseOrbitalTorusImaging):
    def __init__(
        self,
        skycoord,
        elem,
        elem_err,
        potential_func,
        angle_idx=2,
        tree_K=64,
        frozen=None,
    ):
        """
        Parameters
        ----------
        data : data-like
            Must have a `get_skycoord()` method
        """

        self.skycoord = coord.SkyCoord(skycoord).icrs
        self.elem = np.asarray(elem)
        self.elem_err = np.asarray(elem_err)

        if not callable(potential_func):
            raise ValueError("TODO")
        self.potential_func = potential_func

        # TODO: this is a bad interface...
        self.angle_idx = int(angle_idx)

        if frozen is None:
            frozen = dict()
        self.frozen = frozen

        self._galcen_par_names = ["z_sun", "vx_sun", "vy_sun", "vz_sun"]
        self._galcen_par_units = [u.pc, u.km / u.s, u.km / u.s, u.km / u.s]

        sig = inspect.signature(self.potential_func)
        self._pot_par_names = list(sig.parameters.keys())

        self._par_names = self._galcen_par_names + self._pot_par_names

    def unpack_pars(self, par_list):
        par_dict = {}
        i = 0
        for key in self._par_names:
            if key in self.frozen:
                par_dict[key] = self.frozen[key]
            else:
                par_dict[key] = par_list[i]
                i += 1

        return par_dict

    def pack_pars(self, par_dict):
        parvec = []
        for i, key in enumerate(self._par_names):
            if key not in self.frozen:
                parvec.append(par_dict[key])
        return np.array(parvec)

    def get_galcen_frame(self, z_sun, vx_sun, vy_sun, vz_sun):
        v_sun = u.Quantity([vx_sun, vy_sun, vz_sun]).to(u.km / u.s)
        galcen_frame = coord.Galactocentric(
            z_sun=z_sun, galcen_v_sun=v_sun, galcen_distance=8.122 * u.kpc
        )
        return galcen_frame

    def compute_actions_angles(self, pot, galcen_frame):
        galcen = self.skycoord.transform_to(galcen_frame)

        w = gd.PhaseSpacePosition(galcen.data)
        w = gd.Orbit(w.pos, w.vel)

        if (w.energy(pot) > 0).any():
            return None, None

        aaf = get_agama_aaf(pot, w)

        return aaf["actions"], aaf["angles"]

    def get_coeffs(self, pars):
        pot_pars = {k: pars[k] for k in self._pot_par_names}
        pot = self.potential_func(**pot_pars)

        galcen_pars = {
            k: pars[k] * uu
            for k, uu in zip(self._galcen_par_names, self._galcen_par_units)
        }

        galcen_frame = self.get_galcen_frame(**galcen_pars)
        actions, angles = self.compute_actions_angles(pot, galcen_frame)

        if actions is None:
            return None, None

        maschine = AbundanceAnomalyMaschine(actions)

        # TODO: control in init which angle to use here!
        coeff, coeff_cov = maschine.get_coeffs(
            angles[:, self.angle_idx], self.elem, self.elem_err
        )

        return coeff, coeff_cov

    def __call__(self, p):
        pars = self.unpack_pars(p)
        coeff, _ = self.get_coeffs(pars)
        if coeff is None:
            return np.inf

        val = coeff[1] ** 2 + coeff[2] ** 2 + coeff[3] ** 2
        return val

    def minimize(self, x0, **kwargs):
        kwargs.setdefault("method", "nelder-mead")
        res = minimize(self, x0=x0, **kwargs)
        return res
