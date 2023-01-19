from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from astropy.stats import median_absolute_deviation as MAD
from gala.units import UnitSystem, galactic
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import binned_statistic_2d

from .model import VerticalOrbitModel

__all__ = ["VerticalOrbitModel"]


class VerticalLabelModel(VerticalOrbitModel):
    _state_names = ["Omega", "e_vals", "label_vals", "z0", "vz0"]

    def __init__(self, label_knots, e_knots, unit_sys=galactic):
        self.label_knots = jnp.array(label_knots)
        self.e_knots = {int(k): jnp.array(knots) for k, knots in e_knots.items()}

        for m, knots in self.e_knots.items():
            if len(knots) != 2:
                raise NotImplementedError(
                    "The current implementation of the model requires a purely linear "
                    "function for the e_m coefficients, which is equivalent to having "
                    f"just two knots in the (linear) spline. You passed {len(knots)} "
                    f"knots for the m={m} expansion term."
                )

        self.state = None

        # Unit system:
        self.unit_sys = UnitSystem(unit_sys)

    def get_params_init(self, z, vz):
        # TODO: what should we do here?
        pass

    def get_data_im(self, z, vz, label, bins, **binned_statistic_kwargs):
        stat = binned_statistic_2d(
            vz.decompose(self.unit_sys).value,
            z.decompose(self.unit_sys).value,
            label,
            bins=(bins["vz"], bins["z"]),
            **binned_statistic_kwargs,
        )
        xc = 0.5 * (stat.x_edge[:-1] + stat.x_edge[1:])
        yc = 0.5 * (stat.y_edge[:-1] + stat.y_edge[1:])
        xc, yc = jnp.meshgrid(xc, yc)

        # Compute label statistic error
        stat_err = binned_statistic_2d(
            vz.decompose(self.unit_sys).value,
            z.decompose(self.unit_sys).value,
            label,
            bins=(bins["vz"], bins["z"]),
            statistic=lambda x: 1.5 * MAD(x) / np.sqrt(len(x)),
        )

        return {
            "z": jnp.array(yc),
            "vz": jnp.array(xc),
            "label_stat": jnp.array(stat.statistic.T),
            "label_stat_err": jnp.array(stat_err.statistic.T),
        }

    @partial(jax.jit, static_argnames=["self"])
    def get_label(self, rz):
        self._validate_state()
        label_vals = self.state["label_vals"]
        spl = InterpolatedUnivariateSpline(self.label_knots, label_vals, k=3)
        return spl(rz)

    @partial(jax.jit, static_argnames=["self"])
    def ln_label_likelihood(self, params, z, vz, label_stat, label_stat_err):
        self.set_state(params, overwrite=True)
        self._validate_state()

        rzp, thp = self.z_vz_to_rz_theta_prime(z, vz)
        rz = self.get_rz(rzp, thp)
        model_label = self.get_label(rz)

        # log-normal
        return jnp.sum(-0.5 * (label_stat - model_label) ** 2 / label_stat_err**2)

    @partial(jax.jit, static_argnames=["self"])
    def objective(self, params, z, vz, label_stat, label_stat_err):
        return (
            -(self.ln_label_likelihood(params, z, vz, label_stat, label_stat_err))
            / label_stat.size
        )

    def copy(self):
        obj = self.__class__(label_knots=self.label_knots, e_knots=self.e_knots)
        obj.set_state(self.state)
        return obj

    # Disable some functions from the superclass
    def get_ln_dens(self, rz):
        raise NotImplementedError()

    def ln_density(self, z, vz):
        raise NotImplementedError()
