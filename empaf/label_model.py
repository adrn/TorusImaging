from functools import partial
from warnings import warn

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
from astropy.stats import median_absolute_deviation as MAD
from gala.units import UnitSystem, galactic
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import binned_statistic, binned_statistic_2d

from .model import VerticalOrbitModel

__all__ = ["VerticalOrbitModel"]


class VerticalLabelModel(VerticalOrbitModel):
    _state_names = ["nu", "e_vals", "label_vals", "z0", "vz0"]

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

    def get_params_init(self, z, vz, label_stat):
        import numpy as np
        import scipy.interpolate as sci

        if hasattr(z, "unit"):
            z = z.decompose(self.unit_sys).value
        if hasattr(vz, "unit"):
            vz = vz.decompose(self.unit_sys).value

        model = self.copy()

        # First, estimate nu0 with some crazy bullshit:
        med_stat = np.nanpercentile(label_stat, 75)
        annulus_idx = np.abs(label_stat.ravel() - med_stat) < 0.02 * med_stat
        nu = np.nanpercentile(np.abs(z.ravel()[annulus_idx]), 25) / np.nanpercentile(
            np.abs(vz.ravel()[annulus_idx]), 25
        )

        model.set_state({"z0": 0.0, "vz0": 0.0, "nu": nu})
        rzp, _ = model.z_vz_to_rz_theta_prime_arr(z, vz)

        # Now estimate the label function spline values, again, with some craycray:
        bins = np.linspace(0, 1.0, 9) ** 2  # TODO: arbitrary bin max = 1
        stat = binned_statistic(
            rzp.ravel(), label_stat.ravel(), bins=bins, statistic=np.nanmedian
        )
        xc = 0.5 * (stat.bin_edges[:-1] + stat.bin_edges[1:])

        simple_spl = sci.InterpolatedUnivariateSpline(
            xc[np.isfinite(stat.statistic)],
            stat.statistic[np.isfinite(stat.statistic)],
            k=1,
        )
        model.set_state({"label_vals": simple_spl(model.label_knots)})

        # Lastly, set all e vals to 0
        e_vals = {}
        for m in self.e_knots.keys():
            e_vals[m] = jnp.zeros(1)
        model.set_state({"e_vals": e_vals})

        model._validate_state()

        return model

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
        err_floor = 0.1
        stat_err = binned_statistic_2d(
            vz.decompose(self.unit_sys).value,
            z.decompose(self.unit_sys).value,
            label,
            bins=(bins["vz"], bins["z"]),
            statistic=lambda x: np.sqrt((1.5 * MAD(x)) ** 2 + err_floor**2)
            / np.sqrt(len(x)),
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
    def label(self, z, vz):
        self._validate_state()
        rzp, thp = self.z_vz_to_rz_theta_prime(z, vz)
        rz = self.get_rz(rzp, thp)
        return self.get_label(rz)

    @partial(jax.jit, static_argnames=["self"])
    def ln_label_likelihood(self, params, z, vz, label_stat, label_stat_err):
        self.set_state(params, overwrite=True)
        self._validate_state()

        rzp, thp = self.z_vz_to_rz_theta_prime(z, vz)
        rz = self.get_rz(rzp, thp)
        model_label = self.get_label(rz)

        # log-normal
        return -0.5 * jnp.nansum((label_stat - model_label) ** 2 / label_stat_err**2)

    @partial(jax.jit, static_argnames=["self"])
    def objective(self, params, z, vz, label_stat, label_stat_err):
        return (
            -(self.ln_label_likelihood(params, z, vz, label_stat, label_stat_err))
            / label_stat.size
        )

    def optimize(
        self,
        z,
        vz,
        label_stat,
        label_stat_err,
        params0=None,
        bounds=None,
        jaxopt_kwargs=None,
    ):
        if params0 is None:
            params0 = self.get_params()

        if jaxopt_kwargs is None:
            jaxopt_kwargs = dict()
        jaxopt_kwargs.setdefault("maxiter", 16384)

        if bounds is not None:
            jaxopt_kwargs.setdefault("method", "L-BFGS-B")
            optimizer = jaxopt.ScipyBoundedMinimize(
                fun=self.objective,
                **jaxopt_kwargs,
            )
            res = optimizer.run(
                init_params=params0,
                bounds=bounds,
                z=z,
                vz=vz,
                label_stat=label_stat,
                label_stat_err=label_stat_err,
            )

        else:
            jaxopt_kwargs.setdefault("method", "BFGS")
            raise NotImplementedError("TODO")

        # warn if optimization was not successful, set state if successful
        if not res.state.success:
            warn(
                "Optimization failed! See the returned result object for more "
                "information, but the model state was not updated"
            )
        else:
            self.set_state(res.params, overwrite=True)

        return res

    def copy(self):
        obj = VerticalLabelModel(label_knots=self.label_knots, e_knots=self.e_knots)
        obj.set_state(self.state)
        return obj

    # Disable some functions from the superclass
    def get_ln_dens(self, rz):
        raise NotImplementedError()

    def ln_density(self, z, vz):
        raise NotImplementedError()
