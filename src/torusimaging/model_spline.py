import inspect
from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import jax.typing as jtp
import numpy.typing as npt
from gala.units import UnitSystem, galactic
from scipy.stats import binned_statistic

from torusimaging import data
from torusimaging.model import TorusImaging1D, TorusImaging1DParams
from torusimaging.model_helpers import monotonic_quadratic_spline

__all__ = ["TorusImaging1DSpline"]


def label_func_base(
    r: jtp.ArrayLike, label_vals: jtp.ArrayLike, knots: jtp.ArrayLike
) -> jax.Array:
    return monotonic_quadratic_spline(knots, label_vals, r)


def e_func_base(
    r_e: jtp.ArrayLike, vals: jtp.ArrayLike, sign: float, knots: jtp.ArrayLike
) -> jax.Array:
    return sign * monotonic_quadratic_spline(
        knots, jnp.concatenate((jnp.array([0.0]), jnp.exp(vals))), r_e
    )


def regularization_func_default(
    model: TorusImaging1D,
    params: TorusImaging1DParams,
    label_l2_sigma: float,
    label_smooth_sigma: float,
    e_l2_sigmas: dict[int, float],
    e_smooth_sigmas: dict[int, float],
    dacc_dpos_scale: float = 1e-4,
    dacc_strength: float = 1.0,
) -> jax.Array:
    p = 0.0

    if dacc_strength > 0:
        # Soft rectifier regularization meant to keep d(acc)/d(pos) < 0
        # (i.e. this tries to enforce positive density)
        for m in model.e_funcs:
            z_knots = model._e_knots[m][1:] / jnp.sqrt(jnp.exp(params["ln_Omega0"]))
            daz = model._get_dacc_dpos_vmap(z_knots, params) / dacc_dpos_scale
            p += dacc_strength * jnp.sum(jnp.log(1 + jnp.exp(daz)))

    # L2 regularization to keep the value of the functions small:
    for m, func in model.e_funcs.items():
        p += jnp.sum(
            (func(model._e_knots[m], **params["e_params"][m]) / e_l2_sigmas[m]) ** 2
        )

    p += jnp.sum(
        (
            model.label_func(model._label_knots, **params["label_params"])
            / label_l2_sigma
        )
        ** 2
    )

    # L2 regularization for smoothness:
    for m in params["e_params"]:
        diff = params["e_params"][m]["vals"][1:] - params["e_params"][m]["vals"][:-1]
        p += jnp.sum((diff / e_smooth_sigmas[m]) ** 2)

    diff = (
        params["label_params"]["label_vals"][2:]
        - params["label_params"]["label_vals"][1:-1]
    )
    p += jnp.sum((diff / label_smooth_sigma) ** 2)

    return p


class TorusImaging1DSpline(TorusImaging1D):
    """A version of the ``TorusImaging1D`` model that uses splines to model the label function and the Fourier coefficient :math:`e_m` functions.

    Parameters
    ----------
    label_knots
        The spline knot locations for the label function.
    e_knots
        A dictionary keyed by m integers with values as the spline knot locations
        for the e functions.
    e_signs
        A dictionary keyed by m integers with values as the signs of the e
        functions.
    regularization_func
        A function that takes in a ``TorusImaging1DSpline`` instance and a parameter
        dictionary and returns an additional regularization term to add to the
        log-likelihood.
    units
        A Gala :class:`gala.units.UnitSystem` instance.
    """

    def __init__(
        self,
        label_knots: jtp.ArrayLike,
        e_knots: dict[int, jtp.ArrayLike],
        e_signs: dict[int, float | int],
        regularization_func: Callable[[Any], jax.Array] | None = None,
        units: UnitSystem = galactic,
    ):
        self._label_knots = jnp.array(label_knots)
        label_func = partial(label_func_base, knots=self._label_knots)

        self._e_signs = {m: float(v) for m, v in e_signs.items()}
        self._e_knots = {m: jnp.array(knots) for m, knots in e_knots.items()}
        e_funcs = {
            m: partial(e_func_base, sign=e_signs[m], knots=knots)
            for m, knots in self._e_knots.items()
        }

        super().__init__(label_func, e_funcs, regularization_func, units)

    def __reduce__(self):
        return (
            self.__class__,
            (
                self._label_knots,
                self._e_knots,
                self._e_signs,
                self.regularization_func,
                self.units,
            ),
        )

    @classmethod
    def auto_init(
        cls,
        binned_data: dict[str, jtp.ArrayLike],
        label_knots: int | npt.ArrayLike,
        e_knots: dict[int, int | npt.ArrayLike],
        e_signs: dict[int, float | int] | None = None,
        regularization_func: Callable[[Any], jax.Array] | bool | None = None,
        units: UnitSystem = galactic,
        label_knots_spacing_power: float = 1.0,
        e_knots_spacing_power: float = 1.0,
        re_max_factor: float = 1.0,
        **kwargs: Any,
    ) -> tuple["TorusImaging1DSpline", dict[str, Any], TorusImaging1DParams]:
        """
        Parameters
        ----------
        binned_data
            A dictionary with keys "pos", "vel", "label", "label_err".
        label_knots
            Either an integer number of knots to use, or an array of knot positions.
        e_knots
            A dictionary keyed by the m order of the e function, with values either
            the number of knots to use, or an array of knot positions.
        e_signs
            A dictionary keyed by the m order of the e function, with values 1 or -1
            to represent the sign of the gradient of the e function.
        regularization_func
            A function that takes in two arguments: a ``TorusImaging1DSpline`` instance
            and a parameter dictionary and returns an additional regularization term to
            add to the log-likelihood. If not specified, this defaults to the
            :func:`torusimaging.model_spline.regularization_function_default` and
            additional arguments to that function must be specified here. The default
            regularization function tries to enforce smoothness on the splines, and that
            the density is positive. It requires the following keyword arguments:
            ``label_l2_sigma, label_smooth_sigma, e_l2_sigmas, e_smooth_sigmas``. If
            `False`, no regularization is applied.
        units
            A Gala :class:`gala.units.UnitSystem` instance.
        **kwargs
            All other keyword arguments are passed to the constructor.
        """
        import astropy.units as u
        import numpy as np
        from astropy.constants import G  # pylint: disable = no-name-in-module

        bounds = {}

        # TODO: assume binned data - but should it be particle data?
        init_Omega = data.estimate_Omega(binned_data)

        # First estimate r_e_max using the bin limits and estimated frequency:
        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            re_max = re_max_factor * np.mean(
                [
                    (binned_data["pos"].max() * np.sqrt(init_Omega))
                    .decompose(units)
                    .value,
                    (binned_data["vel"].max() / np.sqrt(init_Omega))
                    .decompose(units)
                    .value,
                ]
            )

        # -----------------------------------------------------------------------------
        # Label function: knots, bounds, and initial parameters
        #
        label_knots = np.array(label_knots)
        if label_knots.ndim == 0:
            # Integer passed in, so we need to generate the knots:
            label_knots = np.linspace(
                0, re_max**label_knots_spacing_power, label_knots
            ) ** (1 / label_knots_spacing_power)
        label_n_knots = len(label_knots)

        # Set up reasonable bounds for spline parameters - this estimates the slope of
        # the labels at a few places with respect to position. Later, we have to scale
        # by sqrt(Omega) to get the units right for the label function (defined as a
        # function of r, not position).
        # TODO: magic numbers 10 and 4
        vel_mask = (
            np.abs(binned_data["vel"])
            < np.nanpercentile(np.abs(binned_data["vel"]), 10)
        ) & (binned_data["counts"] > 4)
        label_stat = binned_statistic(
            np.abs(binned_data["pos"][vel_mask]),
            binned_data["label"][vel_mask],
            bins=np.linspace(0, binned_data["pos"].max(), 8),
        )
        xc = 0.5 * (label_stat.bin_edges[1:] + label_stat.bin_edges[:-1])
        # TODO: 10 is a magic number
        label_slope = 10 * np.nanmean(np.diff(label_stat.statistic) / np.diff(xc))

        label_slope_sign = np.sign(label_slope)
        dlabel_dpos = np.abs(label_slope)

        if label_slope_sign > 0:
            dlabel_dpos_bounds = (
                np.full(label_n_knots - 1, 0),
                np.full(label_n_knots - 1, dlabel_dpos),
            )
        else:
            dlabel_dpos_bounds = (
                np.full(label_n_knots - 1, -dlabel_dpos),
                np.full(label_n_knots - 1, 0),
            )
        x0 = label_stat.statistic[np.isfinite(label_stat.statistic)][0]
        label_5span = 5 * np.std(
            binned_data["label"][np.isfinite(binned_data["label"])]
        )
        label0_bounds = x0 + np.array([-label_5span, label_5span])

        bounds["label_params"] = {
            "label_vals": (
                np.concatenate(([label0_bounds[0]], dlabel_dpos_bounds[0])),
                np.concatenate(([label0_bounds[1]], dlabel_dpos_bounds[1])),
            )
        }

        # -----------------------------------------------------------------------------
        # e functions: knots, bounds, and initial parameters
        #

        e_knots = {m: np.array(knots) for m, knots in e_knots.items()}
        for m, knots in e_knots.items():
            if knots.ndim == 0:
                # Integer passed in, so we need to generate the knots:
                e_knots[m] = np.linspace(0, re_max**e_knots_spacing_power, knots) ** (
                    1 / e_knots_spacing_power
                )
        e_n_knots = {m: len(knots) for m, knots in e_knots.items()}

        if e_signs is None:
            e_signs = {}
        default_e_signs = {m: (-1.0 if (m / 2) % 2 == 0 else 1.0) for m in e_knots}
        e_signs = {m: e_signs.get(m, default_e_signs[m]) for m in e_knots}

        # Use some hard-set heuristics for e function parameter bounds
        e_bounds = {}
        for m, n in e_n_knots.items():
            # TODO: hard-set magic numbers - both are truly arbitrary
            # Bounds for e functions, in log-space
            # TODO: change name to log_vals?
            e_bounds.setdefault(
                m, {"vals": (jnp.full(n - 1, -16.0), jnp.full(n - 1, 10.0))}
            )
        bounds["e_params"] = e_bounds

        # -----------------------------------------------------------------------------
        # Regularization function
        #
        if regularization_func is False:
            reg_func = None

        else:
            if regularization_func is None:
                regularization_func = regularization_func_default

            # Regularization function could take other arguments that have to be
            # specified as kwargs to this classmethod, as is the case for the default
            # function:
            sig = inspect.signature(regularization_func)
            arg_names = list(sig.parameters.keys())[2:]

            reg_kw = {}
            for arg_name in arg_names:
                p = sig.parameters[arg_name]
                if arg_name not in kwargs and p.default is inspect._empty:
                    msg = (
                        "The regularization function requires additional arguments: "
                        f"{arg_names!s}, which must be passed as keyword arguments to "
                        "this class method"
                    )
                    raise ValueError(msg)
                reg_kw[arg_name] = kwargs.get(arg_name, p.default)

            reg_func = partial(regularization_func, **reg_kw)

        # Initialize model instance:
        obj = cls(
            label_knots=label_knots,
            e_knots=e_knots,
            e_signs=e_signs,
            regularization_func=reg_func,
            units=units,
        )

        # Other parameter bounds:
        # Wide, physical bounds for the log-midplane density
        dens0_bounds = [0.001, 100] * u.Msun / u.pc**3
        bounds["ln_Omega0"] = 0.5 * np.log(
            (4 * np.pi * G * dens0_bounds).decompose(units).value
        )
        bounds["pos0"] = ([-1.0, 1.0] * u.kpc).decompose(units).value
        bounds["vel0"] = ([-100.0, 100.0] * u.km / u.s).decompose(units).value

        init_params = obj.estimate_init_params(binned_data, bounds)

        # Need to scale the bounds of the label function derivatives by sqrt(Omega)
        sqrtOmega = np.sqrt(np.exp(init_params["ln_Omega0"]))
        bounds["label_params"]["label_vals"][0][1:] /= sqrtOmega
        bounds["label_params"]["label_vals"][1][1:] /= sqrtOmega

        return obj, bounds, init_params

    def estimate_init_params(
        self, binned_data: dict[str, npt.ArrayLike], bounds: dict[str, Any]
    ) -> TorusImaging1DParams:
        import numpy as np

        Omega0 = data.estimate_Omega(binned_data).decompose(self.units).value
        p0 = {"pos0": 0.0, "vel0": 0.0, "ln_Omega0": np.log(Omega0)}

        # Parameters left to estimate: e_params, label_params

        # e_params
        p0["e_params"] = {
            m: {"vals": bounds["e_params"][m]["vals"][0]}  # lower bound
            for m in self._e_knots
        }

        # label_params
        r_e, _ = self._get_elliptical_coords(
            binned_data["pos"].ravel(),
            binned_data["vel"].ravel(),
            pos0=p0["pos0"],
            vel0=p0["vel0"],
            ln_Omega0=p0["ln_Omega0"],
        )

        # Estimate the label value near r_e = 0 and slopes for knot values:
        label = binned_data["label"].ravel()
        fin_mask = np.isfinite(label)
        r1, r2 = np.nanpercentile(r_e[fin_mask], [5, 95])
        label0 = np.nanmean(label[(r_e <= r1) & fin_mask])
        label_slope = (np.nanmedian(label[(r_e >= r2) & fin_mask]) - label0) / (r2 - r1)

        p0["label_params"] = {
            "label_vals": np.concatenate(
                (
                    [label0],
                    np.full(len(self._label_knots) - 1, label_slope),
                )
            )
        }

        return p0
