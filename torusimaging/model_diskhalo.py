import inspect
from functools import partial
from typing import Callable, Optional

import jax.numpy as jnp
import numpy.typing as npt
from gala.units import UnitSystem, galactic
from scipy.stats import binned_statistic

import torusimaging.data as data
from torusimaging.model import TorusImaging1D

__all__ = ["TorusImaging1DDiskHalo"]


def label_func(r, a, b):
    return a * r + b


# def e2_func(r_e, A, ln_B, ln_C):
#     """
#     This is from a symbolic regression fit to the expected e_m function sum for a
#     Miyamoto-Nagai disk embedded in an NFW halo with a constant circular velocity.
#     For mass `m`, scale height `b` in units of 1e10 Msun and kpc:
#     - A should be close to 1
#     - B is like 1.27 * np.sqrt(b) ~= 0.64 for b = 0.25
#     - C is like 0.79 / np.sqrt(m) ~= 0.3 for a ~6–7 x 10^10 Msun disk

#     See: Symbolic-regression-tests.ipynb
#     """
#     B = jnp.exp(ln_B)
#     C = jnp.exp(ln_C)
#     return A * jnp.exp(-B / jnp.sqrt(r_e) - C)


def e2_func(r_e, ln_b, ln_m):
    b = jnp.exp(ln_b)
    m = jnp.exp(ln_m)
    val = (
        -0.8174305758200171
        * jnp.log(
            36.743526841029097 * b / m
            - r_e
            + 1.7748241828470024
            + 0.8174305758200171 * 10.61181062960402 ** (-r_e) * b**2
        )
        - (b - 0.37361794441676788) / m
    )
    return val


def em_func(r_e, a):
    return a * r_e


def regularization_func_default(model, params):
    p = 0.0
    return p


class TorusImaging1DDiskHalo(TorusImaging1D):
    def __init__(
        self,
        e_terms=[2, 4],
        regularization_func: Optional[Callable] = None,
        units: UnitSystem = galactic,
    ):
        """
        Parameters
        ----------
        regularization_func
            A function that takes in a ``TorusImaging1DSpline`` instance and a parameter
            dictionary and returns an additional regularization term to add to the
            log-likelihood.
        units
            A Gala `gala.units.UnitSystem` instance.
        """

        self._e_terms = list(e_terms)
        e_funcs = {m: e2_func if m == 2 else em_func for m in e_terms}
        super().__init__(label_func, e_funcs, regularization_func, units)

    def __reduce__(self):
        return (
            self.__class__,
            (
                self._e_terms,
                self.regularization_func,
                self.units,
            ),
        )

    @classmethod
    def auto_init(
        cls,
        binned_data: dict[str, npt.ArrayLike],
        e_terms: npt.ArrayLike,
        regularization_func: Optional[Callable | bool] = None,
        units: UnitSystem = galactic,
        **kwargs,
    ):
        """
        Parameters
        ----------
        binned_data
            A dictionary with keys "pos", "vel", "label", "label_err".
        regularization_func
            A function that takes in two arguments: a ``TorusImaging1DSpline`` instance
            and a parameter dictionary and returns an additional regularization term to
            add to the log-likelihood. If not specified, this defaults to the
            ``torusimaging.model_spline.regularization_function_default`` and additional
            arguments to that function must be specified here. If `False`, no
            regularization is applied.
        units
            A Gala `gala.units.UnitSystem` instance.
        **kwargs
            All other keyword arguments are passed to the constructor.

        Returns
        -------
        model : TorusImaging1DSpline
        bounds : dict
        init_params : dict
        """
        import astropy.units as u
        import numpy as np
        from astropy.constants import G

        bounds = {}

        # -----------------------------------------------------------------------------
        # Label function:
        #

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
        med_label_slope = np.nanmedian(np.diff(label_stat.statistic) / np.diff(xc))

        label_slope_sign = np.sign(med_label_slope)
        dlabel_dpos = np.abs(med_label_slope)

        # NOTE: the factor of 10 here is a magic number
        if label_slope_sign > 0:
            dlabel_dpos_bounds = (0, 10 * dlabel_dpos)
        else:
            dlabel_dpos_bounds = (-10 * dlabel_dpos, 0)

        x0 = label_stat.statistic[np.isfinite(label_stat.statistic)][0]
        label_5span = 5 * np.std(
            binned_data["label"][np.isfinite(binned_data["label"])]
        )
        label0_bounds = x0 + np.array([-label_5span, label_5span])

        bounds["label_params"] = {"a": dlabel_dpos_bounds, "b": label0_bounds}

        # -----------------------------------------------------------------------------
        # e functions: knots, bounds, and initial parameters
        #

        e_signs = {m: (-1.0 if (m / 2) % 2 == 0 else 1.0) for m in e_terms}

        # Use some hard-set heuristics for e function parameter bounds
        e_bounds = {}
        for m, sign in e_signs.items():
            e_bounds[m] = {}
            if m == 2:
                # if sign > 0:
                #     e_bounds[m]["A"] = (0.0, 10.0)
                # else:
                #     e_bounds[m]["A"] = (-10.0, 0.0)

                # b_lim = [0.02, 2.5]
                # m_lim = [1.0, 20.0]
                # e_bounds[m]["ln_B"] = np.sort(np.log(1.27 * np.sqrt(b_lim)))
                # e_bounds[m]["ln_C"] = np.sort(np.log(0.79 / np.sqrt(m_lim)))

                b_lim = [0.02, 2.5]
                m_lim = [1.0, 20.0]
                e_bounds[m]["ln_b"] = np.sort(np.log(b_lim))
                e_bounds[m]["ln_m"] = np.sort(np.log(m_lim))
            else:
                if sign > 0:
                    e_bounds[m]["a"] = (0, 0.5)
                else:
                    e_bounds[m]["a"] = (-0.5, 0.0)

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
                    raise ValueError(
                        "The regularization function requires additional arguments: "
                        f"{arg_names!s}, which must be passed as keyword arguments to "
                        "this class method"
                    )
                reg_kw[arg_name] = kwargs.get(arg_name, p.default)

            reg_func = partial(regularization_func, **reg_kw)

        # Initialize model instance:
        obj = cls(
            e_terms=e_terms,
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

        init_params = obj.estimate_init_params(binned_data)

        return obj, bounds, init_params

    def estimate_init_params(self, binned_data):
        import numpy as np

        Omega0 = data.estimate_Omega(binned_data).decompose(self.units).value
        p0 = {"pos0": 0.0, "vel0": 0.0, "ln_Omega0": np.log(Omega0)}

        # Parameters left to estimate: e_params, label_params

        # e_params
        p0["e_params"] = {}
        for m in self._e_terms:
            if m == 2:
                # p0["e_params"][m] = {
                #     "A": 1.0,
                #     "ln_B": np.log(0.64),
                #     "ln_C": np.log(0.3),
                # }
                p0["e_params"][m] = {
                    "ln_b": np.log(0.25),
                    "ln_m": np.log(6.8),
                }
            else:
                p0["e_params"][m] = {"a": 0.0}

        # label_params
        r_e, _ = self._get_elliptical_coords(
            binned_data["pos"].ravel(),
            binned_data["vel"].ravel(),
            p0,
        )

        # Estimate the label value near r_e = 0 and slopes for knot values:
        label = binned_data["label"].ravel()
        fin_mask = np.isfinite(label)
        r1, r2 = np.nanpercentile(r_e[fin_mask], [5, 95])
        label0 = np.nanmean(label[(r_e <= r1) & fin_mask])
        label_slope = (np.nanmedian(label[(r_e >= r2) & fin_mask]) - label0) / (r2 - r1)

        p0["label_params"] = {"a": label_slope, "b": label0}

        return p0
