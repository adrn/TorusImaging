from collections.abc import Callable
from functools import partial
from typing import Any, Literal, NewType
from warnings import warn

import astropy.table as at
import astropy.units as u
import jax
import jax.numpy as jnp
import jax.typing as jtp
import jaxopt
import numpy.typing as npt
from gala.units import UnitSystem, galactic
from jax.scipy.special import gammaln
from jaxopt import Bisection
from jaxopt.base import OptStep
from typing_extensions import TypedDict

from torusimaging.jax_helpers import simpson

__all__ = ["TorusImaging1D"]

length_pt = u.get_physical_type("length")
velocity_pt = u.get_physical_type("speed")

EParams = NewType("EParams", dict[int, dict[str, jax.Array | float]])


class TorusImaging1DParams(TypedDict, total=False):
    pos0: jax.Array | float
    vel0: jax.Array | float
    ln_Omega0: jax.Array | float
    e_params: EParams
    label_params: dict[str, jax.Array | float]


class TorusImaging1D:
    r"""A flexible and customizable interface for fitting and MCMC sampling an Orbital
    Torus Imaging model. This implementation assumes that you are working in a 1 degree
    of freedom phase space with position coordinate ``q`` and velocity coordinate ``p``.

    Notation:

    * :math:`\Omega_0` or ``Omega0``: A scale frequency used to compute the
      elliptical radius ``r_e``. This is the asymptotic orbital frequency at zero
      action.
    * :math:`r_e` or ``r_e``: The elliptical radius
      :math:`r_e = \sqrt{q^2\, \Omega_0 + p^2 \, \Omega_0^{-1}}`.
    * :math:`\theta_e` or ``theta_e``: The elliptical angle defined as
      :math:`\tan{\theta_e} = \frac{q}{p}\,\Omega_0`.
    * :math:`r` or ``r``: The distorted elliptical radius
      :math:`r = r_e \, f(r_e, \theta_e)`, which is close  to :math:`\sqrt{J}` (the
      action) and so we sometimes call it the "proxy action" below. :math:`f(\cdot)`
      is the distortion function defined below.
    * :math:`f(r_e, \theta_e)`: The distortion function is a Fourier expansion,
      defined as: :math:`f(r_e, \theta_e) = 1 + \sum_m e_m(r_e)\,\cos(m\,\theta_e)`
    * :math:`J` or ``J``: The action.
    * :math:`\theta` or ``theta``: The conjugate angle.

    Parameters
    ----------
    label_func
        A function that specifies the dependence of the label function on the distorted
        radius :math:`r`.
    e_funcs
        A dictionary that provides functions that specify the dependence of the Fourier
        distortion coefficients :math:`e_m(r_e)`. Keys should be the (integer) "m" order
        of the distortion term (for the distortion function), and values should be
        Python callable objects that can be passed to `jax.jit()`. The first argument of
        each of these functions should be the elliptical radius :math:`r_e` or ``re``.
    regularization_func
        An optional function that computes a regularization term to add to the
        log-likelihood function when optimizing.
    units
        The unit system to work in. Default is to use the "galactic" unit system from
        Gala: (kpc, Myr, Msun, radian).

    """

    def __init__(
        self,
        label_func: Callable[[float], float],
        e_funcs: dict[int, Callable[[float], float]],
        regularization_func: Callable[[Any], float] | None = None,
        units: UnitSystem = galactic,
    ):
        self.label_func = jax.jit(label_func)
        self.e_funcs = {int(m): jax.jit(e_func) for m, e_func in e_funcs.items()}

        # Unit system:
        self.units = UnitSystem(units)

        if regularization_func is None:
            regularization_func = lambda *_, **__: 0.0  # noqa: E731
        self.regularization_func = regularization_func

    # ---------------------------------------------------------------------------------
    # Internal functions used within likelihood functions:
    #
    @partial(jax.jit, static_argnames=["self"])
    def _get_elliptical_coords(
        self,
        pos: float | jax.Array,
        vel: float | jax.Array,
        pos0: float,
        vel0: float,
        ln_Omega0: float,
    ) -> tuple[float | jax.Array, float | jax.Array]:
        r"""Compute the raw elliptical radius :math:`r_e` (``r_e``) and angle
        :math:`\theta_e'` (``theta_e``)
        """
        x = (vel - vel0) / jnp.sqrt(jnp.exp(ln_Omega0))
        y = (pos - pos0) * jnp.sqrt(jnp.exp(ln_Omega0))

        r_e = jnp.sqrt(x**2 + y**2)
        t_e = jnp.arctan2(y, x)

        return r_e, t_e

    @partial(jax.jit, static_argnames=["self"])
    def _get_es(self, r_e: float, e_params: EParams) -> dict[int, float]:
        """Compute the Fourier m-order distortion coefficients"""
        es = {}
        for m, pars in e_params.items():
            es[m] = self.e_funcs[m](r_e, **pars)
        return es

    @partial(jax.jit, static_argnames=["self"])
    def _get_r(
        self,
        r_e: float,
        theta_e: float,
        e_params: EParams,
    ) -> jax.Array:
        """Compute the distorted radius :math:`r`"""
        es = self._get_es(r_e, e_params)
        return r_e * (
            1
            + jnp.sum(
                jnp.array([e * jnp.cos(m * theta_e) for m, e in es.items()]), axis=0
            )
        )

    @partial(jax.jit, static_argnames=["self"])
    def _get_theta(
        self,
        r_e: float,
        theta_e: float,
        e_params: EParams,
    ) -> jax.Array:
        """Compute the phase angle"""
        es = self._get_es(r_e, e_params)
        # TODO: why is the π/2 needed below??
        return theta_e - jnp.sum(
            jnp.array(
                [m / (jnp.pi / 2) * e * jnp.sin(m * theta_e) for m, e in es.items()]
            ),
            axis=0,
        )

    @partial(jax.jit, static_argnames=["self"])
    def _get_r_e(
        self,
        r: float,
        theta_e: float,
        e_params: EParams,
        Bisection_kwargs: dict[str, Any],
    ) -> float:
        """Compute the elliptical radius :math:`r_e` by inverting the distortion
        transformation from :math:`r`
        """
        Bisection_kwargs = dict(Bisection_kwargs)
        Bisection_kwargs.setdefault("lower", 0.0)
        Bisection_kwargs.setdefault("upper", 1.0)
        Bisection_kwargs.setdefault("maxiter", 30)
        Bisection_kwargs.setdefault("tol", 1e-4)

        bisec = Bisection(
            lambda x, rrz, tt_prime, ee_params: self._get_r(x, tt_prime, ee_params)
            - rrz,
            jit=True,
            unroll=True,
            check_bracket=False,
            **Bisection_kwargs,
        )
        return float(bisec.run(r, rrz=r, tt_prime=theta_e, ee_params=e_params).params)

    @partial(jax.jit, static_argnames=["self"])
    def _get_pos(
        self,
        r: float,
        theta_e: float,
        params: TorusImaging1DParams,
        Bisection_kwargs: dict[str, Any],
    ) -> jax.Array:
        """Compute the position given the distorted radius and elliptical angle"""
        r_e = self._get_r_e(r, theta_e, params["e_params"], Bisection_kwargs)
        return r_e * jnp.sin(theta_e) / jnp.sqrt(jnp.exp(params["ln_Omega0"]))

    @partial(jax.jit, static_argnames=["self"])
    def _get_vel(
        self,
        r: float,
        theta_e: float,
        params: TorusImaging1DParams,
        Bisection_kwargs: dict[str, Any],
    ) -> jax.Array:
        """Compute the velocity given the distorted radius and elliptical angle"""
        rzp = self._get_r_e(r, theta_e, params["e_params"], Bisection_kwargs)
        return rzp * jnp.cos(theta_e) * jnp.sqrt(jnp.exp(params["ln_Omega0"]))

    @partial(jax.jit, static_argnames=["self"])
    def _get_label(
        self,
        pos: float,
        vel: float,
        params: TorusImaging1DParams,
    ) -> jax.Array:
        r_e, th_e = self._get_elliptical_coords(
            pos,
            vel,
            pos0=params["pos0"],
            vel0=params["vel0"],
            ln_Omega0=params["ln_Omega0"],
        )
        r = self._get_r(r_e, th_e, params["e_params"])
        return self.label_func(r, **params["label_params"])

    @partial(jax.jit, static_argnames=["self", "N_grid", "Bisection_kwargs"])
    def _get_T_J_theta(
        self,
        pos: float,
        vel: float,
        params: TorusImaging1DParams,
        N_grid: int,
        Bisection_kwargs: dict[str, Any],
    ) -> tuple[float, float, float]:
        re_, the_ = self._get_elliptical_coords(
            pos,
            vel,
            pos0=params["pos0"],
            vel0=params["vel0"],
            ln_Omega0=params["ln_Omega0"],
        )
        r = self._get_r(re_, the_, params["e_params"])

        dpos_dthe_func = jax.vmap(
            jax.grad(self._get_pos, argnums=1), in_axes=[None, 0, None, None]
        )

        get_vel = jax.vmap(self._get_vel, in_axes=[None, 0, None, None])

        # Grid of theta_prime to do the integral over:
        the_grid = jnp.linspace(0, jnp.pi / 2, N_grid)
        v_th = get_vel(r, the_grid, params, Bisection_kwargs)
        dz_dthp = dpos_dthe_func(r, the_grid, params, Bisection_kwargs)

        Tz = 4 * simpson(dz_dthp / v_th, the_grid)
        Jz = 4 / (2 * jnp.pi) * simpson(dz_dthp * v_th, the_grid)

        thp_partial = jnp.linspace(0, the_, N_grid)
        v_th_partial = get_vel(r, thp_partial, params, Bisection_kwargs)
        dpos_dthe_partial = dpos_dthe_func(r, thp_partial, params, Bisection_kwargs)
        dt = simpson(dpos_dthe_partial / v_th_partial, thp_partial)
        thz = 2 * jnp.pi * dt / Tz

        return Tz, Jz, thz

    _get_T_J_theta = jax.vmap(_get_T_J_theta, in_axes=[None, 0, 0, None, None, None])

    @partial(jax.jit, static_argnames=["self"])
    def _get_de_dr_es(self, r_e: float, e_params: EParams) -> dict[int, float]:
        """Compute the derivatives of the Fourier m-order distortion coefficient
        functions
        """
        d_es = {}
        for m, pars in e_params.items():
            d_es[m] = jax.grad(self.e_funcs[m], argnums=0)(r_e, **pars)
        return d_es

    # ---------------------------------------------------------------------------------
    # Public API
    #
    @u.quantity_input
    def compute_elliptical(
        self,
        pos: u.Quantity[length_pt],
        vel: u.Quantity[velocity_pt],
        params: TorusImaging1DParams,
    ) -> tuple[u.Quantity, u.Quantity]:
        """Compute the elliptical radius :math:`r_e` (``r_e``) and angle
        :math:`\theta_e'` (``theta_e``)

        Parameters
        ----------
        pos
            The position values.
        vel
            The velocity values.
        params
            A dictionary of model parameters.
        """

        x = pos.decompose(self.units).value
        v = vel.decompose(self.units).value
        re, te = self._get_elliptical_coords(
            x,
            v,
            pos0=params["pos0"],
            vel0=params["vel0"],
            ln_Omega0=params["ln_Omega0"],
        )
        return (
            re
            * self.units["length"]
            / (self.units["angle"] ** 0.5 / self.units["time"] ** 0.5),
            te * self.units["angle"],
        )

    @u.quantity_input
    def compute_action_angle(
        self,
        pos: u.Quantity[length_pt],
        vel: u.Quantity[velocity_pt],
        params: TorusImaging1DParams,
        N_grid: int = 32,
        Bisection_kwargs: dict[str, Any] | None = None,
    ) -> at.QTable:
        """Compute the vertical period, action, and angle given input phase-space
        coordinates.

        Parameters
        ----------
        pos
            The position values.
        vel
            The velocity values.
        params
            A dictionary of model parameters.
        N_grid
            The number of grid points to use in estimating the action integral.
        """
        x = pos.decompose(self.units).value
        v = vel.decompose(self.units).value

        if Bisection_kwargs is None:
            Bisection_kwargs = {}

        T, J, th = self._get_T_J_theta(x, v, params, N_grid, Bisection_kwargs)

        tbl = at.QTable()
        tbl["T"] = T * self.units["time"]
        tbl["Omega"] = 2 * jnp.pi * u.rad / tbl["T"]
        tbl["J"] = J * self.units["length"] ** 2 / self.units["time"]
        tbl["theta"] = th * self.units["angle"]

        return tbl

    @partial(jax.jit, static_argnames=["self"])
    def _get_acc(
        self,
        pos: float,
        params: TorusImaging1DParams,
    ) -> jax.Array:
        r_e, _ = self._get_elliptical_coords(
            pos,
            0.0,
            pos0=params["pos0"],
            vel0=0.0,
            ln_Omega0=params["ln_Omega0"],
        )

        Om = jnp.exp(params["ln_Omega0"])

        es = self._get_es(r_e, params["e_params"])
        de_dres = self._get_de_dr_es(r_e, params["e_params"])

        numer = 1 + jnp.sum(
            jnp.array(
                [(-1) ** (m / 2) * (es[m] + de_dres[m] * r_e) for m in self.e_funcs]
            )
        )
        denom = 1 + jnp.sum(
            jnp.array(
                [
                    (-1) ** (m / 2) * (es[m] * (1 - m**2) + de_dres[m] * r_e)
                    for m in self.e_funcs
                ]
            )
        )
        return -(Om**2) * (pos - params["pos0"]) * numer / denom

    _get_dacc_dpos = jax.grad(_get_acc, argnums=1)
    _get_dacc_dpos_vmap = jax.vmap(_get_dacc_dpos, in_axes=(None, 0, None))

    @u.quantity_input
    def get_acceleration(
        self,
        pos: u.Quantity[length_pt],
        params: TorusImaging1DParams,
    ) -> u.Quantity:
        """Compute the acceleration as a function of position in the limit as velocity
        goes to zero

        Parameters
        ----------
        pos
            The position values.
        params
            A dictionary of model parameters.
        """
        x = jnp.atleast_1d(pos.decompose(self.units).value)
        in_shape = x.shape
        x = x.ravel()

        get_acc = jax.vmap(self._get_acc, in_axes=[0, None])
        res = get_acc(x, params)
        return res.reshape(in_shape) * self.units["acceleration"]

    @u.quantity_input
    def get_acceleration_deriv(
        self,
        pos: u.Quantity[length_pt],
        params: TorusImaging1DParams,
    ) -> u.Quantity:
        """Compute the derivative of the acceleration with respect to position as a
        function of position in the limit as velocity goes to zero

        Parameters
        ----------
        pos
            The position values.
        params
            A dictionary of model parameters.
        """
        x = jnp.atleast_1d(pos.decompose(self.units).value)
        in_shape = x.shape
        x = x.ravel()

        res = self._get_dacc_dpos_vmap(x, params)
        return res.reshape(in_shape) * self.units["acceleration"] / self.units["length"]

    @u.quantity_input
    def get_label(
        self,
        pos: u.Quantity[length_pt],
        vel: u.Quantity[velocity_pt],
        params: TorusImaging1DParams,
    ) -> jax.Array:
        """Compute the model predicted label value given the input phase-space
        coordinates
        """
        x = pos.decompose(self.units).value
        v = vel.decompose(self.units).value
        return self._get_label(x.ravel(), v.ravel(), params).reshape(x.shape)

    @partial(jax.jit, static_argnames=["self"])
    def ln_poisson_likelihood(
        self,
        params: TorusImaging1DParams,
        pos: jax.Array,
        vel: jax.Array,
        counts: jax.Array,
    ) -> jax.Array:
        """Compute the log-likelihood of the Poisson likelihood function. This should
        be used when the label you are modeling is the log-number of stars per pixel,
        i.e. the phase-space density itself.

        Note: the input position and velocity arrays must already be converted to the
        unit system of the model.
        """
        # Expected number:
        ln_Lambda = self._get_label(pos, vel, params)

        # gammaln(x+1) = log(factorial(x))
        return (counts * ln_Lambda - jnp.exp(ln_Lambda) - gammaln(counts + 1)).sum()

    @partial(jax.jit, static_argnames=["self"])
    def ln_gaussian_likelihood(
        self,
        params: TorusImaging1DParams,
        pos: jax.Array,
        vel: jax.Array,
        label: jax.Array,
        label_err: jax.Array,
    ) -> jax.Array:
        """Compute the log-likelihood of the Gaussian likelihood function.

        Note: the input position and velocity arrays must already be converted to the
        unit system of the model.
        """
        model_label = self._get_label(pos, vel, params)
        return -0.5 * jnp.nansum((label - model_label) ** 2 / label_err**2)

    @partial(jax.jit, static_argnames=["self"])
    def objective_poisson(
        self,
        params: TorusImaging1DParams,
        pos: jax.Array,
        vel: jax.Array,
        counts: npt.ArrayLike,
    ):
        f_val = self.ln_poisson_likelihood(params, pos, vel, counts)
        return -(f_val - self.regularization_func(self, params)) / pos.size

    @partial(jax.jit, static_argnames=["self"])
    def objective_gaussian(
        self,
        params: TorusImaging1DParams,
        pos: jax.Array,
        vel: jax.Array,
        label: jax.Array,
        label_err: jax.Array,
    ):
        f_val = self.ln_gaussian_likelihood(params, pos, vel, label, label_err)
        return -(f_val - self.regularization_func(self, params)) / pos.size

    def optimize(
        self,
        params0: dict,
        objective: Literal["poisson", "gaussian"],
        bounds: tuple[dict] | None = None,
        jaxopt_kwargs: dict | None = None,
        **data: u.Quantity | jtp.ArrayLike,
    ) -> OptStep:
        """Optimize the model parameters given the input data using
        `jaxopt.ScipyboundedMinimize`.

        Parameters
        ----------
        params0
            The initial values of the parameters.
        objective
            The string name of the objective function to use (either "poisson" or
            "gaussian").
        bounds
            The bounds on the parameters. This can either be a tuple of dictionaries, or
            a dictionary of tuples (keyed by parameter names) to specify the lower and
            upper bounds for each parameter.
        jaxopt_kwargs
            Any keyword arguments passed to ``jaxopt.ScipyBoundedMinimize``.
        **data
            Passed through to the objective function.

        """
        import numpy as np

        if jaxopt_kwargs is None:
            jaxopt_kwargs = {}
        jaxopt_kwargs.setdefault("maxiter", 16384)

        vals, treedef = jax.tree_util.tree_flatten(params0)
        params0 = treedef.unflatten([np.array(x, dtype=np.float64) for x in vals])

        jaxopt_kwargs.setdefault("method", "L-BFGS-B")
        optimizer = jaxopt.ScipyBoundedMinimize(
            fun=getattr(self, f"objective_{objective}"),
            **jaxopt_kwargs,
        )

        data = {k: jnp.array(v) for k, v in data.items()}

        if bounds is not None:
            # Detect packed bounds (a single dict):
            if isinstance(bounds, dict):
                bounds = self.unpack_bounds(bounds)

            res = optimizer.run(init_params=params0, bounds=bounds, **data)

        else:
            res = optimizer.run(init_params=params0, **data)

        # warn if optimization was not successful, set state if successful
        if not res.state.success:
            warn(
                "Optimization failed! See the returned result object for more "
                "information, but the model state was not updated",
                stacklevel=1,
            )

        return res

    @classmethod
    def unpack_bounds(cls, bounds: dict) -> tuple[dict]:
        """Split a bounds dictionary that is specified like: {"key": (lower, upper)}
        into two bounds dictionaries for the lower and upper bounds separately, e.g.,
        for the example above: {"key": lower} and {"key": upper}.
        """
        import numpy as np

        def clean_dict(d):
            if isinstance(d, dict):
                return {k: clean_dict(v) for k, v in d.items()}

            d = np.array(d)
            assert d.shape[0] == 2
            return d

        # Make sure all tuples / lists become arrays:
        clean_bounds = clean_dict(bounds)

        vals, treedef = jax.tree_util.tree_flatten(clean_bounds)

        bounds_l = treedef.unflatten([np.array(x[0], dtype=np.float64) for x in vals])
        bounds_r = treedef.unflatten([np.array(x[1], dtype=np.float64) for x in vals])

        return bounds_l, bounds_r

    def check_e_funcs(
        self, e_params: EParams, r_e_max: float
    ) -> tuple[bool, npt.NDArray]:
        """Check that the parameter values and functions used for the e functions are
        valid given the condition that d(r)/d(r_e) > 0.
        """
        import numpy as np

        # TODO: 16 is a magic number
        r_es = np.linspace(np.sqrt(1e-3), np.sqrt(r_e_max), 16) ** 2

        # TODO: potential issue if order of arguments in e_funcs() call is different
        # from the order of the values in the e_params dictionary...
        d_em_d_re_funcs = {
            m: jax.vmap(
                jax.grad(self.e_funcs[m], argnums=0),
                in_axes=[0] + [None] * len(e_params[m]),
            )
            for m in self.e_funcs
        }

        thes = np.linspace(0, np.pi / 2, 128)
        checks = np.zeros((len(r_es), len(thes)))

        for j, th_e in enumerate(thes):
            checks[:, j] = jnp.sum(
                jnp.array(
                    [
                        jnp.cos(m * th_e)
                        * (
                            e_func(r_es, **e_params[m])
                            + r_es * d_em_d_re_funcs[m](r_es, *e_params[m].values())
                        )
                        for m, e_func in self.e_funcs.items()
                    ]
                ),
                axis=0,
            )

        # This condition has to be met such that d(r_z)/d(r_z') > 0 at all theta_z':
        return np.all(checks > -1), checks

    def get_crlb(
        self,
        params: TorusImaging1DParams,
        data: dict[str, npt.ArrayLike],
        objective: str = "gaussian",
        inv: bool = False,
    ) -> npt.NDArray:
        """Returns the Cramer-Rao lower bound matrix for the parameters evaluated at the
        input parameter values.

        To instead return the Fisher information matrix, specify ``inv=True``.
        """
        import numpy as np

        treedef = jax.tree_util.tree_structure(params)

        def wrapper(flat_params, data, sizes):
            arrs = []
            i = 0
            for size in sizes:
                arrs.append(jnp.array(flat_params[i : i + size]))
                i += size
            params = jax.tree_util.tree_unflatten(treedef, arrs)
            ll = getattr(self, f"ln_{objective}_likelihood")(params, **data)
            return -(ll - self.regularization_func(self, params))

        flattened = jax.tree_util.tree_flatten(params)[0]
        sizes = [x.size for x in flattened]
        flat_params = np.concatenate([np.atleast_1d(x) for x in flattened])

        fisher = jax.hessian(wrapper)(flat_params, data, sizes)
        if inv:
            return fisher
        return np.linalg.inv(fisher)

    def get_crlb_uncertainties(
        self,
        params: TorusImaging1DParams,
        data: dict[str, npt.ArrayLike],
        objective: str = "gaussian",
    ) -> dict[str, dict | npt.ArrayLike]:
        """Compute the uncertainties on the parameters using the diagonal of the
        Cramer-Rao lower bound matrix (see :meth:`get_crlb`).
        """
        import numpy as np

        treedef = jax.tree_util.tree_structure(params)
        flattened = jax.tree_util.tree_flatten(params)[0]
        sizes = [x.size for x in flattened]

        fisher_inv = self.get_crlb(params, data, objective=objective)
        diag = np.diag(fisher_inv).copy()
        diag[(diag < 0) | (diag > 1e18)] = 0.0
        flat_param_uncs = np.sqrt(diag)

        arrs = []
        i = 0
        for size in sizes:
            arrs.append(jnp.array(flat_param_uncs[i : i + size]))
            i += size
        return jax.tree_util.tree_unflatten(treedef, arrs)

    def get_crlb_error_samples(
        self,
        params: TorusImaging1DParams,
        data: dict[str, npt.ArrayLike],
        objective: str = "gaussian",
        size: int = 1,
        seed: int | None = None,
        list_of_samples: bool = True,
    ) -> list[dict] | dict[str, dict | npt.ArrayLike]:
        """Generate Gaussian samples of parameter values centered on the input parameter
        values with covariance matrix set by the Cramer-Rao lower bound matrix.
        """
        import numpy as np

        treedef = jax.tree_util.tree_structure(params)
        flattened = jax.tree_util.tree_flatten(params)[0]
        sizes = [x.size for x in flattened]
        flat_params = np.concatenate([np.atleast_1d(x) for x in flattened])

        crlb = self.get_crlb(params, data, objective=objective)
        diag = np.diag(crlb)
        bad_idx = np.where((diag < 0) | (diag > 1e18))[0]

        for i in bad_idx:
            crlb[i] = crlb[:, i] = 0.0
            crlb[i, i] = 1.0

        rng = np.random.default_rng(seed=seed)
        samples = rng.multivariate_normal(flat_params, crlb, size=size)

        for i in bad_idx:
            samples[:, i] = np.nan

        arrs = []
        i = 0
        for size_ in sizes:
            arrs.append(jnp.array(samples[..., i : i + size_]))
            i += size_

        if list_of_samples:
            samples = []
            for n in range(size):
                samples.append(
                    jax.tree_util.tree_unflatten(treedef, [arr[n] for arr in arrs])
                )
            return samples

        return jax.tree_util.tree_unflatten(treedef, arrs)

    def mcmc_run_label(
        self,
        binned_data: dict,
        p0: dict,
        bounds: tuple[dict] | None = None,
        rng_seed: int = 0,
        num_steps: int = 1000,
        num_warmup: int = 1000,
    ) -> tuple[Any, list[dict]]:
        """Currently only supports uniform priors on all parameters, specified by the
        input bounds.

        Parameters
        ----------
        binned_data
            A dictionary containing the binned label moment data.
        p0
            The initial values of the parameters.
        bounds
            The bounds on the parameters, used to define uniform priors on the
            parameters. This can either be a tuple of dictionaries, or a dictionary of
            tuples (keyed by parameter names) to specify the lower and upper bounds for
            each parameter.
        rng_seed
            The random number generator seed.
        num_steps
            The number of MCMC steps to take.
        num_warmup
            The number of warmup or burn-in steps to take to tune the NUTS sampler.

        Returns
        -------
        state
            The HMCState object returned by BlackJAX.
        mcmc_samples
            A list of dictionaries containing the parameter values for each MCMC sample.
        """
        import blackjax
        import numpy as np

        # First check that objective evaluates to a finite value:
        mask = (
            np.isfinite(binned_data["label"])
            & np.isfinite(binned_data["label_err"])
            & (binned_data["label_err"] > 0)
        )
        data = {
            "pos": binned_data["pos"].decompose(self.units).value[mask],
            "vel": binned_data["vel"].decompose(self.units).value[mask],
            "label": binned_data["label"][mask],
            "label_err": binned_data["label_err"][mask],
        }
        test_val = self.objective_gaussian(p0, **data)
        if not np.isfinite(test_val):
            msg = "Objective function evaluated to non-finite value"
            raise RuntimeError(msg)

        lb, ub = self.unpack_bounds(bounds)
        lb_arrs = jax.tree_util.tree_flatten(lb)[0]
        ub_arrs = jax.tree_util.tree_flatten(ub)[0]

        def logprob(p):
            lp = 0.0
            pars, _ = jax.tree_util.tree_flatten(p)
            for i in range(len(pars)):
                lp += jnp.where(
                    jnp.any(pars[i] < lb_arrs[i]) | jnp.any(pars[i] > ub_arrs[i]),
                    -jnp.inf,
                    0.0,
                )

            lp += self.ln_gaussian_likelihood(p, **data)

            lp -= self.regularization_func(self, p)

            return lp

        rng_key = jax.random.PRNGKey(rng_seed)
        warmup = blackjax.window_adaptation(blackjax.nuts, logprob)
        (state, parameters), _ = warmup.run(rng_key, p0, num_steps=num_warmup)

        kernel = blackjax.nuts(logprob, **parameters).step  # pylint: disable=no-member
        states = inference_loop(rng_key, kernel, state, num_steps)

        # Get the pytree structure of a single sample based on the starting point:
        treedef = jax.tree_util.tree_structure(p0)
        arrs, _ = jax.tree_util.tree_flatten(states.position)

        mcmc_samples = []
        for n in range(arrs[0].shape[0]):
            mcmc_samples.append(
                jax.tree_util.tree_unflatten(treedef, [arr[n] for arr in arrs])
            )

        return states, mcmc_samples


def inference_loop(
    rng_key: jax.random.PRNGKey, kernel: Any, initial_state: Any, num_samples: int
) -> Any:
    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states
