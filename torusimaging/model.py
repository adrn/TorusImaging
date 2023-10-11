from functools import partial
from typing import Literal, Optional
from warnings import warn

import astropy.table as at
import astropy.units as u
import jax
import jax.numpy as jnp
import jaxopt
import numpy.typing as npt
from gala.units import UnitSystem, galactic
from jax.scipy.special import gammaln
from jaxopt import Bisection

from torusimaging.jax_helpers import simpson

__all__ = ["TorusImaging1D"]


class TorusImaging1D:
    def __init__(
        self,
        label_func,
        e_funcs,
        regularization_func=None,
        unit_sys=galactic,
        Bisection_kwargs=None,
    ):
        r"""
        TODO: subclass for model with splines everywhere?
        TODO: Automatic creation with SplineTorusImaging1D.auto()

        This inherently assumes that you are working in a 1D phase space with position
        coordinate ``x`` and velocity coordinate ``v``.

        Notation:
        - :math:`\Omega_0` or ``Omega0``: A scale frequency used to compute the
          elliptical radius ``r_e``. This is the asymptotic orbital frequency at zero
          action.
        - :math:`r_e` or ``r_e``: The elliptical radius
          :math:`r_e = \sqrt{x^2\, \Omega_0 + v^2 \, \Omega_0^{-1}}`.
        - :math:`\theta_e` or ``theta_e``: The elliptical angle defined as
          :math:`\tan{\theta_e} = \frac{x}{v}\,\Omega_0`.
        - :math:`r` or ``r``: The distorted elliptical radius
          :math:`r = r_e \, f(r_e, \theta_e)`, which is close  to :math:`\sqrt{J}` (the
          action) and so we sometimes call it the "proxy action" below. :math:`f(\cdot)`
          is the distortion function defined below.
        - :math:`f(r_e, \theta_e)`: The distortion function is a Fourier expansion, \
          defined as: :math:`f(r_e, \theta_e) = 1 + \sum_m e_m(r_e)\,\cos(m\,\theta_e)`
        - :math:`J` or ``J``: The action.
        - :math:`\theta` or ``theta``: The conjugate angle.

        Parameters
        ----------
        e_funcs : dict
            A dictionary that provides functions that specify the dependence of the
            Fourier distortion coefficients :math:`e_m(r_e)`. Keys should be the
            (integer) "m" order of the distortion term (for the distortion function),
            and values should be Python callable objects that can be passed to
            `jax.jit()`. The first argument of each of these functions should be the
            elliptical radius :math:`r_e` or ``re``.
        regularization_func : callable (optional)
            An optional function that computes a regularization term to add to the
            objective function when optimizing.
        unit_sys : `gala.units.UnitSystem` (optional)
            The unit system to work in. Default is to use the "galactic" unit system
            from Gala: (kpc, Myr, Msun, radian).

        """
        self.label_func = jax.jit(label_func)
        self.e_funcs = {int(m): jax.jit(e_func) for m, e_func in e_funcs.items()}

        # Unit system:
        self.unit_sys = UnitSystem(unit_sys)

        if regularization_func is None:
            regularization_func = lambda *_, **__: 0.0  # noqa
        self.regularization_func = regularization_func

        if Bisection_kwargs is None:
            Bisection_kwargs = {}
        self.Bisection_kwargs = dict(Bisection_kwargs)
        self.Bisection_kwargs.setdefault("lower", 0.0)
        self.Bisection_kwargs.setdefault("upper", 1.0)
        self.Bisection_kwargs.setdefault("maxiter", 30)
        self.Bisection_kwargs.setdefault("tol", 1e-4)

    # ---------------------------------------------------------------------------------
    # Internal functions used within likelihood functions:
    #
    @partial(jax.jit, static_argnames=["self"])
    def _get_elliptical_coords(self, pos, vel, params):
        r"""
        Compute the raw elliptical radius :math:`r_e` (``r_e``) and angle
        :math:`\theta_e'` (``theta_e``)

        Parameters
        ----------
        pos : numeric, array-like
        vel : numeric, array-like
        params : dict
        """
        x = (vel - params["vel0"]) / jnp.sqrt(jnp.exp(params["ln_Omega0"]))
        y = (pos - params["pos0"]) * jnp.sqrt(jnp.exp(params["ln_Omega0"]))

        r_e = jnp.sqrt(x**2 + y**2)
        t_e = jnp.arctan2(y, x)

        return r_e, t_e

    @partial(jax.jit, static_argnames=["self"])
    def _get_es(self, r_e, e_params):
        """
        Compute the Fourier m-order distortion coefficients

        Parameters
        ----------
        r_e : numeric, array-like
        e_params : dict
        """
        es = {}
        for m, pars in e_params.items():
            es[m] = self.e_funcs[m](r_e, **pars)
        return es

    @partial(jax.jit, static_argnames=["self"])
    def _get_r(self, r_e, theta_e, e_params):
        """
        Compute the distorted radius :math:`r`

        Parameters
        ----------
        r_e : numeric, array-like
        theta_e : numeric, array-like
        e_params : dict
        """
        es = self.get_es(r_e, e_params)
        return r_e * (
            1
            + jnp.sum(
                jnp.array([e * jnp.cos(m * theta_e) for m, e in es.items()]), axis=0
            )
        )

    @partial(jax.jit, static_argnames=["self"])
    def _get_theta(self, r_e, theta_e, e_params):
        """
        Compute the phase angle

        Parameters
        ----------
        rz_prime : numeric, array-like
        theta_prime : numeric, array-like
        e_params : dict
        """
        es = self.get_es(r_e, e_params)
        # TODO: why is the π/2 needed below??
        return theta_e - jnp.sum(
            jnp.array(
                [m / (jnp.pi / 2) * e * jnp.sin(m * theta_e) for m, e in es.items()]
            ),
            axis=0,
        )

    @partial(jax.jit, static_argnames=["self"])
    def _get_r_e(self, r, theta_e, e_params):
        """
        Compute the elliptical radius :math:`r_e` by inverting the distortion
        transformation from :math:`r`

        Parameters
        ----------
        r : numeric
            The distorted radius.
        theta_e : numeric
            The elliptical angle.
        e_params : dict
            Dictionary of parameter values for the distortion coefficient (e) functions.
        """
        bisec = Bisection(
            lambda x, rrz, tt_prime, ee_params: self.get_r(x, tt_prime, ee_params)
            - rrz,
            jit=True,
            unroll=True,
            check_bracket=False,
            **self.Bisection_kwargs,
        )
        return bisec.run(r, rrz=r, tt_prime=theta_e, ee_params=e_params).params

    @partial(jax.jit, static_argnames=["self"])
    def _get_pos(self, r, theta_e, params):
        """
        Compute the position given the distorted radius and elliptical angle
        """
        r_e = self.get_r_e(r, theta_e, params["e_params"])
        return r_e * jnp.sin(theta_e) / jnp.sqrt(jnp.exp(params["ln_Omega0"]))

    @partial(jax.jit, static_argnames=["self"])
    def _get_vel(self, r, theta_e, params):
        """
        Compute the velocity given the distorted radius and elliptical angle
        """
        rzp = self.get_r_e(r, theta_e, params["e_params"])
        return rzp * jnp.cos(theta_e) * jnp.sqrt(jnp.exp(params["ln_Omega0"]))

    @partial(jax.jit, static_argnames=["self"])
    def _get_label(self, pos, vel, params):
        r_e, th_e = self._get_elliptical_coords(pos, vel, params)
        r = self.get_r(r_e, th_e, params["e_params"])
        return self.label_func(r, **params["label_params"])

    @partial(jax.jit, static_argnames=["self", "N_grid"])
    def _get_T_J_theta(self, pos, vel, params, N_grid):
        re_, the_ = self.get_elliptical_coords(pos, vel, params)
        r = self.get_r(re_, the_, params["e_params"])

        dpos_dthe_func = jax.vmap(
            jax.grad(self.get_pos, argnums=1), in_axes=[None, 0, None]
        )

        get_vel = jax.vmap(self.get_vel, in_axes=[None, 0, None])

        # Grid of theta_prime to do the integral over:
        the_grid = jnp.linspace(0, jnp.pi / 2, N_grid)
        v_th = get_vel(r, the_grid, params)
        dz_dthp = dpos_dthe_func(r, the_grid, params)

        Tz = 4 * simpson(dz_dthp / v_th, the_grid)
        Jz = 4 / (2 * jnp.pi) * simpson(dz_dthp * v_th, the_grid)

        thp_partial = jnp.linspace(0, the_, N_grid)
        v_th_partial = get_vel(r, thp_partial, params)
        dpos_dthe_partial = dpos_dthe_func(r, thp_partial, params)
        dt = simpson(dpos_dthe_partial / v_th_partial, thp_partial)
        thz = 2 * jnp.pi * dt / Tz

        return Tz, Jz, thz

    _get_T_J_theta = jax.vmap(_get_T_J_theta, in_axes=[None, 0, 0, None, None])

    @partial(jax.jit, static_argnames=["self"])
    def _get_de_dr_es(self, r_e, e_params):
        """
        Compute the derivatives of the Fourier m-order distortion coefficient functions

        Parameters
        ----------
        r_e : numeric, array-like
        e_params : dict
        """
        d_es = {}
        for m, pars in e_params.items():
            # Workaround because of:
            # https://github.com/google/jax/issues/7465
            tmp = jax.vmap(partial(jax.grad(self.e_funcs[m], argnums=0), **pars), 0, 0)
            d_es[m] = tmp(r_e)
        return d_es

    # ---------------------------------------------------------------------------------
    # Public API
    #
    @u.quantity_input
    def compute_elliptical(self, pos: u.kpc, vel: u.km / u.s, params):
        """
        Compute the elliptical radius :math:`r_e` (``r_e``) and angle :math:`\theta_e'`
        (``theta_e``)

        Parameters
        ----------
        pos : `astropy.units.Quantity`
        vel : `astropy.units.Quantity`
        params : dict
        """

        x = pos.decompose(self.unit_sys).value
        v = vel.decompose(self.unit_sys).value
        re, te = self._get_elliptical_coords(x, v, params)
        return (
            re
            * self.unit_sys["length"]
            / (self.unit_sys["angle"] ** 0.5 / self.unit_sys["time"] ** 0.5),
            te * self.unit_sys["angle"],
        )

    @u.quantity_input
    def compute_action_angle(self, pos: u.kpc, vel: u.km / u.s, params, N_grid=32):
        """
        Compute the vertical period, action, and angle given input phase-space
        coordinates.

        Parameters
        ----------
        pos : `astropy.units.Quantity`
        vel : `astropy.units.Quantity`
        params : dict
        N_grid : int (optional)
        """
        x = pos.decompose(self.unit_sys).value
        v = vel.decompose(self.unit_sys).value
        T, J, th = self._get_T_J_theta(x, v, params, N_grid)

        tbl = at.QTable()
        tbl["T"] = T * self.unit_sys["time"]
        tbl["Omega"] = 2 * jnp.pi * u.rad / tbl["T"]
        tbl["J"] = J * self.unit_sys["length"] ** 2 / self.unit_sys["time"]
        tbl["theta"] = th * self.unit_sys["angle"]

        return tbl

    @u.quantity_input
    def get_acceleration(self, pos: u.kpc, params):
        """
        Compute the acceleration as a function of position in the limit as velocity
        goes to zero

        Parameters
        ----------
        pos : `astropy.units.Quantity`
        params : dict
        """
        x = jnp.atleast_1d(pos.decompose(self.unit_sys).value)
        in_shape = x.shape
        x = x.ravel()

        r_e, _ = self.get_elliptical_coords(x, jnp.zeros_like(x), params)

        Om = jnp.exp(params["ln_Omega0"])

        es = self.get_es(r_e, params["e_params"])
        de_dres = self._get_de_dr_es(r_e, params["e_params"])

        numer = 1 + jnp.sum(
            jnp.array(
                [
                    (-1) ** (m / 2) * (es[m] + de_dres[m] * r_e)
                    for m in self.e_funcs.keys()
                ]
            ),
            axis=0,
        )
        denom = 1 + jnp.sum(
            jnp.array(
                [
                    (-1) ** (m / 2) * (es[m] * (1 - m**2) + de_dres[m] * r_e)
                    for m in self.e_funcs.keys()
                ]
            ),
            axis=0,
        )
        res = -(Om**2) * x * numer / denom

        return res.reshape(in_shape) * self.unit_sys["acceleration"]

    @partial(jax.jit, static_argnames=["self"])
    def ln_poisson_likelihood(self, params, pos, vel, counts):
        # Expected number:
        ln_Lambda = self._get_label(pos, vel, params)

        # gammaln(x+1) = log(factorial(x))
        return (counts * ln_Lambda - jnp.exp(ln_Lambda) - gammaln(counts + 1)).sum()

    @partial(jax.jit, static_argnames=["self"])
    def ln_gaussian_likelihood(self, params, pos, vel, label, label_err):
        model_label = self._get_label(pos, vel, **params)
        return -0.5 * jnp.nansum((label - model_label) ** 2 / label_err**2)

    @partial(jax.jit, static_argnames=["self"])
    def objective_poisson(self, params, pos, vel, counts):
        f_val = self.ln_poisson_likelihood(params, pos, vel, counts)
        return -(f_val - self.regularization_func(params)) / pos.size

    @partial(jax.jit, static_argnames=["self"])
    def objective_gaussian(self, params, pos, vel, label, label_err):
        f_val = self.ln_gaussian_likelihood(params, pos, vel, label, label_err)
        return -(f_val - self.regularization_func(params)) / pos.size

    def optimize(
        self,
        params0: dict,
        objective: Literal["poisson", "gaussian"],
        bounds: Optional[tuple[dict]] = None,
        jaxopt_kwargs: Optional[dict] = None,
        **data: npt.ArrayLike,
    ) -> jaxopt.base.OptStep:
        """
        Optimize the model parameters given the input data using
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
            jaxopt_kwargs = dict()
        jaxopt_kwargs.setdefault("maxiter", 16384)

        vals, treedef = jax.tree_util.tree_flatten(params0)
        params0 = treedef.unflatten([np.array(x, dtype=np.float64) for x in vals])

        jaxopt_kwargs.setdefault("method", "L-BFGS-B")
        optimizer = jaxopt.ScipyBoundedMinimize(
            fun=getattr(self, f"objective_{objective}"),
            **jaxopt_kwargs,
        )

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
                "information, but the model state was not updated"
            )

        return res

    @classmethod
    def unpack_bounds(cls, bounds: dict) -> tuple[dict]:
        """
        Split a bounds dictionary that is specified like: {"key": (lower, upper)} into
        two bounds dictionaries for the lower and upper bounds separately, e.g., for the
        example above: {"key": lower} and {"key": upper}.
        """
        import numpy as np

        def clean_dict(d):
            if isinstance(d, dict):
                return {k: clean_dict(v) for k, v in d.items()}
            else:
                d = np.array(d)
                assert d.shape[0] == 2
                return d

        # Make sure all tuples / lists become arrays:
        clean_bounds = clean_dict(bounds)

        vals, treedef = jax.tree_util.tree_flatten(clean_bounds)

        bounds_l = treedef.unflatten([np.array(x[0], dtype=np.float64) for x in vals])
        bounds_r = treedef.unflatten([np.array(x[1], dtype=np.float64) for x in vals])

        return bounds_l, bounds_r

    def check_e_funcs(self, e_params: dict, r_e_max: float):
        """
        Check that the parameter values and functions used for the e functions
        are valid given the condition that d(r)/d(r_e) > 0.
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
            for m in self.e_funcs.keys()
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
        params: dict[str, dict | npt.ArrayLike],
        data: dict[str, npt.ArrayLike],
        inv: bool = False,
    ) -> npt.NDArray:
        """
        Returns the Cramer-Rao lower bound matrix for the parameters evaluated at the
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
            ll = getattr(self, self._objective_func)(params, **data)
            return -(ll - self.regularization_func(params))

        flattened = jax.tree_util.tree_flatten(params)[0]
        sizes = [x.size for x in flattened]
        flat_params = np.concatenate([np.atleast_1d(x) for x in flattened])

        fisher = jax.hessian(wrapper)(flat_params, data, sizes)
        if inv:
            return fisher
        fisher_inv = np.linalg.inv(fisher)

        return fisher_inv

    def get_crlb_uncertainties(
        self,
        params: dict[str, dict | npt.ArrayLike],
        data: dict[str, npt.ArrayLike],
    ) -> dict[str, dict | npt.ArrayLike]:
        """
        Compute the uncertainties on the parameters using the diagonal of the Cramer-Rao
        lower bound matrix (see :meth:`get_crlb`).
        """
        import numpy as np

        treedef = jax.tree_util.tree_structure(params)
        flattened = jax.tree_util.tree_flatten(params)[0]
        sizes = [x.size for x in flattened]

        fisher_inv = self.get_crlb(params, data)
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
        params: dict[str, dict | npt.ArrayLike],
        data: dict[str, npt.ArrayLike],
        size: int = 1,
        seed: Optional[int] = None,
        list_of_samples: bool = True,
    ) -> list[dict] | dict[str, dict | npt.ArrayLike]:
        """
        Generate Gaussian samples of parameter values centered on the input parameter
        values with covariance matrix set by the Cramer-Rao lower bound matrix.
        """
        import numpy as np

        treedef = jax.tree_util.tree_structure(params)
        flattened = jax.tree_util.tree_flatten(params)[0]
        sizes = [x.size for x in flattened]
        flat_params = np.concatenate([np.atleast_1d(x) for x in flattened])

        crlb = self.get_crlb(params, data)
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
        else:
            return jax.tree_util.tree_unflatten(treedef, arrs)
