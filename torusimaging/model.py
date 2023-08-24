from functools import partial
from warnings import warn

import astropy.table as at
import astropy.units as u
import jax
import jax.numpy as jnp
import jaxopt
import scipy.interpolate as sci
from astropy.stats import median_absolute_deviation as MAD
from gala.units import UnitSystem, galactic
from jax.scipy.special import gammaln
from jaxopt import Bisection
from scipy.stats import binned_statistic

from torusimaging.jax_helpers import simpson

__all__ = ["DensityOrbitModel", "LabelOrbitModel"]


class OrbitModelBase:
    def __init__(self, e_funcs, regularization_func=None, unit_sys=galactic):
        r"""
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
        self.e_funcs = {int(m): jax.jit(e_func) for m, e_func in e_funcs.items()}

        # Unit system:
        self.unit_sys = UnitSystem(unit_sys)

        if regularization_func is None:
            regularization_func = lambda *_, **__: 0.0  # noqa
        self.regularization_func = regularization_func

    @partial(jax.jit, static_argnames=["self"])
    def get_elliptical_coords(self, pos, vel, params):
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
    def get_es(self, r_e, e_params):
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
    def get_r(self, r_e, theta_e, e_params):
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
    def get_theta(self, r_e, theta_e, e_params):
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
    def get_r_e(self, r, theta_e, e_params):
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
        # TODO: lots of numbers are hard-set below!
        bisec = Bisection(
            lambda x, rrz, tt_prime, ee_params: self.get_r(x, tt_prime, ee_params)
            - rrz,
            lower=0.0,
            upper=1.0,
            maxiter=30,
            jit=True,
            unroll=True,
            check_bracket=False,
            tol=1e-4,
        )
        return bisec.run(r, rrz=r, tt_prime=theta_e, ee_params=e_params).params

    @partial(jax.jit, static_argnames=["self"])
    def get_pos(self, r, theta_e, params):
        """
        Compute the position given the distorted radius and elliptical angle
        """
        r_e = self.get_r_e(r, theta_e, params["e_params"])
        return r_e * jnp.sin(theta_e) / jnp.sqrt(jnp.exp(params["ln_Omega0"]))

    @partial(jax.jit, static_argnames=["self"])
    def get_vel(self, r, theta_e, params):
        """
        Compute the velocity given the distorted radius and elliptical angle
        """
        rzp = self.get_r_e(r, theta_e, params["e_params"])
        return rzp * jnp.cos(theta_e) * jnp.sqrt(jnp.exp(params["ln_Omega0"]))

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

    def get_acceleration(self, pos, params):
        """
        Experimental.

        Implementation of Appendix math from empirical-af paper.
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
    def objective(self, params, pos, vel, *args, **kwargs):
        f = getattr(self, self._objective_func)
        f_val = f(params, pos, vel, *args, **kwargs)
        return -(f_val - self.regularization_func(params)) / pos.size

    def optimize(self, params0, bounds=None, jaxopt_kwargs=None, **data):
        """
        Parameters
        ----------
        params0 : dict (optional)
        bounds : tuple of dict (optional)
        jaxopt_kwargs : dict (optional)
        **data
            Passed through to the objective function

        Returns
        -------
        jaxopt_result : TODO
            TODO
        """
        import numpy as np

        if jaxopt_kwargs is None:
            jaxopt_kwargs = dict()
        jaxopt_kwargs.setdefault("maxiter", 16384)

        vals, treedef = jax.tree_util.tree_flatten(params0)
        params0 = treedef.unflatten([np.array(x, dtype=np.float64) for x in vals])

        if bounds is not None:
            # Detect packed bounds (a single dict):
            if isinstance(bounds, dict):
                bounds = self.unpack_bounds(bounds)

            jaxopt_kwargs.setdefault("method", "L-BFGS-B")
            optimizer = jaxopt.ScipyBoundedMinimize(
                fun=self.objective,
                **jaxopt_kwargs,
            )
            res = optimizer.run(init_params=params0, bounds=bounds, **data)

        else:
            jaxopt_kwargs.setdefault("method", "BFGS")
            raise NotImplementedError("TODO")

        # warn if optimization was not successful, set state if successful
        if not res.state.success:
            warn(
                "Optimization failed! See the returned result object for more "
                "information, but the model state was not updated"
            )

        return res

    @classmethod
    def unpack_bounds(cls, bounds):
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

    def check_e_funcs(self, e_params, r_e_max=1.0):
        """
        Check that the parameter values and functions used for the e_m(r_z') functions
        are valid given the condition that d(r_z)/d(r_z') > 0.
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

    def get_crlb(self, params, data):
        """
        EXPERIMENTAL

        Returns the Cramer-Rao lower bound matrix (inverse of the Fisher information)
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
            return -2 * getattr(self, self._objective_func)(params, **data)

        flattened = jax.tree_util.tree_flatten(params)[0]
        sizes = [x.size for x in flattened]
        flat_params = np.concatenate([np.atleast_1d(x) for x in flattened])

        fisher = jax.hessian(wrapper)(flat_params, data, sizes)
        fisher_inv = np.linalg.inv(fisher)

        return fisher_inv

    def error_propagate_uncertainty(self, params, data):
        """
        EXPERIMENTAL

        Propagate uncertainty using the Fisher information matrix.
        """
        import numpy as np

        treedef = jax.tree_util.tree_structure(params)
        flattened = jax.tree_util.tree_flatten(params)[0]
        sizes = [x.size for x in flattened]

        fisher_inv = self.get_crlb(params, data)
        flat_param_uncs = np.sqrt(np.diag(fisher_inv))

        arrs = []
        i = 0
        for size in sizes:
            arrs.append(jnp.array(flat_param_uncs[i : i + size]))
            i += size
        return jax.tree_util.tree_unflatten(treedef, arrs)

    def get_crlb_error_samples(
        self, params, data, size=1, seed=None, list_of_samples=True
    ):
        import numpy as np

        treedef = jax.tree_util.tree_structure(params)
        flattened = jax.tree_util.tree_flatten(params)[0]
        sizes = [x.size for x in flattened]
        flat_params = np.concatenate([np.atleast_1d(x) for x in flattened])

        crlb = self.get_crlb(params, data)

        rng = np.random.default_rng(seed=seed)
        samples = rng.multivariate_normal(flat_params, crlb, size=size)

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


class DensityOrbitModel(OrbitModelBase):
    _objective_func = "ln_poisson_likelihood"

    def __init__(
        self, ln_dens_func, e_funcs=None, regularization_func=None, unit_sys=galactic
    ):
        """
        {intro}

        Parameters
        ----------
        ln_dens_func : callable (optional)
            TODO
        {params}
        """
        super().__init__(
            e_funcs=e_funcs, regularization_func=regularization_func, unit_sys=unit_sys
        )
        self.ln_dens_func = jax.jit(ln_dens_func)

    __init__.__doc__ = __init__.__doc__.format(
        intro=OrbitModelBase.__init__.__doc__.split("Parameters")[0].rstrip(),
        params=OrbitModelBase.__init__.__doc__.split("----------")[1].lstrip(),
    )

    @u.quantity_input
    def estimate_Omega0(self, pos: u.kpc, vel: u.km / u.s):
        """
        Estimate the asymptotic frequency and zero-points

        Parameters
        ----------
        pos : quantity-like or array-like
        vel : quantity-like or array-like
        """
        import numpy as np

        x = pos.decompose(self.unit_sys).value
        v = vel.decompose(self.unit_sys).value

        std_z = 1.5 * MAD(x, ignore_nan=True)
        std_vz = 1.5 * MAD(v, ignore_nan=True)
        nu = std_vz / std_z

        pars0 = {
            "pos0": np.nanmedian(x),
            "vel0": np.nanmedian(v),
            "ln_Omega0": np.log(nu),
        }
        return pars0

    @u.quantity_input
    def _estimate_data_ln_dens_func(
        self, pos: u.kpc, vel: u.km / u.s, pars0=None, N_r_bins=25, spl_k=3
    ):
        """
        Return a function to compute the log-density of the data

        Parameters
        ----------
        pos : quantity-like or array-like
        vel : quantity-like or array-like
        """
        import numpy as np

        if pars0 is None:
            pars0 = self.estimate_Omega0(pos, vel)

        x = pos.decompose(self.unit_sys).value
        v = vel.decompose(self.unit_sys).value

        r_e, _ = self.get_elliptical_coords(x, v, pars0)

        max_rz = np.nanpercentile(r_e, 99.5)
        rz_bins = np.linspace(0, max_rz, N_r_bins)
        dens_H, xe = np.histogram(r_e, bins=rz_bins)
        xc = 0.5 * (xe[:-1] + xe[1:])
        ln_dens = np.log(dens_H) - np.log(2 * np.pi * xc * (xe[1:] - xe[:-1]))

        # TODO: WTF - this is a total hack -- why is this needed???
        ln_dens = ln_dens - 8.6

        spl = sci.InterpolatedUnivariateSpline(xc, ln_dens, k=spl_k)
        return xc, ln_dens, spl

    @u.quantity_input
    def get_params_init(self, pos: u.kpc, vel: u.km / u.s, ln_dens_params0=None):
        """
        Estimate initial model parameters from the data

        Parameters
        ----------
        pos : quantity-like or array-like
        vel : quantity-like or array-like
        ln_dens_params0 : dict (optional)
        """
        import numpy as np

        pars0 = self.estimate_Omega0(pos, vel)
        xx, yy, ln_dens_spl = self._estimate_data_ln_dens_func(pos, vel, pars0)

        if ln_dens_params0 is not None:
            # Fit the specified ln_dens_func to the measured densities
            # This is a BAG O' HACKS!
            xeval = np.geomspace(1e-3, np.nanmax(xx), 32)  # MAGIC NUMBER:

            def obj(params, x, data_y):
                model_y = self.ln_dens_func(x, **params)
                return jnp.sum((model_y - data_y) ** 2)

            optim = jaxopt.ScipyMinimize(fun=obj, method="BFGS")
            res = optim.run(
                init_params=ln_dens_params0, x=xeval, data_y=ln_dens_spl(xeval)
            )
            if res.state.success:
                pars0["ln_dens_params"] = res.params
            else:
                warn(
                    "Initial parameter estimation failed: Failed to estimate "
                    "parameters of log-density function `ln_dens_func()`",
                    RuntimeWarning,
                )

        return pars0

    @partial(jax.jit, static_argnames=["self"])
    def get_ln_dens(self, r, params):
        return self.ln_dens_func(r, **params["ln_dens_params"])

    @partial(jax.jit, static_argnames=["self"])
    def ln_density(self, pos, vel, params):
        r_e, th_e = self.get_elliptical_coords(pos, vel, params)
        r = self.get_r(r_e, th_e, params["e_params"])
        return self.get_ln_dens(r, params)

    @partial(jax.jit, static_argnames=["self"])
    def ln_poisson_likelihood(self, params, pos, vel, dens):
        # Expected number:
        ln_Lambda = self.ln_density(pos, vel, params)

        # gammaln(x+1) = log(factorial(x))
        return (dens * ln_Lambda - jnp.exp(ln_Lambda) - gammaln(dens + 1)).sum()


class LabelOrbitModel(OrbitModelBase):
    _objective_func = "ln_label_likelihood"

    def __init__(
        self, label_func, e_funcs=None, regularization_func=None, unit_sys=galactic
    ):
        """
        {intro}

        Parameters
        ----------
        label_func : callable (optional)
            TODO
        {params}
        """
        super().__init__(
            e_funcs=e_funcs, regularization_func=regularization_func, unit_sys=unit_sys
        )
        self.label_func = jax.jit(label_func)

    __init__.__doc__ = __init__.__doc__.format(
        intro=OrbitModelBase.__init__.__doc__.split("Parameters")[0].rstrip(),
        params=OrbitModelBase.__init__.__doc__.split("----------")[1].lstrip(),
    )

    @u.quantity_input
    def get_params_init(self, pos: u.kpc, vel: u.km / u.s, label, label_params0=None):
        """
        Estimate initial model parameters from the data

        Parameters
        ----------
        pos : quantity-like or array-like
        vel : quantity-like or array-like
        label : array-like

        """
        import numpy as np
        import scipy.interpolate as sci

        x = pos.decompose(self.unit_sys).value
        v = vel.decompose(self.unit_sys).value

        # First, estimate nu0 with some crazy bullshit:
        med_stat = np.nanpercentile(label, 15)

        fac = 0.02  # MAGIC NUMBER
        for _ in range(16):  # max number of iterations
            annulus_idx = np.abs(label.ravel() - med_stat) < fac * np.abs(med_stat)
            if annulus_idx.sum() < 0.05 * len(annulus_idx):  # MAGIC NUMBER
                fac *= 2
            else:
                break

        else:
            raise ValueError("Shit!")

        vvv = np.abs(v.ravel()[annulus_idx])
        zzz = np.abs(x.ravel()[annulus_idx])
        v_z0 = np.median(vvv[zzz < 0.2 * np.median(zzz)])  # MAGIC NUMBER 0.2
        z_v0 = np.median(zzz[vvv < 0.2 * np.median(vvv)])  # MAGIC NUMBER 0.2
        nu = v_z0 / z_v0

        pars0 = {
            "pos0": np.nanmedian(x),
            "vel0": np.nanmedian(v),
            "ln_Omega0": np.log(nu),
        }
        r_e, _ = self.get_elliptical_coords(x, v, pars0)

        if label_params0 is not None:
            # Now estimate the label function spline values, again, with some craycray:
            bins = np.linspace(0, 1.0, 9) ** 2  # TODO: arbitrary bin max = 1
            stat = binned_statistic(
                r_e.ravel(), label.ravel(), bins=bins, statistic=np.nanmedian
            )
            xc = 0.5 * (stat.bin_edges[:-1] + stat.bin_edges[1:])

            # Fit the specified ln_dens_func to the measured densities
            # This is a BAG O' HACKS!
            spl = sci.InterpolatedUnivariateSpline(
                xc[np.isfinite(stat.statistic)],
                stat.statistic[np.isfinite(stat.statistic)],
                k=1,
            )
            xeval = np.geomspace(1e-3, np.nanmax(xc), 32)  # MAGIC NUMBER:

            def obj(params, x, data_y):
                model_y = self.label_func(x, **params)
                return jnp.sum((model_y - data_y) ** 2)

            optim = jaxopt.ScipyMinimize(fun=obj, method="BFGS")
            res = optim.run(init_params=label_params0, x=xeval, data_y=spl(xeval))
            if res.state.success:
                pars0["label_params"] = res.params
            else:
                warn(
                    "Initial parameter estimation failed: Failed to estimate "
                    "parameters of label function `label_func()`",
                    RuntimeWarning,
                )

        return pars0

    @partial(jax.jit, static_argnames=["self"])
    def get_label(self, r, params):
        return self.label_func(r, **params["label_params"])

    @partial(jax.jit, static_argnames=["self"])
    def label(self, pos, vel, params):
        r_e, th_e = self.get_elliptical_coords(pos, vel, params)
        r = self.get_r(r_e, th_e, params["e_params"])
        return self.get_label(r, params)

    @partial(jax.jit, static_argnames=["self"])
    def ln_label_likelihood(self, params, pos, vel, label, label_err):
        r_e, th_e = self.get_elliptical_coords(pos, vel, params)
        r = self.get_r(r_e, th_e, params["e_params"])
        model_label = self.get_label(r, params)

        # log of a gaussian
        return -0.5 * jnp.nansum((label - model_label) ** 2 / label_err**2)
