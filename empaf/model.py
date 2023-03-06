import abc
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
from scipy.stats import binned_statistic, binned_statistic_2d

from empaf.jax_helpers import simpson
from empaf.model_helpers import custom_tanh_func_alt

__all__ = ["DensityOrbitModel", "LabelOrbitModel"]


class OrbitModelBase:
    def __init__(self, e_funcs=None, unit_sys=galactic):
        r"""
        Notation:
        - :math:`\Omega_0` or ``Omega_0``: A scale frequency used to compute the
          elliptical radius ``rz_prime``. This can be interpreted as the asymptotic
          orbital frequency at :math:`J_z=0`.
        - :math:`r_z'` or ``rz_prime``: The "raw" elliptical radius :math:`\sqrt{z^2\,
          \Omega_0 + v_z^2 \, \Omega_0^{-1}}`.
        - :math:`\theta'` or ``theta_prime``: The "raw" z angle defined as :math:`\tan
          {\theta'} = \frac{z}{v_z}\,\Omega_0`.
        - :math:`r_z` or ``rz``: The distorted elliptical radius :math:`r_z = r_z' \,
          f(r_z', \theta_z')`, which is close to :math:`\sqrt{J_z}` and so we call it
          the "proxy action" below. :math:`f(\cdot)` is the distortion function defined
          next.
        - :math:`f(r_z', \theta_z')`: The distortion function is a Fourier expansion,
          defined as: :math:`f(r_z', \theta_z') = 1+\sum_m e_m(r_z')\,\cos(m\,\theta')`
        - :math:`\theta_z` or ``theta_z``: The vertical angle.

        Parameters
        ----------
        e_funcs : dict (optional)
            A dictionary that provides functions that specify the dependence of the
            Fourier distortion coefficients :math:`e_m(r_z')`. Keys should be the
            (integer) "m" order of the distortion term (for the distortion function),
            and values should be Python callable objects that can be passed to
            `jax.jit()`. The first argument of each of these functions should be the raw
            elliptical radius :math:`r_z'` or ``rz_prime``. If not specified, default
            monotonic functions will be used:
            `empaf.model_helpers.custom_tanh_func_alt()`.
        unit_sys : `gala.units.UnitSystem` (optional)
            The unit system to work in. Default is to use the "galactic" unit system
            from Gala: (kpc, Myr, Msun, radian).

        """
        if e_funcs is None:
            # Default functions:
            self.e_funcs = {
                2: lambda *a, **k: custom_tanh_func_alt(*a, f0=0.0, **k),
                4: lambda *a, **k: custom_tanh_func_alt(*a, f0=0.0, **k),
            }
            self._default_e_funcs = True
        else:
            self.e_funcs = {int(m): jax.jit(e_func) for m, e_func in e_funcs.items()}
            self._default_e_funcs = False

        # Unit system:
        self.unit_sys = UnitSystem(unit_sys)

        # TODO: decide if we will keep parameter validation. For JIT/JAX maybe not?
        # Fill a list of parameter names - used later to validate input `params`
        # self._param_names = ["ln_Omega", "e_params", "z0", "vz0"]

    def _get_default_e_params(self):
        pars0 = {}

        # If default e_funcs, we can specify some defaults:
        if self._default_e_funcs:
            pars0["e_params"] = {2: {}, 4: {}}
            pars0["e_params"][2]["f1"] = 0.1
            pars0["e_params"][2]["alpha"] = 0.33
            pars0["e_params"][2]["x0"] = 3.0

            pars0["e_params"][4]["f1"] = -0.02
            pars0["e_params"][4]["alpha"] = 0.45
            pars0["e_params"][4]["x0"] = 3.0
        else:
            warn(
                "With custom e_funcs, you must set your own initial parameters. Use "
                "the dictionary returned by this function and add initial guesses "
                "for all parameters expected by the e_funcs.",
                RuntimeWarning,
            )

        return pars0

    @partial(jax.jit, static_argnames=["self"])
    def z_vz_to_rz_theta_prime(self, z, vz, params):
        r"""
        Compute the raw elliptical radius :math:`r_z'` (``rz_prime``) and the apparent
        phase :math:`\theta_z'` (``theta_prime``)

        Parameters
        ----------
        z : numeric, array-like
        vz : numeric, array-like
        params : dict
        """
        x = (vz - params["vz0"]) / jnp.sqrt(jnp.exp(params["ln_Omega"]))
        y = (z - params["z0"]) * jnp.sqrt(jnp.exp(params["ln_Omega"]))

        rz_prime = jnp.sqrt(x**2 + y**2)
        th_prime = jnp.arctan2(y, x)

        return rz_prime, th_prime

    @partial(jax.jit, static_argnames=["self"])
    def get_es(self, rz_prime, e_params):
        """
        Compute the Fourier m-order distortion coefficients

        Parameters
        ----------
        rz_prime : numeric, array-like
        e_params : dict
        """
        es = {}
        for m, pars in e_params.items():
            es[m] = self.e_funcs[m](rz_prime, **pars)
        return es

    @partial(jax.jit, static_argnames=["self"])
    def get_rz(self, rz_prime, theta_prime, e_params):
        """
        Compute the "proxy action" or distorted radius :math:`r_z`

        Parameters
        ----------
        rz_prime : numeric, array-like
        theta_prime : numeric, array-like
        e_params : dict
        """
        es = self.get_es(rz_prime, e_params)
        return rz_prime * (
            1
            + jnp.sum(
                jnp.array([e * jnp.cos(m * theta_prime) for m, e in es.items()]), axis=0
            )
        )

    @partial(jax.jit, static_argnames=["self"])
    def get_thetaz(self, rz_prime, theta_prime, e_params):
        """
        Compute the vertical phase angle

        Parameters
        ----------
        rz_prime : numeric, array-like
        theta_prime : numeric, array-like
        e_params : dict
        """
        es = self.get_es(rz_prime, e_params)
        # TODO: why t.f. is the π/2 needed below??
        return theta_prime - jnp.sum(
            jnp.array(
                [m / (jnp.pi / 2) * e * jnp.sin(m * theta_prime) for m, e in es.items()]
            ),
            axis=0,
        )

    @partial(jax.jit, static_argnames=["self"])
    def get_rz_prime(self, rz, theta_prime, e_params):
        """
        Compute the raw radius :math:`r_z'` by inverting the distortion transformation

        Parameters
        ----------
        rz : numeric
            The "proxy action" or distorted radius.
        theta_prime : numeric
            The raw "apparent" angle.
        e_params : dict
            Dictionary of parameter values for the distortion coefficient (e) functions.
        """
        # TODO lots of numbers are hard-set below!
        bisec = Bisection(
            lambda x, rrz, tt_prime, ee_params: self.get_rz(x, tt_prime, ee_params)
            - rrz,
            lower=0.0,
            upper=1.0,
            maxiter=30,
            jit=True,
            unroll=True,
            check_bracket=False,
            tol=1e-4,
        )
        return bisec.run(rz, rrz=rz, tt_prime=theta_prime, ee_params=e_params).params

    @partial(jax.jit, static_argnames=["self"])
    def get_z(self, rz, theta_prime, params):
        rzp = self.get_rz_prime(rz, theta_prime, params["e_params"])
        return rzp * jnp.sin(theta_prime) / jnp.sqrt(jnp.exp(params["ln_Omega"]))

    @partial(jax.jit, static_argnames=["self"])
    def get_vz(self, rz, theta_prime, params):
        rzp = self.get_rz_prime(rz, theta_prime, params["e_params"])
        return rzp * jnp.cos(theta_prime) * jnp.sqrt(jnp.exp(params["ln_Omega"]))

    @partial(jax.jit, static_argnames=["self", "N_grid"])
    def _get_Tz_Jz_thetaz(self, z, vz, params, N_grid):
        rzp_, thp_ = self.z_vz_to_rz_theta_prime(z, vz, params)
        rz = self.get_rz(rzp_, thp_, params["e_params"])

        dz_dthp_func = jax.vmap(
            jax.grad(self.get_z, argnums=1), in_axes=[None, 0, None]
        )

        get_vz = jax.vmap(self.get_vz, in_axes=[None, 0, None])

        # Grid of theta_prime to do the integral over:
        thp_grid = jnp.linspace(0, jnp.pi / 2, N_grid)
        vz_th = get_vz(rz, thp_grid, params)
        dz_dthp = dz_dthp_func(rz, thp_grid, params)

        Tz = 4 * simpson(dz_dthp / vz_th, thp_grid)
        Jz = 4 / (2 * jnp.pi) * simpson(dz_dthp * vz_th, thp_grid)

        thp_partial = jnp.linspace(0, thp_, N_grid)
        vz_th_partial = get_vz(rz, thp_partial, params)
        dz_dthp_partial = dz_dthp_func(rz, thp_partial, params)
        dt = simpson(dz_dthp_partial / vz_th_partial, thp_partial)
        thz = 2 * jnp.pi * dt / Tz

        return Tz, Jz, thz

    _get_Tz_Jz_thetaz = jax.vmap(_get_Tz_Jz_thetaz, in_axes=[None, 0, 0, None, None])

    @u.quantity_input
    def compute_action_angle(self, z: u.kpc, vz: u.km / u.s, params, N_grid=101):
        """
        Compute the vertical period, action, and angle given input phase-space
        coordinates.

        Parameters
        ----------
        TODO
        """
        z = z.decompose(self.unit_sys).value
        vz = vz.decompose(self.unit_sys).value
        Tz, Jz, thz = self._get_Tz_Jz_thetaz(z, vz, params, N_grid)

        tbl = at.QTable()
        tbl["T_z"] = Tz * self.unit_sys["time"]
        tbl["Omega_z"] = 2 * jnp.pi * u.rad / tbl["T_z"]
        tbl["J_z"] = Jz * self.unit_sys["length"] ** 2 / self.unit_sys["time"]
        tbl["theta_z"] = thz * self.unit_sys["angle"]

        return tbl

    @abc.abstractmethod
    def objective(self):
        pass

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

    def check_e_funcs(self, e_params, rz_prime_max=1.0):
        """
        Check that the parameter values and functions used for the e_m(r_z') functions
        are valid given the condition that d(r_z)/d(r_z') > 0.
        """
        import numpy as np

        # TODO: 16 is a magic number
        rz_primes = np.linspace(np.sqrt(1e-3), np.sqrt(rz_prime_max), 16) ** 2

        # TODO: potential issue if order of arguments in e_funcs() call is different
        # from the order of the values in the e_params dictionary...
        d_em_d_rzp_funcs = {
            m: jax.vmap(
                jax.grad(self.e_funcs[m], argnums=0),
                in_axes=[0] + [None] * len(e_params[m]),
            )
            for m in self.e_funcs.keys()
        }

        thps = np.linspace(0, np.pi / 2, 128)
        checks = np.zeros((len(rz_primes), len(thps)))

        for j, thp in enumerate(thps):
            checks[:, j] = jnp.sum(
                jnp.array(
                    [
                        jnp.cos(m * thp)
                        * (
                            e_func(rz_primes, **e_params[m])
                            + rz_primes
                            * d_em_d_rzp_funcs[m](rz_primes, *e_params[m].values())
                        )
                        for m, e_func in self.e_funcs.items()
                    ]
                ),
                axis=0,
            )

        # This condition has to be met such that d(r_z)/d(r_z') > 0 at all theta_z':
        return np.all(checks > -1), checks


class DensityOrbitModel(OrbitModelBase):
    def __init__(self, ln_dens_func, e_funcs=None, unit_sys=galactic):
        """
        {intro}

        Parameters
        ----------
        ln_dens_func : callable (optional)
            TODO
        {params}
        """
        super().__init__(e_funcs=e_funcs, unit_sys=unit_sys)
        self.ln_dens_func = jax.jit(ln_dens_func)

    __init__.__doc__ = __init__.__doc__.format(
        intro=OrbitModelBase.__init__.__doc__.split("Parameters")[0].rstrip(),
        params=OrbitModelBase.__init__.__doc__.split("----------")[1].lstrip(),
    )

    @u.quantity_input
    def get_params_init(self, z: u.kpc, vz: u.km / u.s, ln_dens_params0=None):
        """
        Estimate initial model parameters from the data

        Parameters
        ----------
        z : quantity-like or array-like
        vz : quantity-like or array-like
        ln_dens_params0 : dict (optional)
        """
        import numpy as np

        z = z.decompose(self.unit_sys).value
        vz = vz.decompose(self.unit_sys).value

        std_z = 1.5 * MAD(z, ignore_nan=True)
        std_vz = 1.5 * MAD(vz, ignore_nan=True)
        nu = std_vz / std_z

        pars0 = {"z0": np.nanmedian(z), "vz0": np.nanmedian(vz), "ln_Omega": np.log(nu)}
        rzp, _ = self.z_vz_to_rz_theta_prime(z, vz, pars0)

        max_rz = np.nanpercentile(rzp, 99.5)
        rz_bins = np.linspace(0, max_rz, 25)  # TODO: fixed number
        dens_H, xe = np.histogram(rzp, bins=rz_bins)
        xc = 0.5 * (xe[:-1] + xe[1:])
        ln_dens = np.log(dens_H) - np.log(2 * np.pi * xc * (xe[1:] - xe[:-1]))

        # TODO: WTF - this is a total hack -- why is this needed???
        ln_dens = ln_dens - 8.6

        if ln_dens_params0 is not None:
            # Fit the specified ln_dens_func to the measured densities
            # This is a BAG O' HACKS!
            spl = sci.InterpolatedUnivariateSpline(xc, ln_dens, k=3)
            xeval = np.geomspace(1e-3, np.nanmax(xc), 32)  # MAGIC NUMBER:

            def obj(params, x, data_y):
                model_y = self.ln_dens_func(x, **params)
                return jnp.sum((model_y - data_y) ** 2)

            optim = jaxopt.ScipyMinimize(fun=obj, method="BFGS")
            res = optim.run(init_params=ln_dens_params0, x=xeval, data_y=spl(xeval))
            if res.state.success:
                pars0["ln_dens_params"] = res.params
            else:
                warn(
                    "Initial parameter estimation failed: Failed to estimate "
                    "parameters of log-density function `ln_dens_func()`",
                    RuntimeWarning,
                )

        # If default e_funcs, we can specify some defaults:
        pars0.update(self._get_default_e_params())

        return pars0

    @classmethod
    def get_data_im(cls, z, vz, bins):
        """
        Convert the raw data (stellar positions and velocities z, vz) into a binned 2D
        histogram / image of number counts.

        Parameters
        ----------
        z : array-like
        vz : array-like
        bins : dict
        """
        data_H, xe, ye = jnp.histogram2d(
            vz,
            z,
            bins=(bins["vz"], bins["z"]),
        )
        xc = 0.5 * (xe[:-1] + xe[1:])
        yc = 0.5 * (ye[:-1] + ye[1:])
        xc, yc = jnp.meshgrid(xc, yc)

        return {"z": jnp.array(yc), "vz": jnp.array(xc), "H": jnp.array(data_H.T)}

    @partial(jax.jit, static_argnames=["self"])
    def get_ln_dens(self, rz, params):
        return self.ln_dens_func(rz, **params["ln_dens_params"])

    @partial(jax.jit, static_argnames=["self"])
    def ln_density(self, z, vz, params):
        rzp, thp = self.z_vz_to_rz_theta_prime(z, vz, params)
        rz = self.get_rz(rzp, thp, params["e_params"])
        return self.get_ln_dens(rz, params)

    @partial(jax.jit, static_argnames=["self"])
    def ln_poisson_likelihood(self, params, z, vz, H):
        # Expected number:
        ln_Lambda = self.ln_density(z, vz, params)

        # gammaln(x+1) = log(factorial(x))
        return (H * ln_Lambda - jnp.exp(ln_Lambda) - gammaln(H + 1)).sum()

    @partial(jax.jit, static_argnames=["self"])
    def objective(self, params, z, vz, H):
        return -(self.ln_poisson_likelihood(params, z, vz, H)) / H.size


class LabelOrbitModel(OrbitModelBase):
    def __init__(self, label_func, e_funcs=None, unit_sys=galactic):
        """
        {intro}

        Parameters
        ----------
        label_func : callable (optional)
            TODO
        {params}
        """
        super().__init__(e_funcs=e_funcs, unit_sys=unit_sys)
        self.label_func = jax.jit(label_func)

    __init__.__doc__ = __init__.__doc__.format(
        intro=OrbitModelBase.__init__.__doc__.split("Parameters")[0].rstrip(),
        params=OrbitModelBase.__init__.__doc__.split("----------")[1].lstrip(),
    )

    @u.quantity_input
    def get_params_init(self, z: u.kpc, vz: u.km / u.s, label, label_params0=None):
        """
        Estimate initial model parameters from the data

        Parameters
        ----------
        z : quantity-like or array-like
        vz : quantity-like or array-like
        label : array-like

        """
        import numpy as np
        import scipy.interpolate as sci

        z = z.decompose(self.unit_sys).value
        vz = vz.decompose(self.unit_sys).value

        # First, estimate nu0 with some crazy bullshit:
        med_stat = np.nanpercentile(label, 25)

        fac = 0.02  # MAGIC NUMBER
        for _ in range(16):  # max number of iterations
            annulus_idx = np.abs(label.ravel() - med_stat) < fac * np.abs(med_stat)
            if annulus_idx.sum() < 0.05 * len(annulus_idx):  # MAGIC NUMBER
                fac *= 2
            else:
                break

        else:
            raise ValueError("Shit!")

        nu = np.nanpercentile(np.abs(vz.ravel()[annulus_idx]), 25) / np.nanpercentile(
            np.abs(z.ravel()[annulus_idx]), 25
        )

        pars0 = {"z0": np.nanmedian(z), "vz0": np.nanmedian(vz), "ln_Omega": np.log(nu)}
        rzp, _ = self.z_vz_to_rz_theta_prime(z, vz, pars0)

        if label_params0 is not None:
            # Now estimate the label function spline values, again, with some craycray:
            bins = np.linspace(0, 1.0, 9) ** 2  # TODO: arbitrary bin max = 1
            stat = binned_statistic(
                rzp.ravel(), label.ravel(), bins=bins, statistic=np.nanmedian
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

        # If default e_funcs, we can specify some defaults:
        pars0.update(self._get_default_e_params())

        return pars0

    @classmethod
    def get_data_im(cls, z, vz, label, bins, **binned_statistic_kwargs):
        import numpy as np

        stat = binned_statistic_2d(
            vz,
            z,
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
            vz,
            z,
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
    def get_label(self, rz, params):
        return self.label_func(rz, **params["label_params"])

    @partial(jax.jit, static_argnames=["self"])
    def label(self, z, vz, params):
        rzp, thp = self.z_vz_to_rz_theta_prime(z, vz, params)
        rz = self.get_rz(rzp, thp, params["e_params"])
        return self.get_label(rz, params)

    @partial(jax.jit, static_argnames=["self"])
    def ln_label_likelihood(self, params, z, vz, label, label_err):
        rzp, thp = self.z_vz_to_rz_theta_prime(z, vz, params)
        rz = self.get_rz(rzp, thp, params["e_params"])
        model_label = self.get_label(rz, params)

        # log of a gaussian
        return -0.5 * jnp.nansum((label - model_label) ** 2 / label_err**2)

    @partial(jax.jit, static_argnames=["self"])
    def objective(self, params, z, vz, label, label_err):
        return -(self.ln_label_likelihood(params, z, vz, label, label_err)) / label.size
