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
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
from jaxopt import Bisection
from scipy.stats import binned_statistic, binned_statistic_2d

from empaf.jax_helpers import simpson

__all__ = ["DensityOrbitModel", "LabelOrbitModel"]


class OrbitModelBase:
    _state_names = ["nu", "e_vals", "z0", "vz0"]
    _spl_name = ""

    def __init_subclass__(cls):
        if cls._spl_name == "":
            raise ValueError("TODO")
        cls._state_names = cls._state_names + [f"{cls._spl_name}_vals"]  # make a copy

    def __init__(self, e_knots, e_signs, e_k=1, dens_k=1, unit_sys=galactic):
        r"""
        TODO:
        - nu below should actually be Omega
        -

        Notation:
        - :math:`\nu_0` or ``nu_0``: A scale frequency used to compute the elliptical
          radius ``rz_prime``.
        - :math:`r_z'` or ``rz_prime``: The "raw" elliptical radius :math:`\sqrt{z^2\,
          \nu_0 + v_z^2 \, \nu_0^{-1}}`.
        - :math:`\theta'` or ``theta_prime``: The "raw" z angle defined as :math:`\tan
          {\theta'} = \frac{z}{v_z}\,\nu_0`.
        - :math:`r_z` or ``rz``: The distorted elliptical radius :math:`r_z = r_z' \,
          f(r_z', \theta_z')` where :math:`f(\cdot)` is the distortion function.
        - :math:`\theta` or ``theta``: The true vertical angle.
        - :math:`f(r_z', \theta_z')`: The distortion function is a Fourier expansion,
          defined as: :math:`f(r_z', \theta_z') = 1+\sum_m e_m(r_z')\,\cos(m\,\theta')`

        Parameters
        ----------
        e_knots : dict
            The locations of knots for the splines that control the variation of each
            :math:`e_m(r_z')` function. Keys should be the (integer) "m" order of the
            distortion term (for the distortion function), and values should be the knot
            locations for interpolating the values of the distortion coefficients
            :math:`e_m(r_z')`.
        e_signs : dict
            The overall sign of the :math:`e_m(r_z')` functions. Keys should be the
            (integer) "m" order of the distortion term (for the distortion function),
            and values should be 1 or -1.
        e_k : int (optional)
            The order of the spline used for the :math:`e_m` coefficients. Default: 1,
            linear. Only k=2 (quadratic) or k=3 (cubic) are supported.
        dens_k : int (optional)
            The order of the spline used for the density function. Default: 1, linear.
            Only k=2 (quadratic) or k=3 (cubic) are supported.
        unit_sys : `gala.units.UnitSystem` (optional)
            The unit system to work in. Default is to use the "galactic" unit system
            from Gala: (kpc, Myr, Msun).

        """
        self.e_knots = {int(m): jnp.array(knots) for m, knots in e_knots.items()}
        self.e_signs = {int(m): float(e_signs.get(m, 1.0)) for m in e_knots.keys()}
        self.e_k = int(e_k)
        self.dens_k = int(dens_k)

        for m, knots in self.e_knots.items():
            if knots[0] != 0.0:
                raise ValueError(
                    f"The first knot must be at rz=0. Knots for m={m} start at "
                    f"{knots[0]} instead"
                )

        self.state = None

        # Unit system:
        self.unit_sys = UnitSystem(unit_sys)

    def _validate_state(self, names=None):
        if self.state is None or not hasattr(self.state, "keys"):
            raise RuntimeError(
                "Model state is not set or is invalid. Maybe you didn't initialize "
                "properly, or run .optimize() yet?"
            )

        else:
            if names is None:
                names = self._state_names

            for name in names:
                assert name in self._state_names
                if name not in self.state:
                    raise RuntimeError(
                        f"Parameter {name} is missing from the model state"
                    )

    def set_state(self, params, overwrite=False):
        """
        Set the model state parameters.

        Default behavior is to not overwrite any existing state parameter values.

        Parameters
        ----------
        params : dict
            Parameters used to set the model state parameters.
        overwrite : bool (optional)
            Overwrite any existing state parameter values.
        """
        if params is None:
            self.state = None
            return

        # TODO: could have a "copy" argument?
        if self.state is None or overwrite:
            self.state = {}

        for k in params:
            if k == "ln_nu":
                self.state.setdefault("nu", jnp.exp(params["ln_nu"]))
            elif k == "e_vals":
                self.state.setdefault(k, {})
                for m, vals in params[k].items():
                    self.state[k].setdefault(m, jnp.array(vals))
            else:
                self.state.setdefault(k, jnp.array(params[k]))

    def get_params(self):
        """
        Transform the current model state to optimization parameter values
        """
        self._validate_state()

        params = {}
        for k in self.state:
            if k == "nu":
                params["ln_nu"] = jnp.log(self.state[k])
            elif k == "e_vals":
                params[k] = {}
                for m, vals in self.state[k].items():
                    params[k][m] = jnp.array(vals)
            else:
                params[k] = jnp.array(self.state[k])

        return params

    @partial(jax.jit, static_argnames=["self"])
    def get_es(self, rz_prime):
        """
        Compute the Fourier m-order coefficients
        """
        self._validate_state(["e_vals"])
        e_vals = self.state["e_vals"]

        es = {}
        for m, vals in e_vals.items():
            tmp_e = self.e_signs[m] * jnp.cumsum(
                jnp.concatenate((jnp.array([0.0]), jnp.array(vals)))
            )
            es[m] = InterpolatedUnivariateSpline(self.e_knots[m], tmp_e, k=self.e_k)(
                rz_prime
            )
        return es

    @partial(jax.jit, static_argnames=["self"])
    def z_vz_to_rz_theta_prime(self, z, vz):
        r"""
        Compute :math:`r_z'` (``rz_prime``) and :math:`\theta_z'` (``theta_prime``)
        """
        self._validate_state(["z0", "vz0", "nu"])

        x = (vz - self.state["vz0"]) / jnp.sqrt(self.state["nu"])
        y = (z - self.state["z0"]) * jnp.sqrt(self.state["nu"])

        rz_prime = jnp.sqrt(x**2 + y**2)
        th_prime = jnp.arctan2(y, x)

        return rz_prime, th_prime

    z_vz_to_rz_theta_prime_arr = jax.vmap(z_vz_to_rz_theta_prime, in_axes=[None, 0, 0])

    @partial(jax.jit, static_argnames=["self"])
    def get_rz(self, rz_prime, theta_prime):
        """
        Compute the distorted radius :math:`r_z`
        """
        es = self.get_es(rz_prime)
        return rz_prime * (
            1
            + jnp.sum(
                jnp.array([e * jnp.cos(n * theta_prime) for n, e in es.items()]), axis=0
            )
        )

    @partial(jax.jit, static_argnames=["self"])
    def get_rz_prime(self, rz, theta_prime):
        """
        Compute the raw radius :math:`r_z'` by inverting the distortion transformation
        """
        bisec = Bisection(
            lambda x, rrz, tt_prime: self.get_rz(x, tt_prime) - rrz,
            lower=0.0,
            upper=1.0,
            maxiter=30,
            jit=True,
            unroll=True,
            check_bracket=False,
            tol=1e-4,
        )
        return bisec.run(rz, rrz=rz, tt_prime=theta_prime).params

        # The below only works for purely linear functions in e2, e4, etc., and only
        # when they are fixed to be zero at rz=0
        # self._validate_state()
        # e_vals = self.state["e_vals"]

        # # Shorthand
        # thp = theta_prime

        # # convert e_vals and e_knots to slope and intercept
        # e_as = {k: e_vals[k][0] / self.e_knots[k][1] for k in e_vals}
        # terms2 = jnp.sum(jnp.array([e_as[k] * jnp.cos(k * thp) for k in e_as]),
        # axis=0)
        # return (2 * rz) / (1 + jnp.sqrt(1 + 4 * rz * terms2))

    @partial(jax.jit, static_argnames=["self"])
    def get_z(self, rz, theta_prime):
        self._validate_state()
        nu = self.state["nu"]
        rzp = self.get_rz_prime(rz, theta_prime)
        return rzp * jnp.sin(theta_prime) / jnp.sqrt(nu)

    @partial(jax.jit, static_argnames=["self"])
    def get_vz(self, rz, theta_prime):
        self._validate_state()
        nu = self.state["nu"]
        rzp = self.get_rz_prime(rz, theta_prime)
        return rzp * jnp.cos(theta_prime) * jnp.sqrt(nu)

    @partial(jax.jit, static_argnames=["self", "N_grid"])
    def _get_Tz_Jz_thetaz(self, z, vz, N_grid):
        rzp_, thp_ = self.z_vz_to_rz_theta_prime(z, vz)
        rz = self.get_rz(rzp_, thp_)

        dz_dthp_func = jax.vmap(jax.grad(self.get_z, argnums=1), in_axes=[None, 0])

        get_vz = jax.vmap(self.get_vz, in_axes=[None, 0])

        # Grid of theta_prime to do the integral over:
        thp_grid = jnp.linspace(0, jnp.pi / 2, N_grid)
        vz_th = get_vz(rz, thp_grid)
        dz_dthp = dz_dthp_func(rz, thp_grid)

        Tz = 4 * simpson(dz_dthp / vz_th, thp_grid)
        Jz = 4 / (2 * jnp.pi) * simpson(dz_dthp * vz_th, thp_grid)

        thp_partial = jnp.linspace(0, thp_, N_grid)
        vz_th_partial = get_vz(rz, thp_partial)
        dz_dthp_partial = dz_dthp_func(rz, thp_partial)
        dt = simpson(dz_dthp_partial / vz_th_partial, thp_partial)
        thz = 2 * jnp.pi * dt / Tz

        return Tz, Jz, thz

    _get_Tz_Jz_thetaz = jax.vmap(_get_Tz_Jz_thetaz, in_axes=[None, 0, 0, None])

    @u.quantity_input
    def get_aaf(self, z: u.kpc, vz: u.km / u.s, N_grid=101):
        """
        Compute the vertical period, action, and angle given input phase-space
        coordinates.
        """
        z = z.decompose(self.unit_sys).value
        vz = vz.decompose(self.unit_sys).value
        Tz, Jz, thz = self._get_Tz_Jz_thetaz(z, vz, N_grid)

        tbl = at.QTable()
        tbl["T_z"] = Tz * self.unit_sys["time"]
        tbl["Omega_z"] = 2 * jnp.pi * u.rad / tbl["T_z"]
        tbl["J_z"] = Jz * self.unit_sys["length"] ** 2 / self.unit_sys["time"]
        tbl["theta_z"] = thz * self.unit_sys["angle"]

        return tbl

    @abc.abstractmethod
    def objective(self):
        pass

    def optimize(self, params0=None, bounds=None, jaxopt_kwargs=None, **data):
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
        else:
            self.set_state(res.params, overwrite=True)

        return res

    def copy(self):
        knot_name = f"{self._spl_name}_knots"
        kw = {knot_name: getattr(self, knot_name)}
        obj = self.__class__(
            e_knots=self.e_knots, e_signs=self.e_signs, unit_sys=self.unit_sys, **kw
        )
        obj.set_state(self.state, overwrite=True)
        return obj


class DensityOrbitModel(OrbitModelBase):
    _spl_name = "ln_dens"

    def __init__(self, ln_dens_knots, e_knots, e_signs, unit_sys=galactic):
        f"""
        {OrbitModelBase.__init__.__doc__.split('Parameters')[0]}

        Parameters
        ----------
        ln_dens_knots : array_like
            The knot locations for the spline that controls the density function. These
            are locations in :math:`r_z`.
        {OrbitModelBase.__init__.__doc__.split('----------')[1]}

        """
        self.ln_dens_knots = jnp.array(ln_dens_knots)
        super().__init__(e_knots=e_knots, e_signs=e_signs, unit_sys=unit_sys)

    @u.quantity_input
    def get_params_init(self, z: u.kpc, vz: u.km / u.s):
        """
        Estimate initial model parameters from the data

        Parameters
        ----------
        z : quantity-like or array-like
        vz : quantity-like or array-like

        Returns
        -------
        model : `VerticalOrbitModel`
            A copy of the initial model with state set to initial parameter estimates.
        """
        import numpy as np

        z = z.decompose(self.unit_sys).value
        vz = vz.decompose(self.unit_sys).value

        std_z = 1.5 * MAD(z, ignore_nan=True)
        std_vz = 1.5 * MAD(vz, ignore_nan=True)
        nu = std_vz / std_z

        model = self.copy()

        model.set_state({"z0": np.nanmedian(z), "vz0": np.nanmedian(vz), "nu": nu})
        rzp, _ = model.z_vz_to_rz_theta_prime_arr(z, vz)

        max_rz = np.nanpercentile(rzp, 99.5)
        rz_bins = np.linspace(0, max_rz, 25)  # TODO: fixed number
        dens_H, xe = np.histogram(rzp, bins=rz_bins)
        xc = 0.5 * (xe[:-1] + xe[1:])
        ln_dens = np.log(dens_H) - np.log(2 * np.pi * xc * (xe[1:] - xe[:-1]))

        # TODO: WTF - this is a total hack -- why is this needed???
        ln_dens = ln_dens - 8.6

        spl = sci.InterpolatedUnivariateSpline(xc, ln_dens, k=self.dens_k)
        ln_dens_vals = spl(model.ln_dens_knots)

        model.set_state({"ln_dens_vals": ln_dens_vals})

        # TODO: is there a better way to estimate these?
        e_vals = {}
        e_vals[2] = jnp.full(len(self.e_knots[2]) - 1, 0.1 / len(self.e_knots[2]))
        e_vals[4] = jnp.full(len(self.e_knots[4]) - 1, 0.05 / len(self.e_knots[4]))
        for m in self.e_knots.keys():
            if m in e_vals:
                continue
            e_vals[m] = jnp.zeros(len(self.e_knots[m]) - 1)
        model.set_state({"e_vals": e_vals})

        model._validate_state()

        return model

    @u.quantity_input
    def get_data_im(self, z: u.kpc, vz: u.km / u.s, bins):
        """
        Convert the raw data (stellar positions and velocities z, vz) into a binned 2D
        histogram / image of number counts.

        Parameters
        ----------
        z : quantity-like
        vz : quantity-like
        bins : dict
        """
        data_H, xe, ye = jnp.histogram2d(
            vz.decompose(self.unit_sys).value,
            z.decompose(self.unit_sys).value,
            bins=(bins["vz"], bins["z"]),
        )
        xc = 0.5 * (xe[:-1] + xe[1:])
        yc = 0.5 * (ye[:-1] + ye[1:])
        xc, yc = jnp.meshgrid(xc, yc)

        return {"z": jnp.array(yc), "vz": jnp.array(xc), "H": jnp.array(data_H.T)}

    @partial(jax.jit, static_argnames=["self"])
    def get_ln_dens(self, rz):
        self._validate_state()
        ln_dens_vals = self.state["ln_dens_vals"]
        spl = InterpolatedUnivariateSpline(
            self.ln_dens_knots, ln_dens_vals, k=self.dens_k
        )
        return spl(rz)

    @partial(jax.jit, static_argnames=["self"])
    def ln_density(self, z, vz):
        self._validate_state()
        rzp, thp = self.z_vz_to_rz_theta_prime(z, vz)
        rz = self.get_rz(rzp, thp)
        return self.get_ln_dens(rz)

    @partial(jax.jit, static_argnames=["self"])
    def ln_poisson_likelihood(self, params, z, vz, H):
        self.set_state(params, overwrite=True)

        # Expected number:
        ln_Lambda = self.ln_density(z, vz)

        # gammaln(x+1) = log(factorial(x))
        return (H * ln_Lambda - jnp.exp(ln_Lambda) - gammaln(H + 1)).sum()

    @partial(jax.jit, static_argnames=["self"])
    def objective(self, params, z, vz, H):
        return -(self.ln_poisson_likelihood(params, z, vz, H)) / H.size


class LabelOrbitModel(OrbitModelBase):
    _spl_name = "label"

    def __init__(
        self, label_knots, e_knots, e_signs, e_k=1, label_k=3, unit_sys=galactic
    ):
        f"""
        {OrbitModelBase.__init__.__doc__.split('Parameters')[0]}

        Parameters
        ----------
        label_knots : array_like
            The knot locations for the spline that controls the label function. These
            are locations in :math:`r_z`.
        {OrbitModelBase.__init__.__doc__.split('----------')[1]}

        """
        self.label_knots = jnp.array(label_knots)
        self.label_k = int(label_k)
        super().__init__(e_knots=e_knots, e_signs=e_signs, e_k=e_k, unit_sys=unit_sys)

    @u.quantity_input
    def get_params_init(self, z: u.kpc, vz: u.km / u.s, label_stat):
        import numpy as np
        import scipy.interpolate as sci

        z = z.decompose(self.unit_sys).value
        vz = vz.decompose(self.unit_sys).value

        model = self.copy()

        # First, estimate nu0 with some crazy bullshit:
        med_stat = np.nanpercentile(label_stat, 15)
        annulus_idx = np.abs(label_stat.ravel() - med_stat) < 0.02 * med_stat
        nu = np.nanpercentile(np.abs(vz.ravel()[annulus_idx]), 25) / np.nanpercentile(
            np.abs(z.ravel()[annulus_idx]), 25
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
        e_vals[2] = jnp.full(len(self.e_knots[2]) - 1, 0.1 / len(self.e_knots[2]))
        e_vals[4] = jnp.full(len(self.e_knots[4]) - 1, 0.05 / len(self.e_knots[4]))
        for m in self.e_knots.keys():
            if m in e_vals:
                continue
            e_vals[m] = jnp.zeros(len(self.e_knots[m]) - 1)
        model.set_state({"e_vals": e_vals})

        model._validate_state()

        return model

    @u.quantity_input
    def get_data_im(
        self, z: u.kpc, vz: u.km / u.s, label, bins, **binned_statistic_kwargs
    ):
        import numpy as np

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
        spl = InterpolatedUnivariateSpline(self.label_knots, label_vals, k=self.label_k)
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
