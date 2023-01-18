from functools import partial
from warnings import warn

import jax
import jax.numpy as jnp
import jaxopt
import scipy.interpolate as sci
from astropy.stats import median_absolute_deviation as MAD
from gala.units import UnitSystem, galactic
from jax.scipy.special import gammaln
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline

from empaf.jax_helpers import simpson

__all__ = ["VerticalOrbitModel"]


class VerticalOrbitModel:
    _state_names = ["Omega", "e_vals", "ln_dens_vals", "z0", "vz0"]

    def __init__(self, dens_knots, e_knots, unit_sys=galactic):
        r"""
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
        dens_knots : array_like
            The knot locations for the spline that controls the density function. These
            are locations in :math:`r_z`.
        e_knots : dict
            Keys should be the (integer) "m" order of the distortion term (for the
            distortion function), and values should be the knot locations for
            interpolating the values of the distortion coefficients :math:`e_m(r_z')`.
            Currently, this functionality has been partially disabled and the functions
            are required to be linear, so you must pass in two knots.

        """
        self.dens_knots = jnp.array(dens_knots)
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

        if hasattr(z, "unit"):
            z = z.decompose(self.unit_sys).value
        if hasattr(vz, "unit"):
            vz = vz.decompose(self.unit_sys).value

        std_z = 1.5 * MAD(z, ignore_nan=True)
        std_vz = 1.5 * MAD(vz, ignore_nan=True)
        Omega = std_vz / std_z

        model = self.copy()

        model.set_state(
            {"z0": np.nanmedian(z), "vz0": np.nanmedian(vz), "Omega": Omega}
        )
        rzp, _ = model.z_vz_to_rz_theta_prime_arr(z, vz)

        max_rz = np.nanpercentile(rzp, 99.5)
        rz_bins = np.linspace(0, max_rz, 25)  # TODO: fixed number
        dens_H, xe = np.histogram(rzp, bins=rz_bins)
        xc = 0.5 * (xe[:-1] + xe[1:])
        ln_dens = np.log(dens_H) - np.log(2 * np.pi * xc * (xe[1:] - xe[:-1]))

        # TODO: WTF - this is a total hack -- why is this needed???
        ln_dens = ln_dens - 8.6

        spl = sci.InterpolatedUnivariateSpline(xc, ln_dens, k=1)
        ln_dens_vals = spl(model.dens_knots)

        model.set_state({"ln_dens_vals": ln_dens_vals})

        # TODO: is there a better way to estimate these?
        e_vals = {}
        e_vals[2] = jnp.array([0.0, 0.1])
        e_vals[4] = jnp.array([0.0, -0.05])
        for m in self.e_knots.keys():
            if m in e_vals:
                continue
            e_vals[m] = jnp.zeros(2)
        model.set_state({"e_vals": e_vals})

        model._validate_state()

        return model

    def get_data_im(self, z, vz, bins):
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
            if k == "ln_Omega":
                self.state.setdefault("Omega", jnp.exp(params["ln_Omega"]))
            else:
                self.state.setdefault(k, params[k])

    def get_params(self):
        """
        Transform the current model state to optimization parameter values
        """
        self._validate_state()
        params = self.state.copy()
        params["ln_Omega"] = jnp.log(params.pop("Omega"))
        return params

    @partial(jax.jit, static_argnames=["self"])
    def get_es(self, rz_prime):
        """
        Compute the Fourier m-order coefficients
        """
        self._validate_state(["e_vals"])
        e_vals = self.state["e_vals"]

        es = {}
        for k, vals in e_vals.items():
            es[k] = InterpolatedUnivariateSpline(self.e_knots[k], vals, k=1)(rz_prime)
        return es

    @partial(jax.jit, static_argnames=["self"])
    def z_vz_to_rz_theta_prime(self, z, vz):
        r"""
        Compute :math:`r_z'` (``rz_prime``) and :math:`\theta_z'` (``theta_prime``)
        """
        self._validate_state(["z0", "vz0", "Omega"])

        x = (vz - self.state["vz0"]) / jnp.sqrt(self.state["Omega"])
        y = (z - self.state["z0"]) * jnp.sqrt(self.state["Omega"])

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
        self._validate_state()
        e_vals = self.state["e_vals"]

        # Shorthand
        thp = theta_prime

        # convert e_vals and e_knots to slope and intercept
        e_as = {
            k: (e_vals[k][1] - e_vals[k][0]) / (self.e_knots[k][1] - self.e_knots[k][0])
            for k in e_vals
        }
        e_bs = {k: -e_as[k] * self.e_knots[k][0] + e_vals[k][0] for k in e_vals}

        terms1 = jnp.sum(jnp.array([e_bs[k] * jnp.cos(k * thp) for k in e_bs]), axis=0)
        terms2 = jnp.sum(jnp.array([e_as[k] * jnp.cos(k * thp) for k in e_bs]), axis=0)
        return (2 * rz) / (1 + terms1 + jnp.sqrt((1 + terms1) ** 2 + 4 * rz * terms2))

    @partial(jax.jit, static_argnames=["self"])
    def get_z(self, rz, theta_prime):
        self._validate_state()
        Omega = self.state["Omega"]
        rzp = self.get_rz_prime(rz, theta_prime)
        return rzp * jnp.sin(theta_prime) / jnp.sqrt(Omega)

    @partial(jax.jit, static_argnames=["self"])
    def get_vz(self, rz, theta_prime):
        self._validate_state()
        Omega = self.state["Omega"]
        rzp = self.get_rz_prime(rz, theta_prime)
        return rzp * jnp.cos(theta_prime) * jnp.sqrt(Omega)

    @partial(jax.jit, static_argnames=["self", "N_grid"])
    def get_Tz_Jz_thetaz(self, z, vz, N_grid):
        """
        Compute the vertical period, action, and angle given input phase-space
        coordinates.
        """
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

    get_Tz_Jz_thetaz = jax.vmap(get_Tz_Jz_thetaz, in_axes=[None, 0, 0, None])

    @partial(jax.jit, static_argnames=["self"])
    def get_ln_dens(self, rz):
        self._validate_state()
        ln_dens_vals = self.state["ln_dens_vals"]
        spl = InterpolatedUnivariateSpline(self.dens_knots, ln_dens_vals, k=3)
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

    def optimize(self, z, vz, H, params0=None, bounds=None, jaxopt_kwargs=None):
        """
        Parameters
        ----------
        z : array-like
        vz : array-like
        H : array-like
        params0 : dict (optional)
        bounds : tuple of dict (optional)
        jaxopt_kwargs : dict (optional)

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
            res = optimizer.run(
                init_params=params0,
                bounds=bounds,
                z=z,
                vz=vz,
                H=H,
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
        obj = self.__class__(dens_knots=self.dens_knots, e_knots=self.e_knots)
        obj.set_state(self.state)
        return obj
