from functools import partial
from typing import Literal, Optional
from warnings import warn

import astropy.units as u
import jax
import jax.numpy as jnp
import jaxopt
import numpy.typing as npt
from gala.units import UnitSystem, galactic
from jax.scipy.special import gammaln
from jaxopt.base import OptStep

__all__ = ["TorusImaging1DEnergy"]


class TorusImaging1DEnergy:
    def __init__(self, label_func, pot_func, units=galactic):
        self.label_func = jax.jit(label_func)
        self.pot_func = jax.jit(pot_func)

        # Unit system:
        self.units = UnitSystem(units)

        regularization_func = None
        if regularization_func is None:
            regularization_func = lambda *_, **__: 0.0  # noqa
        self.regularization_func = regularization_func

    @partial(jax.jit, static_argnames=["self"])
    def _get_E(self, pos, vel, params):
        return 0.5 * vel**2 + self.pot_func(pos, **params["pot_params"])

    @partial(jax.jit, static_argnames=["self"])
    def _get_label(self, pos, vel, params):
        E = self._get_E(pos, vel, params)
        return self.label_func(E, **params["label_params"])

    @partial(jax.jit, static_argnames=["self"])
    def _get_acc(self, pos, params):
        return jax.grad(self.pot_func)(pos, **params["pot_params"])

    _get_dacc_dpos = jax.grad(_get_acc, argnums=1)
    _get_dacc_dpos_vmap = jax.vmap(_get_dacc_dpos, in_axes=(None, 0, None))

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
        x = jnp.atleast_1d(pos.decompose(self.units).value)
        in_shape = x.shape
        x = x.ravel()

        get_acc = jax.vmap(self._get_acc, in_axes=[0, None])
        res = get_acc(x, params)
        return res.reshape(in_shape) * self.units["acceleration"]

    @u.quantity_input
    def get_acceleration_deriv(self, pos: u.kpc, params):
        """
        Compute the derivative of the acceleration with respect to position as a
        function of position in the limit as velocity goes to zero

        Parameters
        ----------
        pos : `astropy.units.Quantity`
        params : dict
        """
        x = jnp.atleast_1d(pos.decompose(self.units).value)
        in_shape = x.shape
        x = x.ravel()

        res = self._get_dacc_dpos_vmap(x, params)
        return res.reshape(in_shape) * self.units["acceleration"] / self.units["length"]

    @u.quantity_input
    def get_label(self, pos: u.kpc, vel: u.km / u.s, params):
        x = pos.decompose(self.units).value
        v = vel.decompose(self.units).value
        return self._get_label(x.ravel(), v.ravel(), params).reshape(x.shape)

    @partial(jax.jit, static_argnames=["self"])
    def ln_poisson_likelihood(self, params, pos, vel, counts):
        # Expected number:
        ln_Lambda = self._get_label(pos, vel, params)

        # gammaln(x+1) = log(factorial(x))
        return (counts * ln_Lambda - jnp.exp(ln_Lambda) - gammaln(counts + 1)).sum()

    @partial(jax.jit, static_argnames=["self"])
    def ln_gaussian_likelihood(self, params, pos, vel, label, label_err):
        model_label = self._get_label(pos, vel, params)
        return -0.5 * jnp.nansum((label - model_label) ** 2 / label_err**2)

    @partial(jax.jit, static_argnames=["self"])
    def objective_poisson(self, params, pos, vel, counts):
        f_val = self.ln_poisson_likelihood(params, pos, vel, counts)
        return -(f_val - self.regularization_func(self, params)) / pos.size

    @partial(jax.jit, static_argnames=["self"])
    def objective_gaussian(self, params, pos, vel, label, label_err):
        f_val = self.ln_gaussian_likelihood(params, pos, vel, label, label_err)
        return -(f_val - self.regularization_func(self, params)) / pos.size

    def optimize(
        self,
        params0: dict,
        objective: Literal["poisson", "gaussian"],
        bounds: Optional[tuple[dict]] = None,
        jaxopt_kwargs: Optional[dict] = None,
        **data: npt.ArrayLike,
    ) -> OptStep:
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

    def get_crlb(
        self,
        params: dict[str, dict | npt.ArrayLike],
        data: dict[str, npt.ArrayLike],
        objective: str = "gaussian",
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
            ll = getattr(self, f"ln_{objective}_likelihood")(params, **data)
            return -(ll - self.regularization_func(self, params))

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
        objective: str = "gaussian",
    ) -> dict[str, dict | npt.ArrayLike]:
        """
        Compute the uncertainties on the parameters using the diagonal of the Cramer-Rao
        lower bound matrix (see :meth:`get_crlb`).
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
        params: dict[str, dict | npt.ArrayLike],
        data: dict[str, npt.ArrayLike],
        objective: str = "gaussian",
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
        else:
            return jax.tree_util.tree_unflatten(treedef, arrs)

    def mcmc_run_label(
        self,
        binned_data,
        p0,
        bounds,
        rng_seed=0,
        num_steps=1000,
        num_warmup=1000,
    ):
        """
        EXPERIMENTAL

        Currently only supports uniform priors on all parameters, specified by the input
        bounds.
        """
        import blackjax
        import numpy as np

        # First check that objective evaluates to a finite value:
        mask = (
            np.isfinite(binned_data["label"])
            & np.isfinite(binned_data["label_err"])
            & (binned_data["label_err"] > 0)
        )
        data = dict(
            pos=binned_data["pos"].decompose(self.units).value[mask],
            vel=binned_data["vel"].decompose(self.units).value[mask],
            label=binned_data["label"][mask],
            label_err=binned_data["label_err"][mask],
        )
        test_val = self.objective_gaussian(p0, **data)
        if not np.isfinite(test_val):
            raise RuntimeError("Objective function evaluated to non-finite value")

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

        kernel = blackjax.nuts(logprob, **parameters).step
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


def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states
