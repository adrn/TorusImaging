import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

__all__ = ["simpson"]


@jax.jit
def simpson(y, x):
    """
    Evaluate the definite integral of a function using Simpson's rule

    Note: x must be a regularly-spaced grid of points!
    """

    num_points = len(x)
    n_odd = num_points - 1 if num_points % 2 == 0 else num_points

    dx = jnp.diff(x)[0]
    weights_first = jnp.asarray([1.0])
    weights_mid = jnp.tile(jnp.asarray([4.0, 2.0]), [(n_odd - 3) // 2])
    weights_last = jnp.asarray([4.0, 1.0])
    weights = jnp.concatenate([weights_first, weights_mid, weights_last], axis=0)
    integral = dx / 3 * jnp.sum(y[:n_odd] * weights, axis=-1)

    if n_odd == num_points:  # odd
        return integral

    return integral + 0.5 * dx * (y[-1] + y[-2])


@jax.jit
def ln_simpson(ln_y, x):
    """
    Evaluate the log of the definite integral of a function using Simpson's rule

    Note: x must be a regularly-spaced grid of points!
    """

    num_points = len(x)
    n_odd = num_points - 1 if num_points % 2 == 0 else num_points

    dx = jnp.diff(x)[0]
    weights_first = jnp.asarray([1.0])
    weights_mid = jnp.tile(jnp.asarray([4.0, 2.0]), [(n_odd - 3) // 2])
    weights_last = jnp.asarray([4.0, 1.0])
    weights = jnp.concatenate([weights_first, weights_mid, weights_last], axis=0)
    ln_integral = logsumexp(ln_y[:n_odd] + jnp.log(weights), axis=-1) + jnp.log(dx / 3)

    if n_odd == num_points:  # odd
        return ln_integral

    return jnp.logaddexp(
        ln_integral, jnp.log(0.5 * dx) + jnp.logaddexp(ln_y[-1], ln_y[-2])
    )
