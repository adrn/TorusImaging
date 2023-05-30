import jax
import jax.numpy as jnp

__all__ = ["simpson"]


@jax.jit
def simpson(y, x):
    """
    Evaluate the definite integral of a function using Simpson's rule

    Note: x must be a regularly-spaced grid of points!
    """

    num_points = len(x)
    if num_points % 2 == 0:
        n_odd = num_points - 1
    else:
        n_odd = num_points

    dx = jnp.diff(x)[0]
    weights_first = jnp.asarray([1.0])
    weights_mid = jnp.tile(jnp.asarray([4.0, 2.0]), [(n_odd - 3) // 2])
    weights_last = jnp.asarray([4.0, 1.0])
    weights = jnp.concatenate([weights_first, weights_mid, weights_last], axis=0)
    integral = dx / 3 * jnp.sum(y[:n_odd] * weights, axis=-1)

    if n_odd == num_points:  # odd
        return integral

    else:  # even
        return integral + 0.5 * dx * (y[-1] + y[-2])
