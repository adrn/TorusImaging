import jax
import jax.numpy as jnp

__all__ = ["simpson"]


@jax.jit
def simpson(y, x):
    """
    Evaluate the definite integral of a function using Simpson's rule

    Note: x must be a regularly-spaced grid of points!
    """

    dx = jnp.diff(x)[0]
    num_points = len(x)
    if num_points % 2 == 0:
        raise ValueError("Because of laziness, the input size must be odd")

    weights_first = jnp.asarray([1.0])
    weights_mid = jnp.tile(jnp.asarray([4.0, 2.0]), [(num_points - 3) // 2])
    weights_last = jnp.asarray([4.0, 1.0])
    weights = jnp.concatenate([weights_first, weights_mid, weights_last], axis=0)

    return dx / 3 * jnp.sum(y * weights, axis=-1)
