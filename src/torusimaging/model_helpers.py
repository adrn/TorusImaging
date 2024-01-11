import jax
import jax.numpy as jnp

__all__ = [
    "monotonic_poly_func",
    "monotonic_poly_func_alt",
    "custom_tanh_func",
    "custom_tanh_func_alt",
    "generalized_logistic_func",
    "generalized_logistic_func_alt",
    "monotonic_quadratic_spline",
]


@jax.jit
def monotonic_poly_func(x, A, alpha, x0, c=1.0):
    r"""
    .. math::

        A \, \left[c - (1 - (x/x_0)^(1/\beta))^\beta \right]
        \beta = \frac{1 - \alpha}{\alpha}

    This is a custom family of functions that can be controlled such that they are
    monotonic and have constant sign of the curvature (second derivative). This is
    mostly used internally to set the dependence of the :math:`e_m(r_z)` functions. This
    was inspired by the discussion in this StackExchange post:
    https://math.stackexchange.com/questions/65641/i-need-to-define-a-family-one-parameter-of-monotonic-curves
    """
    beta = (1 - alpha) / alpha
    return A * (c - (1 - (x / x0) ** (1 / beta)) ** beta)


@jax.jit
def monotonic_poly_func_alt(x, f0, fx, alpha, x0, xval=1.0):
    """
    An alternate parametrization of the designer function ``monotonic_designer_func()``

    This is a custom family of functions that can be controlled such that they are
    monotonic and have constant sign of the curvature (second derivative). This is
    mostly used internally to set the dependence of the :math:`e_m(r_z)` functions. This
    was inspired by the discussion in this StackExchange post:
    https://math.stackexchange.com/questions/65641/i-need-to-define-a-family-one-parameter-of-monotonic-curves
    """
    A = (fx - f0) / (1 + monotonic_poly_func(xval, 1.0, alpha, x0, c=0.0))
    offset = f0 + A
    return monotonic_poly_func(x, c=0.0, A=A, alpha=alpha, x0=x0) + offset


@jax.jit
def custom_tanh_func(x, A, alpha, x0):
    r"""
    .. math::

        A \, \tanh\left( (x/x_0)^(1/\alpha) \right)^\alpha

    This is a custom family of functions that can be controlled such that they are
    monotonic and have constant sign of the curvature (second derivative).
    """
    return A * jnp.tanh((x / x0) ** (1 / alpha)) ** alpha


@jax.jit
def custom_tanh_func_alt(x, f_xval, alpha, x0, xval=1.0):
    r"""
    An alternate parametrization of ``custom_tanh_func()``

    .. math::

        A \, \tanh\left( (x/x_0)^(1/\alpha) \right)^\alpha
    """
    A = f_xval / jnp.tanh((xval / x0) ** (1 / alpha)) ** alpha
    return custom_tanh_func(x, A, alpha, x0)


def generalized_logistic_func(t, t0, A, B, C, K, nu):
    """
    https://en.wikipedia.org/wiki/Generalised_logistic_function
    """
    denom = (C + jnp.exp(-B * (t - t0))) ** (1 / nu)
    return A + (K - A) / denom


def generalized_logistic_func_alt(t, t0, F1, B, C, nu, t1=1.0):
    """
    https://en.wikipedia.org/wiki/Generalised_logistic_function

    Now: F1 is the value at t1, and it is constrained to be 0 at 0.
    """
    D0 = (C + jnp.exp(B * t0)) ** (1 / nu)
    D1 = (C + jnp.exp(-B * (t1 - t0))) ** (1 / nu)
    A = D1 * F1 / (D1 - D0)
    K = A * (1 - D0)
    return generalized_logistic_func(t, t0, A, B, C, K, nu)


@jax.jit
def monotonic_quadratic_spline(x, y, x_eval):
    """
    The zeroth element in the knot value array is the value of the spline at x[0], but
    all other values passed in via y are the *derivatives* of the function at the knot
    locations x[1:].
    """

    # Checked that using .at[].set() is faster than making padded arrays and stacking
    x = jnp.array(x)
    y = jnp.array(y)
    x_eval = jnp.array(x_eval)

    N = 3 * (len(x) - 1)
    A = jnp.zeros((N, N))
    b = jnp.zeros((N,))
    A = A.at[0, :3].set([x[0] ** 2, x[0], 1])
    b = b.at[0].set(y[0])
    A = A.at[1, :3].set([2 * x[1], 1, 0])
    b = b.at[1].set(y[1])

    for i, n in enumerate(2 * jnp.arange(1, len(x) - 1, 1), start=1):
        A = A.at[n, 3 * i : 3 * i + 3].set([2 * x[i], 1, 0])
        b = b.at[n].set(y[i])
        A = A.at[n + 1, 3 * i : 3 * i + 3].set([2 * x[i + 1], 1, 0])
        b = b.at[n + 1].set(y[i + 1])

    for j, m in enumerate(jnp.arange(2 * (len(x) - 1), N - 1)):
        A = A.at[m, 3 * j : 3 * j + 3].set([x[j + 1] ** 2, x[j + 1], 1])
        A = A.at[m, 3 * (j + 1) : 3 * (j + 1) + 3].set(
            [-(x[j + 1] ** 2), -x[j + 1], -1]
        )

    A = A.at[-1, 0].set(1.0)

    coeffs = jnp.linalg.solve(A, b)

    # Determine the interval that x lies in
    ind = jnp.digitize(x_eval, x) - 1
    ind = 3 * jnp.clip(ind, 0, len(x) - 2)
    coeff_ind = jnp.stack((ind, ind + 1, ind + 2), axis=0)

    xxx = jnp.stack([x_eval**2, x_eval, jnp.ones_like(x_eval)], axis=0)
    return jnp.sum(coeffs[coeff_ind] * xxx, axis=0)
