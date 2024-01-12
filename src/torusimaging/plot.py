"""Utilities for plotting torusimaging models and data."""

__all__ = ["plot_data_models_residual", "plot_spline_functions"]

import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from astropy.convolution import Gaussian2DKernel, convolve
from gala.units import UnitSystem

from .model import TorusImaging1D
from .model_spline import TorusImaging1DSpline


def plot_data_models_residual(
    binned_data: dict[str, u.Quantity | npt.ArrayLike],
    model: TorusImaging1D,
    params_fit: dict,
    params_init: dict | None = None,
    smooth_residual: float | None = None,
    vlim_residual: float | None = None,
    residual_normalization: npt.ArrayLike | None = None,
    usys: UnitSystem | None = None,
) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """Make a 4 panel figure showing data (number counts of stars in z-vz), initial
    model, fitted model, and residual of the fitted model

    Parameters
    ----------
    binned_data
        The binned data dictionary.
    model
        The model instance.
    params_fit
        The optimized parameters, or the MAP parameters, or the parameters you would
        like to show.
    params_init
        The initial parameters. If specified, will plot the initial model as well.
    smooth_residual
        If specified (as a float), smooth the residual image by a Gaussian with kernel
        with set by this parameter.
    vlim_residual
        The vmin, vmax for the residual colormap are set using this value such that
        ``vmin=-vlim_residual`` and ``vmax=vlim_residual``.
    residual_normalization
        If specified, the residual is divided by this value. This is useful for
        plotting fractional residuals.
    usys
        The unit system to use for plotting. If None, will use the unit system of the
        model.
    """
    if usys is None:
        usys = model.units

    bd = {
        k: v.decompose(usys).value if hasattr(v, "unit") else v
        for k, v in binned_data.items()
    }

    vlim = {
        "norm": mpl.colors.Normalize(
            *np.percentile(
                binned_data["label"][np.isfinite(binned_data["label"])], [1, 99]
            )
        ),
        "shading": "auto",
    }
    model_func = model._get_label

    ncols = 3
    if params_init is not None:
        ncols += 1

    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(5.5 * ncols, 5.4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    cs = axes[0].pcolormesh(bd["vel"], bd["pos"], bd["label"], **vlim)

    i = 1

    if params_init is not None:
        # Initial model:
        model0_H = model_func(pos=bd["pos"], vel=bd["vel"], params=params_init)
        cs = axes[1].pcolormesh(bd["vel"], bd["pos"], model0_H, **vlim)
        i += 1

    # Fitted model:
    model_H = model_func(pos=bd["pos"], vel=bd["vel"], params=params_fit)
    cs = axes[i].pcolormesh(bd["vel"], bd["pos"], model_H, **vlim)
    fig.colorbar(cs, ax=axes[: i + 1], aspect=40)

    # Residual:
    if residual_normalization is not None:
        resid = np.array((bd["label"] - model_H) / residual_normalization)
    else:
        resid = np.array(bd["label"] - model_H)
    if smooth_residual is not None:
        resid = convolve(resid, Gaussian2DKernel(smooth_residual))

    if vlim_residual is None:
        vlim_residual = np.nanpercentile(np.abs(resid), 99)

    if not hasattr(vlim_residual, "__len__"):
        vlim_residual = (-vlim_residual, vlim_residual)  # pylint: disable=E1130

    cs = axes[i + 1].pcolormesh(
        bd["vel"],
        bd["pos"],
        resid,
        vmin=vlim_residual[0],
        vmax=vlim_residual[1],
        cmap="RdYlBu_r",
        shading="auto",
    )
    fig.colorbar(cs, ax=axes[i + 1], aspect=40)

    for ax in axes:
        ax.set_xlabel(f'$v_z$ [{usys["length"] / usys["time"]:latex_inline}]')
    axes[0].set_ylabel(f'$z$ [{usys["length"]:latex_inline}]')

    axes[0].set_title("data")
    i = 1
    if params_init is not None:
        axes[1].set_title("initial model")
        i += 1
    axes[i].set_title("fitted model")

    if residual_normalization is not None:
        axes[i + 1].set_title("normalized residual")
    else:
        axes[i + 1].set_title("residual")

    return fig, axes


def plot_spline_functions(
    model: TorusImaging1DSpline, params: dict
) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """Plot the spline functions used in a ``TorusImaging1DSpline`` model.

    Parameters
    ----------
    model
        The model instance.
    params
        A dictionary of parameter values.
    """
    r_e_grid = np.linspace(0, model._label_knots.max(), 128)
    e_vals = model._get_es(r_e_grid, params["e_params"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")

    ax = axes[0]
    sum_ = None
    for m, vals in e_vals.items():
        (l,) = ax.plot(r_e_grid, vals, marker="", label=f"$e_{m}$")  # noqa: E741
        ax.scatter(
            model._e_knots[m],
            model.e_funcs[m](model._e_knots[m], params["e_params"][m]["vals"]),
            color=l.get_color(),
        )

        if sum_ is None:
            sum_ = vals
        else:
            sum_ += vals

    ax.plot(r_e_grid, sum_, ls="--", marker="")
    ax.set_title("$e_m$ functions")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_xlabel("$r_e$")

    ax = axes[1]
    l_vals = model.label_func(r_e_grid, **params["label_params"])
    (l,) = ax.plot(r_e_grid, l_vals, marker="")  # noqa: E741

    l_vals = model.label_func(model._label_knots, **params["label_params"])
    ax.scatter(model._label_knots, l_vals, color=l.get_color())
    ax.set_xlabel("$r$")
    ax.set_title("Label function")

    return fig, axes


def plot_cov_ellipse(mean, cov, nstd=1, ax=None, **kwargs):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = mpl.patches.Ellipse(
        xy=mean, width=width, height=height, angle=theta, **kwargs
    )

    ax.add_artist(ellip)
    return ellip


def plot_cov_corner(cov, mean=None, labels=None, subplots_kw=None, ellipse_kw=None):
    if mean is None:
        mean = np.zeros(cov.shape[0])

    if subplots_kw is None:
        subplots_kw = {}

    if ellipse_kw is None:
        ellipse_kw = {}

    K = cov.shape[0]
    subplots_kw.setdefault("figsize", (K * 3 + 1, K * 3))
    subplots_kw.setdefault("sharex", "col")
    subplots_kw.setdefault("sharey", "row")
    subplots_kw.setdefault("constrained_layout", True)

    fig, axes = plt.subplots(K - 1, K - 1, **subplots_kw)

    for i in range(K - 1):
        for j in range(1, K):
            ax = axes[j - 1, i]
            if i >= j:
                ax.set_visible(False)
                continue

            idx = [i, j]
            subcov = cov[idx][:, idx]

            if np.any(~np.isfinite(mean[idx])) or np.any(np.diag(subcov) <= 0):
                continue

            plot_cov_ellipse(mean[idx], subcov, ax=ax, **ellipse_kw)

            xsize = np.sqrt(subcov[0, 0])
            ysize = np.sqrt(subcov[1, 1])
            ax.set_xlim(-1.25 * xsize, 1.25 * xsize)
            ax.set_ylim(-1.25 * ysize, 1.25 * ysize)

    if labels is not None:
        for i in range(K - 1):
            axes[i, 0].set_ylabel(labels[i + 1])
        for i in range(K - 1):
            axes[-1, i].set_xlabel(labels[i])

    return fig, axes
