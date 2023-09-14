import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve


def plot_data_models_residual(
    binned_data,
    model,
    params_fit,
    params_init=None,
    label_name="counts",
    smooth_residual=None,
    vlim_residual=0.3,
    fractional_residual=True,
    usys=None,
):
    """
    Make a 4 panel figure showing data (number counts of stars in z-vz), initial model,
    fitted model, and residual of the fitted model

    Parameters
    ----------
    binned_data : dict
    model :
        The model at parameter initialization.
    smooth_residual : float, None (optional)
        If specified (as a float), smooth the residual image by a Gaussian with kernel
        with set by this parameter.
    vlim_residual : float (optional)
        The vmin, vmax for the residual colormap are set using this value such that
        ``vmin=-vlim_residual`` and ``vmax=vlim_residual``.
    """
    if usys is None:
        usys = model.unit_sys

    bd = {
        k: v.decompose(usys).value if hasattr(v, "unit") else v
        for k, v in binned_data.items()
    }

    if label_name == "counts":
        vlim = dict(norm=mpl.colors.LogNorm(vmin=0.5), shading="auto")

        def model_func(*args, **kwargs):
            return np.exp(model.ln_density(*args, **kwargs))

    else:
        vlim = dict(
            norm=mpl.colors.Normalize(
                *np.nanpercentile(binned_data[label_name], [1, 99])
            ),
            shading="auto",
        )
        model_func = model.label

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

    cs = axes[0].pcolormesh(bd["vel"], bd["pos"], bd[label_name], **vlim)

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
    if fractional_residual:
        resid = np.array((bd[label_name] - model_H) / model_H)
    else:
        resid = np.array((bd[label_name] - model_H))
    if smooth_residual is not None:
        resid = convolve(resid, Gaussian2DKernel(smooth_residual))

    if not hasattr(vlim_residual, "__len__"):
        vlim_residual = (-vlim_residual, vlim_residual)

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

    if fractional_residual:
        axes[i + 1].set_title("fractional residual")
    else:
        axes[i + 1].set_title("residual")

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
    print(K, np.shape(axes))
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
