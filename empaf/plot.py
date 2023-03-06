import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve


def plot_data_models_residual(
    data_H,
    model,
    params_init,
    params_fit,
    smooth_residual=None,
    vlim_residual=0.3,
    usys=None,
):
    """
    Make a 4 panel figure showing data (number counts of stars in z-vz), initial model,
    fitted model, and residual of the fitted model

    Parameters
    ----------
    data_H : dict
        Containing arrays in keys 'z', 'vz', 'H' for the bins in z, vz and number counts
        in H.
    model0 : `VerticalOrbitModel`
        The model at parameter initialization.
    model : `VerticalOrbitModel`
        The model after fitting to the data.
    smooth_residual : float, None (optional)
        If specified (as a float), smooth the residual image by a Gaussian with kernel
        with set by this parameter.
    vlim_residual : float (optional)
        The vmin, vmax for the residual colormap are set using this value such that
        ``vmin=-vlim_residual`` and ``vmax=vlim_residual``.
    """
    if usys is None:
        usys = model.unit_sys

    vlim = dict(norm=mpl.colors.LogNorm(vmin=1e-1, vmax=3e4), shading="auto")

    fig, axes = plt.subplots(
        1, 4, figsize=(22, 5.4), sharex=True, sharey=True, constrained_layout=True
    )

    cs = axes[0].pcolormesh(data_H["vz"], data_H["z"], data_H["H"], **vlim)

    # Initial model:
    model0_H = np.exp(
        model.ln_density(z=data_H["z"], vz=data_H["vz"], params=params_init)
    )
    cs = axes[1].pcolormesh(data_H["vz"], data_H["z"], model0_H, **vlim)

    # Fitted model:
    model_H = np.exp(
        model.ln_density(z=data_H["z"], vz=data_H["vz"], params=params_fit)
    )
    cs = axes[2].pcolormesh(data_H["vz"], data_H["z"], model_H, **vlim)
    fig.colorbar(cs, ax=axes[:3], aspect=40)

    # Residual:
    resid = np.array((data_H["H"] - model_H) / model_H)
    resid[data_H["H"] < 5] = np.nan
    if smooth_residual is not None:
        resid = convolve(resid, Gaussian2DKernel(smooth_residual))
    cs = axes[3].pcolormesh(
        data_H["vz"],
        data_H["z"],
        resid,
        vmin=-vlim_residual,
        vmax=vlim_residual,
        cmap="RdYlBu_r",
        shading="auto",
    )
    fig.colorbar(cs, ax=axes[3], aspect=40)

    for ax in axes:
        ax.set_xlabel(f'$v_z$ [{usys["length"] / usys["time"]:latex_inline}]')
    axes[0].set_ylabel(f'$z$ [{usys["length"]:latex_inline}]')

    axes[0].set_title("data")
    axes[1].set_title("initial model")
    axes[2].set_title("fitted model")
    axes[3].set_title("normalized residual")

    return fig, axes


def plot_data_models_label_residual(
    data_H,
    model,
    params_init,
    params_fit,
    smooth_residual=None,
    vlim=None,
    vlim_residual=0.02,
    usys=None,
    mask_no_data=True,
):
    if usys is None:
        usys = model.unit_sys

    if vlim is None:
        vlim = np.nanpercentile(data_H["label"], [1, 99])
    pcolor_kw = dict(shading="auto", vmin=vlim[0], vmax=vlim[1])

    fig, axes = plt.subplots(
        1, 4, figsize=(22, 5.4), sharex=True, sharey=True, constrained_layout=True
    )

    cs = axes[0].pcolormesh(data_H["vz"], data_H["z"], data_H["label"], **pcolor_kw)

    # Initial model:
    model0_H = np.array(model.label(z=data_H["z"], vz=data_H["vz"], params=params_init))
    if mask_no_data:
        model0_H[~np.isfinite(data_H["label"])] = np.nan
    cs = axes[1].pcolormesh(
        data_H["vz"],
        data_H["z"],
        model0_H,
        **pcolor_kw,
    )

    # Fitted model:
    model_H = np.array(model.label(z=data_H["z"], vz=data_H["vz"], params=params_fit))
    if mask_no_data:
        model_H[~np.isfinite(data_H["label"])] = np.nan
    cs = axes[2].pcolormesh(
        data_H["vz"],
        data_H["z"],
        model_H,
        **pcolor_kw,
    )
    fig.colorbar(cs, ax=axes[:3], aspect=40)

    # Residual:
    #     resid = np.array((data_H['label_stat'] - model_H) / model_H)
    resid = np.array(data_H["label"] - model_H)
    # resid[data_H['H'] < 5] = np.nan
    if smooth_residual is not None:
        resid = convolve(resid, Gaussian2DKernel(smooth_residual))
    cs = axes[3].pcolormesh(
        data_H["vz"],
        data_H["z"],
        resid,
        vmin=-vlim_residual,
        vmax=vlim_residual,
        cmap="RdYlBu_r",
        shading="auto",
    )
    fig.colorbar(cs, ax=axes[3], aspect=40)

    for ax in axes:
        ax.set_xlabel(f'$v_z$ [{usys["length"] / usys["time"]:latex_inline}]')
    axes[0].set_ylabel(f'$z$ [{usys["length"]:latex_inline}]')

    axes[0].set_title("data")
    axes[1].set_title("initial model")
    axes[2].set_title("fitted model")
    axes[3].set_title("residual")

    return fig, axes
