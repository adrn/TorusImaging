from astropy.convolution import convolve, Gaussian2DKernel
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def plot_data_models_residual(
        data_H, model0, model, smooth_residual=None, vlim_residual=0.3, usys=None
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
    
    vlim = dict(norm=mpl.colors.LogNorm(vmax=3e4), shading='auto')

    fig, axes = plt.subplots(
        1, 4, figsize=(22, 5.4), sharex=True, sharey=True,
        constrained_layout=True
    )

    cs = axes[0].pcolormesh(data_H['vz'], data_H['z'], data_H['H'], **vlim)

    # Initial model:
    model0_H = np.exp(model0.ln_density(z=data_H['z'], vz=data_H['vz']))
    cs = axes[1].pcolormesh(data_H['vz'], data_H['z'], model0_H, **vlim)

    # Fitted model:
    model_H = np.exp(model.ln_density(z=data_H['z'], vz=data_H['vz']))
    cs = axes[2].pcolormesh(data_H['vz'], data_H['z'], model_H, **vlim)
    fig.colorbar(cs, ax=axes[:3], aspect=40)

    # Residual:
    resid = np.array((data_H['H'] - model_H) / model_H)
    resid[data_H['H'] < 5] = np.nan
    if smooth_residual is not None:
        resid = convolve(resid, Gaussian2DKernel(smooth_residual))
    cs = axes[3].pcolormesh(
        data_H['vz'], data_H['z'],
        resid,
        vmin=-0.3, vmax=0.3,
        cmap='RdYlBu_r',
        shading='auto'
    )
    fig.colorbar(cs, ax=axes[3], aspect=40)

    for ax in axes:
        ax.set_xlabel(f'$v_z$ [{usys["length"] / usys["time"]:latex_inline}]')
    axes[0].set_ylabel(f'$z$ [{usys["length"]:latex_inline}]')

    axes[0].set_title('data')
    axes[1].set_title('initial model')
    axes[2].set_title('fitted model')
    axes[3].set_title('normalized residual')

    return fig, axes
