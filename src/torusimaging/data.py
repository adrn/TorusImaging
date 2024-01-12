"""Utilities for working with kinematic and stellar label data."""


__all__ = ["get_binned_counts", "get_binned_label"]


import astropy.units as u
import numpy as np
import numpy.typing as npt
from astropy.stats import median_absolute_deviation as MAD
from gala.units import UnitSystem
from scipy.stats import binned_statistic_2d


def _get_bins_tuple(bins, units=None):
    if isinstance(bins, dict):
        bins = (bins["pos"], bins["vel"])

    if units is not None:
        bins = [b.decompose(units).value if hasattr(b, "unit") else b for b in bins]
    else:
        bins = [b.value if hasattr(b, "unit") else b for b in bins]

    return bins


def _get_arr(x, units):
    if units is not None:
        return x.decompose(units).value if hasattr(x, "unit") else x
    return x.value if hasattr(x, "unit") else x


@u.quantity_input
def get_binned_counts(
    pos: u.Quantity[u.kpc],
    vel: u.Quantity[u.km / u.s],
    bins: dict[str, u.Quantity] | tuple,
    units: UnitSystem | None = None,
) -> dict[str, u.Quantity | npt.NDArray]:
    """Bin the data in pixels of phase-space coordinates (pos, vel) and return the
    number of stars (counts) and log-number of stars (label).

    Parameters
    ----------
    pos
        The position values.
    vel
        The velocity values.
    bins
        A specification of the bins. This can either be a tuple, where the order
        is assumed to be (pos, vel), or a dictionary with keys "pos" and "vel".
    units
        The unit system to work in.

    Returns
    -------
    dict
        Keys are "pos", "vel", "counts", "label", where label is the natural log of the
        counts.
    """

    pos = _get_arr(pos, units)
    vel = _get_arr(vel, units)

    H, xe, ye = np.histogram2d(
        pos,
        vel,
        bins=_get_bins_tuple(bins, units=units),
    )
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    xc, yc = np.meshgrid(xc, yc)

    return {
        "pos": xc * units["length"],
        "vel": yc * units["length"] / units["time"],
        "counts": H.T,
        "label": np.log(H.T),
    }


def _infer_intrinsic_scatter(y, y_err, nan_safe=False):
    import jax
    import jax.numpy as jnp
    import jaxopt

    @jax.jit
    def ln_likelihood(p, y, y_err):
        V = jnp.exp(2 * p["ln_s"]) + y_err**2
        return jnp.nansum(-0.5 * ((y - p["mean"]) ** 2 / V + jnp.log(2 * jnp.pi * V)))

    @jax.jit
    def neg_ln_likelihood(p, y, y_err):
        return -ln_likelihood(p, y, y_err)

    opt = jaxopt.ScipyMinimize(
        method="L-BFGS-B", fun=neg_ln_likelihood, tol=1e-10, maxiter=1000
    )

    p0 = {"mean": np.nanmean(y), "ln_s": jnp.log(np.nanstd(y))}
    res = opt.run(p0, y=y, y_err=y_err)

    s = np.exp(res.params["ln_s"])
    if not res.state.success:
        if nan_safe:
            s = np.nan
        else:
            msg = (
                "Failed to determine error-deconvolved estimate of the intrinsic "
                "scatter in a phase-space pixel."
            )
            raise RuntimeError(msg)

    return s


@u.quantity_input
def get_binned_label(
    pos: u.Quantity[u.kpc],
    vel: u.Quantity[u.km / u.s],
    label: npt.ArrayLike,
    bins: dict[str, u.Quantity] | tuple,
    moment: str = "mean",
    label_err: npt.ArrayLike | None = None,
    units: UnitSystem | None = None,
    s: float | None = None,
    s_N_thresh: int | None = 128,
) -> dict[str, u.Quantity | npt.NDArray]:
    """Bin the data in pixels of phase-space coordinates (pos, vel) and return the
    mean (or other moment) of the label values in each pixel.

    Parameters
    ----------
    pos
        The position values.
    vel
        The velocity values.
    label
        The label values.
    bins
        A specification of the bins. This can either be a tuple, where the order
        is assumed to be (pos, vel), or a dictionary with keys "pos" and "vel".
    moment
        The type of moment to compute. Currently only supports "mean".
    label_err
        The measurement error for each label value.
    units
        The unit system to work in.
    s
        The intrinsic scatter of label values within each pixel. If not provided,
        this will be estimated from the data.
    s_N_thresh
        If the intrinsic scatter ``s`` is not specified, this sets the threshold for the
        number of objects per bin required to estimate the intrinsic scatter.

    Returns
    -------
    dict
        Keys are "pos", "vel", "counts", "label", and "label_err".
    """
    if moment != "mean":
        msg = "Only the mean is currently supported."
        raise NotImplementedError(msg)

    pre_err_state = np.geterr()
    np.seterr(divide="ignore", invalid="ignore")

    pos = _get_arr(pos, units)
    vel = _get_arr(vel, units)
    bins = _get_bins_tuple(bins, units)

    xc = 0.5 * (bins[0][:-1] + bins[0][1:])
    yc = 0.5 * (bins[1][:-1] + bins[1][1:])
    xc, yc = np.meshgrid(xc, yc)

    binned = {
        "pos": xc * units["length"],
        "vel": yc * units["length"] / units["time"],
    }

    # For bin numbers and other stuff below:
    counts_stat = binned_statistic_2d(
        pos, vel, None, bins=bins, statistic="count", expand_binnumbers=True
    )
    counts = counts_stat.statistic

    if label_err is None:
        # No label errors provided - assume dominated by intrinsic scatter
        label_err = np.zeros_like(label)

        if s is None:
            # estimate just doing the stddev of bins with more than N objects
            std_stat = binned_statistic_2d(
                pos, vel, label, bins=bins, statistic=np.nanstd
            )
            s = np.nanmean(std_stat.statistic[counts > s_N_thresh])

    if s is None:
        # Label errors provided, but no intrinsic scatter provided - need to estimate
        # this for bins with many objects
        high_N_bins = np.stack(np.where(counts > s_N_thresh)) + 1
        high_N_bins = high_N_bins[
            :, np.argsort(counts[np.where(counts > s_N_thresh)])[::-1]
        ]
        high_N_bins = high_N_bins[:, np.any(high_N_bins != 1, axis=0)]

        # TODO: magic number - this limits to a subset of the 16 most populated bins
        s_trials = []
        for bin_idx in high_N_bins.T[:16]:
            bin_mask = np.all(counts_stat.binnumber == bin_idx[:, None], axis=0)
            # to get bin location: stat.x_edge[bin_idx[0]], stat.y_edge[bin_idx[1]]

            s_trials.append(
                _infer_intrinsic_scatter(
                    label[bin_mask], label_err[bin_mask], nan_safe=True
                )
            )
        s = np.nanmean(s_trials)

        if not np.isfinite(s):
            msg = "Failed to determine intrinsic scatter from label data"
            raise ValueError(msg)

    if np.all(label_err == 0):
        # No label errors provided
        stat_mean = binned_statistic_2d(
            pos, vel, label, bins=bins, statistic=np.nanmean
        )
        mean = stat_mean.statistic
        mean_err = 0.0

    else:
        # Compute the mean and the "error on the mean" in each bin:
        stat_mean1 = binned_statistic_2d(
            pos,
            vel,
            label / label_err**2,
            bins=bins,
            statistic="sum",
        )
        stat_mean2 = binned_statistic_2d(
            pos, vel, 1 / label_err**2, bins=bins, statistic="sum"
        )
        mean = stat_mean1.statistic / stat_mean2.statistic
        mean_err = np.sqrt(1 / stat_mean2.statistic)

    binned["counts"] = counts.T
    binned["label"] = mean.T
    binned["label_err"] = np.sqrt(mean_err**2 + s**2 / counts).T
    binned["label_err"][~np.isfinite(binned["label_err"])] = np.nan
    # binned["s"] = s

    np.seterr(**pre_err_state)

    return binned


def estimate_Omega(binned_data):
    # TODO: percentile values are hard-coded and arbitrary
    inner_mask = (
        np.abs(binned_data["pos"]) < np.nanpercentile(np.abs(binned_data["pos"]), 15)
    ) & (np.abs(binned_data["vel"]) < np.nanpercentile(np.abs(binned_data["vel"]), 15))
    inner_label_val = np.nanmean(binned_data["label"][inner_mask])

    diff = np.abs(binned_data["label"] - inner_label_val)

    ell_mask = diff < np.nanpercentile(diff, 15)
    tmpv = binned_data["vel"][ell_mask]
    tmpz = binned_data["pos"][ell_mask]
    init_Omega = MAD(tmpv) / MAD(tmpz)

    return init_Omega * u.rad
