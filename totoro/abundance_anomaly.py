# Standard library
from collections import defaultdict
import pickle

# Third-party
import astropy.coordinates as coord
import astropy.table as at
import astropy.units as u
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from scipy.stats import binned_statistic

__all__ = [
    "AbundanceAnomalyMaschine",
    "run_bootstrap_coeffs",
    "get_cos2th_zerocross",
]


class AbundanceAnomalyMaschine:
    def __init__(self, data, tree_K=64, sinusoid_K=2):
        """
        Parameters
        ----------
        data : `~astropy.table.QTable`
            Table of actions, angles, and abundances.
        tree_K : int (optional)
            The number of neighbors used to estimate the action-local
            mean abundances.
        sinusoid_K : int (optional)
            The number of cos/sin terms in the sinusoid fit to the
            abundance anomaly variations with angle.
        """

        self.data = at.QTable(data, copy=False)

        # TODO: take defaults from config file
        self.tree_K = int(tree_K)
        self.sinusoid_K = int(sinusoid_K)

    def get_theta_anomaly(
        self, elem_name, angle_index, action_unit=30 * u.km / u.s * u.kpc
    ):
        """
        Compute the mean abundance anomaly for the given abundance ratio column
        name as a function of the specified angle coordinate.

        Parameters
        ----------
        elem_name : str
            The column name of the abundance ratio.
        angle_index : int
            The index specifying which angle coordinate to use. 0=R, 1=phi, 2=z
        action_unit : unit-like, quantity-like (optional)
            The unit to convert the actions to when constructing the KD tree
            used to find action-space neighbors.

        Returns
        -------
        angles : `~astropy.units.Quantity`
        elem_anomaly : ndarray
        elem_anomaly_error : ndarray

        """
        action_unit = u.Quantity(action_unit)

        # Actions without units:
        X = self.data["actions"].to_value(action_unit)
        angles = coord.Angle(self.data["angles"]).wrap_at(360 * u.deg)

        # element abundance
        elem = self.data[elem_name]
        # TODO: need to read error column format from config file
        elem_errs = self.data[f"{elem_name}_ERR"]

        tree = cKDTree(X)
        dists, idx = tree.query(X, k=self.tree_K + 1)

        xhat = np.mean(X[idx[:, 1:]], axis=1) - X
        dx = X[idx[:, 1:]] - X[:, None]
        x = np.einsum("nij,nj->ni", dx, xhat)
        y = elem[idx[:, 1:]]

        # The fix for steep gradients: see Appendix of PW+2021
        w = np.sum(x ** 2, axis=1)[:, None] - x * np.sum(x, axis=1)[:, None]
        means = np.sum(y * w, axis=1) / np.sum(w, axis=1)

        d_elem = np.array(elem - means)
        d_elem_errs = np.array(elem_errs)

        return angles, d_elem, d_elem_errs

    def get_M(self, angle):
        """
        Construct the design matrix
        """
        M = np.full((len(angle), 1 + 2 * self.sinusoid_K), np.nan)
        M[:, 0] = 1.0

        for n in range(self.sinusoid_K):
            M[:, 1 + 2 * n] = np.cos((n + 1) * angle)
            M[:, 2 + 2 * n] = np.sin((n + 1) * angle)

        return M

    def get_coeffs(self, angle, y, yerr):
        """
        Internal function to compute maximum likelihood sin/cos term
        coefficients for the input
        """
        M = self.get_M(angle.to_value(u.radian))
        Cinv_diag = 1 / yerr ** 2
        MT_Cinv = M.T * Cinv_diag[None]
        MT_Cinv_M = MT_Cinv @ M
        coeffs = np.linalg.solve(MT_Cinv_M, MT_Cinv @ y)
        coeffs_cov = np.linalg.inv(MT_Cinv_M)
        return coeffs, coeffs_cov

    def get_coeffs_for_elem(self, elem_name):
        """
        Retrieve maximum likelihood sin/cos term coefficients for the element
        """
        tz, d_elem, d_elem_errs = self.get_theta_z_anomaly(elem_name)
        return self.get_coeffs(tz, d_elem, d_elem_errs)

    def get_binned_anomaly(self, elem_name, theta_z_step=5 * u.deg):
        """
        Retrieve the binned mean abundance anomaly for the specified abundance
        ratio column name

        Parameters
        ----------
        elem_name : str
            The column name of the abundance ratio.
        theta_z_step : `astropy.units.Quantity` [angle] (optional)
            The bin step size for the vertical angle bins.

        Returns
        -------
        bin_centers : `~numpy.ndarray`
        mean_abundance_deviation : `~numpy.ndarray`
        mean_abundance_deviation_error : `~numpy.ndarray`
        """
        step = coord.Angle(theta_z_step).to_value(u.rad)
        angz_bins = np.arange(0, 2 * np.pi + step, step)
        theta_z, d_elem, d_elem_errs = self.get_theta_z_anomaly(elem_name)
        d_elem_ivar = 1 / d_elem_errs ** 2
        # d_elem_ivar = np.full_like(d_elem, 1 / 0.04**2)  # MAGIC NUMBER ??

        stat1 = binned_statistic(
            theta_z, d_elem * d_elem_ivar, bins=angz_bins, statistic="sum"
        )
        stat2 = binned_statistic(
            theta_z, d_elem_ivar, bins=angz_bins, statistic="sum"
        )

        binx = 0.5 * (angz_bins[:-1] + angz_bins[1:])
        means = stat1.statistic / stat2.statistic
        errs = np.sqrt(1 / stat2.statistic)

        return binx, means, errs


def run_bootstrap_coeffs(
    aafs, elem_name, bootstrap_K=128, seed=42, overwrite=False, cache_path=None
):
    if cache_path is not None:
        cache_filename = f"coeffs-bootstrap{bootstrap_K}-{elem_name}.pkl"
        coeffs_cache = cache_path / cache_filename

        if coeffs_cache.exists() and not overwrite:
            with open(coeffs_cache, "rb") as f:
                all_bs_coeffs = pickle.load(f)
            return all_bs_coeffs

    all_bs_coeffs = {}
    for name in aafs:
        aaf = aafs[name]

        if seed is not None:
            np.random.seed(seed)

        bs_coeffs = []
        for k in range(bootstrap_K):
            bootstrap_idx = np.random.choice(len(aaf), size=len(aaf))
            atm = AbundanceAnomalyMaschine(aaf[bootstrap_idx])
            coeffs, _ = atm.get_coeffs_for_elem(elem_name)
            bs_coeffs.append(coeffs)

        all_bs_coeffs[name] = np.array(bs_coeffs)

    if cache_path is not None:
        with open(coeffs_cache, "wb") as f:
            pickle.dump(all_bs_coeffs, f)

    return all_bs_coeffs


def get_cos2th_zerocross(coeffs):
    summary = defaultdict(lambda *args: defaultdict(list))
    for i in range(5):
        for k in coeffs:
            summary[i]["mdisk"].append(float(k))
            summary[i]["y"].append(np.mean(coeffs[k][:, i]))
            summary[i]["y_err"].append(np.std(coeffs[k][:, i]))

        summary[i]["mdisk"] = np.array(summary[i]["mdisk"])
        summary[i]["y"] = np.array(summary[i]["y"])
        summary[i]["y_err"] = np.array(summary[i]["y_err"])

        idx = summary[i]["mdisk"].argsort()
        for key in summary[i].keys():
            summary[i][key] = summary[i][key][idx]

    # cos2theta term:
    s = summary[3]
    zero_cross = interp1d(s["y"], s["mdisk"], fill_value="extrapolate")(0.0)
    zero_cross1 = interp1d(
        np.array(s["y"]) - np.array(s["y_err"]),
        s["mdisk"],
        fill_value="extrapolate",
    )(0.0)
    zero_cross2 = interp1d(
        np.array(s["y"]) + np.array(s["y_err"]),
        s["mdisk"],
        fill_value="extrapolate",
    )(0.0)

    zero_cross_err = (
        np.abs(zero_cross2 - zero_cross),
        np.abs(zero_cross1 - zero_cross),
    )
    return summary, zero_cross, zero_cross_err


class ZeroCrossWorker:
    def __init__(self, aafs, cache_path=None, bootstrap_K=128):
        self.aafs = aafs
        self.cache_path = cache_path
        self.bootstrap_K = int(bootstrap_K)

    def __call__(self, elem_name):
        clean_aafs = {}
        for k in self.aafs:
            clean_aafs[k] = self.aafs[k][self.aafs[k][elem_name] > -3]

        bs_coeffs = run_bootstrap_coeffs(
            clean_aafs,
            elem_name,
            bootstrap_K=self.bootstrap_K,
            cache_path=self.cache_path,
        )
        s, zc, zc_err = get_cos2th_zerocross(bs_coeffs)
        return elem_name, [zc, zc_err]
