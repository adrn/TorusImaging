# Standard library
from collections import defaultdict
import pickle

# Third-party
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from scipy.stats import binned_statistic

__all__ = [
    "AbundanceAnomalyMaschine",
    "run_bootstrap_coeffs",
    "get_cos2th_zerocross",
    "ZeroCrossWorker",
]


class AbundanceAnomalyMaschine:
    def __init__(self, actions, tree_K=64, sinusoid_K=2):
        """
        Parameters
        ----------
        actions : quantity-like
            The value of the actions for the sample of stars. This should have shape
            ``(n_stars, n_actions)`` where ``n_actions`` is at most 3.
        tree_K : int (optional)
            The number of neighbors used to estimate the action-local
            mean abundances.
        sinusoid_K : int (optional)
            The number of cos/sin terms in the sinusoid fit to the
            abundance anomaly variations with angle.
        """
        self.actions = u.Quantity(actions)
        self.tree_K = int(tree_K)
        self.sinusoid_K = int(sinusoid_K)
        self._cache = dict()

    def get_mean_abundance_anomaly(self, elem, elem_err):
        """
        Compute the mean abundance anomaly for the given abundance ratio as a
        function of the specified angle coordinate.

        Parameters
        ----------
        actions : quantity-like
            The value of the actions for the sample of stars. This should have shape
            ``(n_stars, n_actions)`` where ``n_actions`` is at most 3.
        elem : array-like
            The element abundance data for the sample of stars.
        elem_err : array-like
            The element abundance uncertainties.

        Returns
        -------
        elem_anomaly : `numpy.ndarray`
        elem_anomaly_error : numpy.ndarray`

        """

        # Actions without units:
        X = self.actions.value

        assert len(elem) == X.shape[0]
        if not np.all(np.isfinite(elem)):
            raise ValueError(
                "You passed in NaN or Inf values in element abundance array!"
            )

        if "action_neighbors_idx" not in self._cache:
            # Construct the tree for all actions to find neighbors
            tree = cKDTree(X)
            _, idx = tree.query(X, k=self.tree_K + 1)
            self._cache["action_neighbors_idx"] = idx

            xhat = np.mean(X[idx[:, 1:]], axis=1) - X
            dx = X[idx[:, 1:]] - X[:, None]
            x = np.einsum("nij,nj->ni", dx, xhat)

            self._cache["action_neighbors_x"] = x

        else:
            idx = self._cache["action_neighbors_idx"]
            x = self._cache["action_neighbors_x"]

        y = elem[idx[:, 1:]]

        # The fix for steep gradients: see Appendix of Price-Whelan et al. 2021
        w = np.sum(x ** 2, axis=1)[:, None] - x * np.sum(x, axis=1)[:, None]
        # TODO: this line might be wrong!! Weighting by inverse-variance as well to
        # account for the element abundance uncertainties
        # w = w * 1 / elem_err[:, None] ** 2
        means = np.sum(y * w, axis=1) / np.sum(w, axis=1)
        # mean_vars = 1 / np.sum(1 / elem_err ** 2)
        mean_vars = 0.0

        d_elem = np.asarray(elem - means)
        d_elem_errs = np.sqrt(np.asarray(elem_err ** 2 + mean_vars))

        return d_elem, d_elem_errs

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

    @u.quantity_input
    def get_coeffs(self, angle: u.radian, elem, elem_err):
        """
        Internal function to compute maximum likelihood sin/cos term
        coefficients for the input
        """
        y, yerr = self.get_mean_abundance_anomaly(elem, elem_err)

        M = self.get_M(angle.to_value(u.radian))
        Cinv_diag = 1 / yerr ** 2
        MT_Cinv = M.T * Cinv_diag[None]
        MT_Cinv_M = MT_Cinv @ M
        coeffs = np.linalg.solve(MT_Cinv_M, MT_Cinv @ y)
        coeffs_cov = np.linalg.inv(MT_Cinv_M)
        return coeffs, coeffs_cov

    def get_binned_anomaly(self, angle, elem, elem_err, theta_z_step=5 * u.deg):
        """
        Retrieve the binned mean abundance anomaly for the specified angle and
        abundance ratio. Mainly used for visualization.

        Parameters
        ----------
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
        d_elem, d_elem_errs = self.get_mean_abundance_anomaly(elem, elem_err)
        d_elem_ivar = 1 / d_elem_errs ** 2

        # TODO: somehow incorporate intrinsic spread in d_elem here (in PW21
        # eyeballed to be ~0.04)
        # d_elem_ivar = np.full_like(d_elem, 1 / 0.04**2)  # MAGIC NUMBER

        angle_rad = angle.to_value(u.radian)
        stat1 = binned_statistic(
            angle_rad, d_elem * d_elem_ivar, bins=angz_bins, statistic="sum"
        )
        stat2 = binned_statistic(
            angle_rad, d_elem_ivar, bins=angz_bins, statistic="sum"
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
