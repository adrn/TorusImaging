import astropy.units as u
import numpy as np
from gala.units import galactic
from scipy.stats import binned_statistic_2d


class OTIData:
    def __init__(self, pos, vel, units=galactic, labels=None, label_errs=None):
        self.units = units
        self._label_names = list(labels.keys())

        # For now, we only support 1D pos and vel:
        self._pos = u.Quantity(pos).decompose(galactic).value
        self._vel = u.Quantity(vel).decompose(galactic).value

        if labels is None:
            labels = {}
        self.labels = {k: np.asarray(v) for k, v in labels.items()}

        if label_errs is None:
            label_errs = {}
        self.label_errs = {k: np.asarray(v) for k, v in label_errs.items()}

        if self._pos.shape != self._vel.shape:
            raise ValueError("Input position and velocity must have the same shape")
        elif self._pos.ndim != 1:
            raise ValueError("Input position and velocity must be 1D arrays")

        for k, v in self.labels.items():
            if self._pos.shape != v.shape:
                raise ValueError(
                    "Input position and velocity must have the same shape as the "
                    f"input labels: label {k} has shape {v.shape} but expected "
                    f"{self._pos.shape}"
                )

            if k in self.label_errs:
                e = self.label_errs[k]
                if self._pos.shape != e.shape:
                    raise ValueError(
                        "Input label errors must have the same shape as the input "
                        f"input labels: label err {k} has shape {e.shape} but expected "
                        f"{v.shape}"
                    )

    def __getitem__(self, slc):
        return OTIData(
            self._pos[slc],
            self._vel[slc],
            units=self.units,
            labels={k: v[slc] for k, v in self.labels.items()},
            label_errs={k: v[slc] for k, v in self.label_errs.items()},
        )

    def __len__(self):
        return len(self._pos)

    @property
    def pos(self):
        return self._pos * self.units["length"]

    @property
    def vel(self):
        return self._vel * self.units["length"] / self.units["time"]

    def _get_bins_tuple(self, bins):
        if isinstance(bins, dict):
            bins = (bins["pos"], bins["vel"])

        bins = [
            b.decompose(self.units).value if hasattr(b, "unit") else b for b in bins
        ]
        return bins

    def get_binned_counts(self, bins):
        """
        Parameters
        ----------
        bins : tuple, dict
            A specification of the bins. This can either be a tuple, where the order
            is assumed to be (pos, vel), or a dictionary with keys "pos" and "vel".
        """

        H, xe, ye = np.histogram2d(
            self._pos,
            self._vel,
            bins=self._get_bins_tuple(bins),
        )
        xc = 0.5 * (xe[:-1] + xe[1:])
        yc = 0.5 * (ye[:-1] + ye[1:])
        xc, yc = np.meshgrid(xc, yc)

        return {
            "pos": xc * self.units["length"],
            "vel": yc * self.units["length"] / self.units["time"],
            "counts": H.T,
        }

    def get_binned_label(
        self,
        bins,
        label_name=None,
    ):
        """
        Parameters
        ----------
        bins : tuple, dict
            A specification of the bins. This can either be a tuple, where the order
            is assumed to be (pos, vel), or a dictionary with keys "pos" and "vel".
        """
        if label_name is None and len(self._label_names) == 1:
            label_name = self._label_names[0]

        if label_name not in self.labels:
            raise KeyError(
                f"Invalid label name '{label_name}' – expected one of "
                f"{self._label_names}"
            )

        bins = self._get_bins_tuple(bins)
        xc = 0.5 * (bins[0][:-1] + bins[0][1:])
        yc = 0.5 * (bins[1][:-1] + bins[1][1:])
        xc, yc = np.meshgrid(xc, yc)

        bdata = {
            "pos": xc * self.units["length"],
            "vel": yc * self.units["length"] / self.units["time"],
        }

        label_data = self.labels[label_name]
        label_err = self.label_errs.get(label_name)

        if label_err is None:
            # No label errors provided: uncertainty on the mean is related to the
            # intrinsic scatter
            stat_mean = binned_statistic_2d(
                self._pos, self._vel, label_data, bins=bins, statistic="mean"
            )
            mean = stat_mean.statistic

            stat_err = binned_statistic_2d(
                self._pos,
                self._vel,
                label_data,
                bins=bins,
                statistic=lambda x: np.nanstd(x) / np.sqrt(len(x)),
            )
            err = stat_err.statistic

        else:
            # Label errors provided:
            stat_mean1 = binned_statistic_2d(
                self._pos,
                self._vel,
                label_data / label_err**2,
                bins=bins,
                statistic="sum",
            )
            stat_mean2 = binned_statistic_2d(
                self._pos, self._vel, 1 / label_err**2, bins=bins, statistic="sum"
            )
            mean = stat_mean1.statistic / stat_mean2.statistic
            err = np.sqrt(1 / stat_mean2.statistic)

        bdata[label_name] = mean.T
        bdata[f"{label_name}_err"] = err.T

        return bdata, label_name
