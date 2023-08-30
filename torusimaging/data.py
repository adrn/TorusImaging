import astropy.units as u
import numpy as np
from gala.units import galactic
from scipy.stats import binned_statistic_2d


class OTIData:
    def __init__(self, pos, vel, units=galactic, **labels):
        self.units = units
        self._label_names = list(labels.keys())

        # For now, we only support 1D pos and vel:
        self._pos = u.Quantity(pos).decompose(galactic).value
        self._vel = u.Quantity(vel).decompose(galactic).value
        self.labels = {k: np.asarray(v) for k, v in labels.items()}

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

    def __getitem__(self, slc):
        return OTIData(
            self._pos[slc],
            self._vel[slc],
            units=self.units,
            **{k: v[slc] for k, v in self.labels.items()},
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

    def get_binned_label(self, bins, label_name=None, **binned_statistic_kwargs):
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

        stat = binned_statistic_2d(
            self._pos,
            self._vel,
            self.labels[label_name],
            bins=self._get_bins_tuple(bins),
            **binned_statistic_kwargs,
        )
        xc = 0.5 * (stat.x_edge[:-1] + stat.x_edge[1:])
        yc = 0.5 * (stat.y_edge[:-1] + stat.y_edge[1:])
        xc, yc = np.meshgrid(xc, yc)

        return {
            "pos": xc * self.units["length"],
            "vel": yc * self.units["length"] / self.units["time"],
            label_name: stat.statistic.T,
        }, label_name

    # TODO: add a way to get a binned label error bars
    # stat_err = binned_statistic_2d(
    #         vz,
    #         z,
    #         label,
    #         bins=(bins["vz"], bins["z"]),
    #         statistic=lambda x: np.sqrt((1.5 * MAD(x)) ** 2 + err_floor**2)
    #         / np.sqrt(len(x)),
    #     )
