import astropy.units as u
import numpy as np
from gala.units import galactic

import torusimaging as oti


def harmonic_oscillator():
    """Make test data in a Harmonic oscillator potential"""
    Omega = 0.08 * u.rad / u.Myr
    N = 200_000
    scale_vz = 50 * u.km / u.s
    rng = np.random.default_rng(42)

    sz = (scale_vz / np.sqrt(Omega)).decompose(galactic)

    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        Jzs = (rng.exponential(scale=sz.value**2, size=N) * sz.unit**2).to(
            galactic["length"] ** 2 / galactic["time"]
        )
        thzs = rng.uniform(0, 2 * np.pi, size=N) * u.rad

    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        pdata = {
            "z": (np.sqrt(2 * Jzs / Omega) * np.sin(thzs)).to(galactic["length"]),
            "vz": (np.sqrt(2 * Jzs * Omega) * np.cos(thzs)).to(
                galactic["length"] / galactic["time"]
            ),
            "Jz": Jzs,
            "thetaz": thzs,
        }
        pdata["r_e"] = np.sqrt(pdata["z"] ** 2 * Omega + pdata["vz"] ** 2 / Omega)
        pdata["mgfe"] = rng.normal(
            np.sqrt(0.15) * pdata["Jz"].value ** 0.5 + 0.025, 0.04
        )
        pdata["mgfe_err"] = np.exp(rng.normal(-4.0, 0.5, size=N))
        pdata["mgfe"] = rng.normal(pdata["mgfe"], pdata["mgfe_err"])

    # Bin the particle data to make 2D arrays to save for tests:
    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        vzlim = [-120, 120] * u.km / u.s
        zlim = (vzlim / Omega).to(u.kpc)
    Nbins = 128
    bins = {
        "vel": np.linspace(*vzlim.value, Nbins) * vzlim.unit,
        "pos": np.linspace(*zlim.value, Nbins) * zlim.unit,
    }

    # Make a 2D histogram of the data:
    bdata_label = oti.data.get_binned_label(
        pdata["z"],
        pdata["vz"],
        pdata["mgfe"],
        label_err=pdata["mgfe_err"],
        bins=bins,
        s_N_thresh=32,
        units=galactic,
    )
    test_data_label = {
        "pos": bdata_label["pos"].decompose(galactic).value,
        "vel": bdata_label["vel"].decompose(galactic).value,
        "label": bdata_label["label"],
        "label_err": bdata_label["label_err"],
        "counts": bdata_label["counts"],
    }

    np.savez("sho_test_data.npz", **test_data_label)


if __name__ == "__main__":
    harmonic_oscillator()
