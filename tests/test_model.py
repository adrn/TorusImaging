import pathlib

import jax
import numpy as np
from gala.units import galactic

import torusimaging as oti

jax.config.update("jax_enable_x64", True)


def test_harmonic_oscillator():
    data_path = pathlib.Path("tests/data/sho_test_data.npz").absolute()
    bdata = np.load(data_path)
    bdata = {k: bdata[k] for k in bdata}

    bdata["pos"] *= galactic["length"]
    bdata["vel"] *= galactic["velocity"]

    model, bounds, init_params = oti.TorusImaging1DSpline.auto_init(
        bdata,
        label_knots=8,
        e_knots={2: 8},
        label_l2_sigma=1.0,
        label_smooth_sigma=0.5,
        e_l2_sigmas={2: 0.5},
        e_smooth_sigmas={2: 0.5},
    )
    init_params["ln_Omega0"] = np.log(0.07)

    data_kw = {
        "pos": bdata["pos"],
        "vel": bdata["vel"],
        "label": bdata["label"],
        "label_err": bdata["label_err"],
    }
    mask = (
        np.isfinite(bdata["label"])
        & np.isfinite(bdata["label_err"])
        & (bdata["label_err"] > 0)
    )
    data_kw = {k: v[mask] for k, v in data_kw.items()}

    test_obj = model.objective_gaussian(init_params, **data_kw)
    assert np.isfinite(test_obj)

    init_params["pos0"] = 0.1
    init_params["vel0"] = 0.005

    res = model.optimize(init_params, objective="gaussian", bounds=bounds, **data_kw)
    assert res.state.success
    print(res.params, res.state.iter_num, res.state.success)
    assert np.isclose(res.params["pos0"], 0.0, atol=1e-3)
    assert np.isclose(res.params["vel0"], 0.0, atol=1e-3)
    assert np.isclose(res.params["ln_Omega0"], np.log(0.08), atol=1e-2)
