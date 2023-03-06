import copy

import pytest

from empaf.model import DensityOrbitModel, _e_param_names

models = []
states = []

model = DensityOrbitModel(e_signs={2: 1.0, 4: -1.0})
models.append(model)

valid_state = {}
valid_state["e_params"] = {m: {k: 1.0 for k in _e_param_names} for m in [2, 4]}
valid_state["ln_dens_params"] = {"f0": 1.0, "f3": -1, "alpha": 0.5, "x0": 1.0}
valid_state["Omega"] = 0.1
valid_state["z0"] = 0.0
valid_state["vz0"] = 0.0
states.append(valid_state)


@pytest.mark.parametrize(["model", "valid_state"], zip(models, states))
def test_density_model_validate(model, valid_state):
    # This should work:
    model.state = valid_state
    model._validate_state()

    # These should fail:
    for k in valid_state:
        tmp_state = valid_state.copy()
        tmp_state.pop(k)

        model.state = tmp_state
        with pytest.raises(RuntimeError):
            model._validate_state()

    # These should also fail: remove a single key from the "e_params" sub-dictionaries
    for m in valid_state["e_params"]:
        tmp_state = copy.deepcopy(valid_state)
        for sub_k in valid_state["e_params"][m].keys():
            tmp_state["e_params"][m].pop(sub_k)
            model.state = tmp_state
            with pytest.raises(RuntimeError):
                model._validate_state()

    # These should also fail: remove a single key from the "*_params" dictionaries
    name = f"{model.fit_name}_params"
    for m in valid_state[name]:
        tmp_state = copy.deepcopy(valid_state)
        for sub_k in valid_state[name].keys():
            tmp_state[name].pop(sub_k)
            model.state = tmp_state
            with pytest.raises(RuntimeError):
                model._validate_state()


@pytest.mark.parametrize(["model", "valid_state"], zip(models, states))
def test_density_model_get_params(model, valid_state):

    model = model.copy()
    model.state = valid_state
    model.get_params()
