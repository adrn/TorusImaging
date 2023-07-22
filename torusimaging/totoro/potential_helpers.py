# Third-party
import agama
import astropy.units as u
import gala.potential as gp
import numpy as np
from gala.units import galactic

# TODO: move to Gala, set up with external dependencies?
HAS_AGAMA = True

###############################################################################
# Agama interoperability
#

if HAS_AGAMA:
    _gala_to_agama = {
        gp.HernquistPotential: (
            "dehnen",
            {"mass": "m", "gamma": 1.0, "scaleRadius": "c"},
        ),
        gp.JaffePotential: (
            "dehnen",
            {"mass": "m", "gamma": 2.0, "scaleRadius": "c"},
        ),
        gp.NFWPotential: ("nfw", {"mass": "m", "scaleRadius": "r_s"}),
        gp.MiyamotoNagaiPotential: (
            "miyamotonagai",
            {"mass": "m", "scaleRadius": "a", "scaleHeight": "b"},
        ),
        gp.PlummerPotential: ("plummer", {"mass": "m", "scaleRadius": "b"}),
        gp.IsochronePotential: ("isochrone", {"mass": "m", "scaleRadius": "b"}),
    }


def gala_to_agama_potential(potential):

    if not HAS_AGAMA:
        raise ImportError(
            "Failed to import agama: Converting a potential to an "
            "agama potential requires agama to be installed."
        )

    agama.setUnits(
        length=potential.units["length"].to(u.kpc),
        mass=potential.units["mass"].to(u.Msun),
        time=potential.units["time"].to(u.Myr),
    )

    if isinstance(potential, gp.CompositePotential):
        # TODO:
        pot = []
        for k in potential.keys():
            pot.append(gala_to_agama_potential(potential[k]))
        pot = agama.Potential(*pot)

    else:
        if potential.__class__ not in _gala_to_agama:
            raise TypeError(
                f"Converting potential class {potential.__class__.__name__} "
                "to agama is currently not supported"
            )

        potential_type, converters = _gala_to_agama[potential.__class__]
        gala_pars = potential.parameters.copy()

        agama_pars = {}
        for agama_par_name, conv in converters.items():
            if isinstance(conv, str):
                agama_pars[agama_par_name] = gala_pars[conv]
            elif hasattr(conv, "__call__"):
                agama_pars[agama_par_name] = conv(gala_pars)
            elif isinstance(conv, (int, float, u.Quantity, np.ndarray)):
                agama_pars[agama_par_name] = conv
            else:
                # TODO: invalid parameter??
                print(f"FAIL: {agama_par_name}, {conv}")

            par = agama_pars[agama_par_name]
            if hasattr(par, "unit"):
                agama_pars[agama_par_name] = par.decompose(galactic).value

        pot = agama.Potential(type=potential_type, **agama_pars)

    return pot
