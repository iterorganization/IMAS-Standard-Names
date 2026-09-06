"""Gate: extremum qualifier embedded as an infix inside a flux base.

An extremum of a transport flux over a spatial domain (peak/maximum value on
a wall, a divertor target, …) is a reduction transformation. Its one canonical
spelling puts the reduction in transformation (prefix) position —
``maximum_of_<channel>_flux_at_<locus>`` / ``minimum_of_...``. The same token
embedded as an infix qualifier inside the flux base (``energy_peak_flux``,
``energy_maximum_flux``) names the same concept a second way and must fail both
parse and compose so the pipeline can never mint it.

The gate is scoped to the flux family (base ``flux`` — energy_flux, heat_flux,
particle_flux, momentum_flux): an extremum infix on any other base (peak
temperature, peak density) is untouched, and the extremum token in
transformation position stays legal.
"""

import pytest

from imas_standard_names.grammar.model import (
    compose_standard_name,
    parse_standard_name,
)

REJECTED = [
    # peak / maximum / minimum embedded as an infix inside a flux base
    "energy_peak_flux",
    "energy_maximum_flux",
    "energy_minimum_flux",
    "energy_peak_flux_at_first_wall",
    "energy_maximum_flux_at_divertor_target",
    "heat_peak_flux",
    "particle_maximum_flux",
    "momentum_peak_flux",
]

ACCEPTED = [
    # The reduction operator precedes the tail in the canonical spelling.
    "energy_flux_maximum_at_first_wall",
    "energy_flux_maximum_at_divertor_target",
    "energy_flux_minimum_at_first_wall",
    "heat_flux_maximum_at_divertor_target",
    "particle_flux_maximum_at_first_wall",
    # (b) a plain flux base is unaffected
    "energy_flux",
    "energy_flux_at_first_wall",
    "heat_flux",
    "particle_flux_at_divertor_target",
    # (c) an extremum infix on a NON-flux base keeps the infix spelling
    "electron_peak_temperature",
]


@pytest.mark.parametrize("name", REJECTED)
def test_extremum_infix_on_flux_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="reduction transformation"):
        parse_standard_name(name)


@pytest.mark.parametrize("name", ACCEPTED)
def test_legal_names_round_trip(name: str) -> None:
    model = parse_standard_name(name)
    assert compose_standard_name(model) == name


def test_compose_rejects_infix_flux_base() -> None:
    # Build the infix physical_base directly and confirm compose refuses it,
    # so the guard holds regardless of how the model was constructed.
    with pytest.raises(ValueError, match="reduction transformation"):
        compose_standard_name({"channel": "energy", "physical_base": "peak_flux"})
    with pytest.raises(ValueError, match="reduction transformation"):
        compose_standard_name({"channel": "energy", "physical_base": "maximum_flux"})
