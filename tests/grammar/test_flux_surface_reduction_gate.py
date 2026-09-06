"""Gate: flux-surface reduction operators on flux-function bases.

A base flagged ``constant_on_flux_surface`` (a flux function — safety
factor, magnetic shear, flux labels, pressure) is constant on any flux
surface, so a flux-surface reduction (``flux_surface_averaged``,
``maximum_over_flux_surface``, ``minimum_over_flux_surface``) is a no-op
applied to it: the reduction of a flux function is the value itself.
Such names must fail both parse and compose so the pipeline can never
mint them; the local and averaged DD leaves share one name instead.

Volume/line averages of a flux function are not no-ops and stay legal.
"""

import pytest

from imas_standard_names.grammar.model import (
    compose_standard_name,
    parse_standard_name,
)

REJECTED = [
    # transformation slot (bare prefix qualifier form)
    "flux_surface_averaged_safety_factor",
    "flux_surface_averaged_safety_factor_at_plasma_boundary",
    "flux_surface_averaged_magnetic_shear",
    "flux_surface_averaged_magnetic_shear_at_plasma_boundary",
    "flux_surface_averaged_plasma_pressure_at_plasma_boundary",
    "flux_surface_averaged_poloidal_magnetic_flux",
    # geometry-carrier flux labels
    "flux_surface_averaged_toroidal_flux_coordinate",
    # max/min over a flux surface of a flux function is equally degenerate
    "maximum_over_flux_surface_safety_factor",
    "minimum_over_flux_surface_magnetic_shear",
]

ACCEPTED = [
    # Surface-varying bases keep the reduction immediately before the tail.
    "electron_density_flux_surface_averaged_at_plasma_boundary",
    "electron_temperature_flux_surface_averaged_at_plasma_boundary",
    "toroidal_argon_velocity_flux_surface_averaged_at_plasma_boundary",
    "maximum_over_flux_surface_electron_temperature",
    # non-flux-surface reductions of flux functions remain legal
    "volume_averaged_plasma_pressure",
    "time_derivative_of_safety_factor",
    # bare flux-function names are untouched
    "safety_factor_at_plasma_boundary",
    "magnetic_shear_at_plasma_boundary",
    "poloidal_magnetic_flux_at_plasma_boundary",
]


@pytest.mark.parametrize("name", REJECTED)
def test_flux_surface_reduction_on_flux_function_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="constant on a flux surface"):
        parse_standard_name(name)


@pytest.mark.parametrize("name", ACCEPTED)
def test_legal_names_round_trip(name: str) -> None:
    model = parse_standard_name(name)
    assert compose_standard_name(model) == name


def test_compose_rejects_flagged_combination() -> None:
    model = parse_standard_name("safety_factor_at_plasma_boundary")
    data = model.model_dump(exclude_none=True, exclude_defaults=True)
    data["transformation"] = "flux_surface_averaged"
    with pytest.raises(ValueError, match="constant on a flux surface"):
        compose_standard_name(data)
