"""Canonical operator placement around optional locus tails."""

import pytest

from imas_standard_names.grammar.parser import parse
from imas_standard_names.grammar.render import compose


@pytest.mark.parametrize(
    ("current", "canonical"),
    [
        (
            "root_mean_square_of_wave_current_of_antenna_strap",
            "wave_current_root_mean_square_of_antenna_strap",
        ),
        (
            "voltage_of_ion_cyclotron_heating_antenna_amplitude",
            "voltage_amplitude_of_ion_cyclotron_heating_antenna",
        ),
        (
            "time_derivative_of_pressure_at_magnetic_axis",
            "pressure_time_derivative_at_magnetic_axis",
        ),
        (
            "maximum_of_energy_flux_at_first_wall",
            "energy_flux_maximum_at_first_wall",
        ),
        (
            "maximum_of_energy_flux_at_divertor_target",
            "energy_flux_maximum_at_divertor_target",
        ),
        (
            "minimum_of_energy_flux_at_first_wall",
            "energy_flux_minimum_at_first_wall",
        ),
        (
            "maximum_of_heat_flux_at_divertor_target",
            "heat_flux_maximum_at_divertor_target",
        ),
        (
            "maximum_of_particle_flux_at_first_wall",
            "particle_flux_maximum_at_first_wall",
        ),
        (
            "magnetic_field_of_iron_core_segment_magnitude",
            "magnetic_field_magnitude_of_iron_core_segment",
        ),
        (
            "velocity_due_to_pellet_injection_magnitude",
            "velocity_magnitude_due_to_pellet_injection",
        ),
        (
            "maximum_magnetic_field_of_flux_surface_magnitude",
            "maximum_magnetic_field_magnitude_of_flux_surface",
        ),
        (
            "maximum_of_inverse_of_electron_density_due_to_non_inductive_current_drive",
            "electron_density_inverse_maximum_due_to_non_inductive_current_drive",
        ),
        (
            "maximum_of_inverse_of_electron_density_at_magnetic_axis",
            "electron_density_inverse_maximum_at_magnetic_axis",
        ),
        (
            "root_mean_square_of_radial_electron_pressure_at_plasma_boundary",
            "radial_electron_pressure_root_mean_square_at_plasma_boundary",
        ),
        (
            "maximum_of_derivative_of_pressure_at_pedestal",
            "pressure_derivative_maximum_at_pedestal",
        ),
    ],
)
def test_tailed_operator_moves_between_base_and_locus(
    current: str, canonical: str
) -> None:
    ir = parse(current).ir

    assert compose(ir) == canonical
    assert parse(canonical, strict=True).ir == ir


# A bare, joiner-free operator is exempt from the relocation above: it leads
# its name and its operand runs to the end of the string, tail included. The
# spelling on the left of each pair states the narrower scope and is rejected.
@pytest.mark.parametrize(
    ("misplaced", "canonical"),
    [
        (
            "electron_density_flux_surface_averaged_at_plasma_boundary",
            "flux_surface_averaged_electron_density_at_plasma_boundary",
        ),
        (
            "electron_temperature_flux_surface_averaged_at_plasma_boundary",
            "flux_surface_averaged_electron_temperature_at_plasma_boundary",
        ),
        (
            "toroidal_argon_velocity_flux_surface_averaged_at_plasma_boundary",
            "flux_surface_averaged_toroidal_argon_velocity_at_plasma_boundary",
        ),
    ],
)
def test_bare_operator_leads_instead_of_moving_before_the_locus(
    misplaced: str, canonical: str
) -> None:
    assert compose(parse(canonical, strict=True).ir) == canonical
    assert parse(misplaced).ir == parse(canonical).ir
    assert compose(parse(misplaced).ir) == canonical
    with pytest.raises(ValueError, match="not canonical") as rejection:
        parse(misplaced, strict=True)
    assert canonical in str(rejection.value)


def test_an_outer_operator_cannot_move_past_a_leading_reduction() -> None:
    """A joiner-taking operator stays put when a reduction leads its operand.

    The reduction is innermost either way, so the trailing spelling denotes the
    same quantity and converges on its representation; only the spelling is
    refused, and the rejection names the form to migrate to.
    """
    canonical = "maximum_of_volume_averaged_electron_density_at_magnetic_axis"
    misplaced = "electron_density_volume_averaged_maximum_at_magnetic_axis"
    assert compose(parse(canonical, strict=True).ir) == canonical
    assert parse(misplaced).ir == parse(canonical).ir
    assert compose(parse(misplaced).ir) == canonical
    with pytest.raises(ValueError, match="not canonical") as rejection:
        parse(misplaced, strict=True)
    assert canonical in str(rejection.value)


@pytest.mark.parametrize(
    "name",
    [
        "accumulated_ammonia_count",
        (
            "flux_surface_averaged_ratio_of_square_of_"
            "toroidal_flux_coordinate_gradient_magnitude_"
            "to_square_of_major_radius"
        ),
    ],
)
def test_operator_without_repositioning_tail_keeps_natural_reading(name: str) -> None:
    ir = parse(name, strict=True).ir

    assert compose(ir) == name
    assert parse(compose(ir), strict=True).ir == ir


def test_operator_in_composite_operand_keeps_natural_reading() -> None:
    name = "ratio_of_pressure_to_square_of_temperature_at_magnetic_axis"
    ir = parse(name, strict=True).ir

    assert compose(ir) == name
    assert parse(compose(ir), strict=True).ir == ir
