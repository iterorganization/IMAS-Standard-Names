"""Closed-vocabulary coverage for DD-backed quantity families.

Two accepted additions are intentionally additive, while the strain-gauge
candidate is rejected on positive evidence:

* Rejected: ``measurement_direction_unit_vector``. Accepted alternatives:
  ``first_measurement_direction_unit_vector`` for ``sensor/direction/{x,y,z}``
  and ``second_measurement_direction_unit_vector`` for
  ``sensor/direction_second/{x,y,z}``. The plain path is the primary member of
  the pair because the companion is explicitly ``direction_second``. Collapsing
  the two would erase which distinct strain-rosette measurement axis a vector
  describes. Generic ``direction_unit_vector`` names any device direction and
  carries no strain-rosette measurement-axis role; the first and second carriers
  identify the two ordered measurement axes. The six leaves report DD unit
  ``m`` despite their parent documentation defining a unit vector. That is a DD
  unit defect; it does not turn an orientation carrier into a dimensional
  physical quantity.
* ``s0`` through ``s3`` project the existing ``stokes_parameter`` base. The
  eight FOCS leaves explicitly document the four indices as initial and output
  Stokes-vector components; Stokes coordinates are dimensionless (unit ``1``).
* ``relative_humidity`` is the dimensionless fraction of humidity from zero to
  one measured at the X-ray detector, as documented for
  ``camera_x_rays/detector_humidity`` (unit ``1``).
The shipped catalog snapshot contains none of the representative identities for
the two accepted additions or an equivalent token in those families.
Unindexed ``stokes_parameter`` erases which polarization coordinate is stored;
generic fractions do not say relative humidity.
"""

from dataclasses import replace

import pytest

from imas_standard_names import ParseError, compose, parse
from imas_standard_names.grammar.parser import load_default_vocabularies


@pytest.mark.parametrize(
    ("registry", "tokens"),
    [
        ("component_axes", {"s0", "s1", "s2", "s3"}),
        ("bases", {"relative_humidity"}),
    ],
)
def test_dd_backed_tokens_are_registered(registry: str, tokens: set[str]) -> None:
    vocabularies = load_default_vocabularies()
    assert tokens <= getattr(vocabularies, registry)


@pytest.mark.parametrize(
    "name",
    [
        "s0_stokes_parameter_of_fiber_optic_current_sensor",
        "s1_stokes_parameter_of_fiber_optic_current_sensor",
        "s2_stokes_parameter_of_fiber_optic_current_sensor",
        "s3_stokes_parameter_of_fiber_optic_current_sensor",
        "relative_humidity_of_detector",
    ],
)
def test_dd_backed_vocabulary_name_round_trips_through_render(name: str) -> None:
    result = parse(name, strict=True)
    assert compose(result.ir) == name


@pytest.mark.parametrize(
    ("registry", "token", "name"),
    [
        (
            "component_axes",
            "s0",
            "s0_stokes_parameter_of_fiber_optic_current_sensor",
        ),
        (
            "component_axes",
            "s1",
            "s1_stokes_parameter_of_fiber_optic_current_sensor",
        ),
        (
            "component_axes",
            "s2",
            "s2_stokes_parameter_of_fiber_optic_current_sensor",
        ),
        (
            "component_axes",
            "s3",
            "s3_stokes_parameter_of_fiber_optic_current_sensor",
        ),
        ("bases", "relative_humidity", "relative_humidity_of_detector"),
    ],
)
def test_dd_backed_name_fails_when_its_token_is_removed(
    registry: str, token: str, name: str
) -> None:
    vocabularies = load_default_vocabularies()
    reduced_tokens = getattr(vocabularies, registry) - {token}
    reduced_vocabularies = replace(vocabularies, **{registry: reduced_tokens})

    with pytest.raises(ParseError):
        parse(name, vocabs=reduced_vocabularies, strict=True)


@pytest.mark.parametrize(
    "name",
    [
        "x_first_measurement_direction_unit_vector_of_strain_gauge",
        "y_first_measurement_direction_unit_vector_of_strain_gauge",
        "z_first_measurement_direction_unit_vector_of_strain_gauge",
        "x_second_measurement_direction_unit_vector_of_strain_gauge",
        "y_second_measurement_direction_unit_vector_of_strain_gauge",
        "z_second_measurement_direction_unit_vector_of_strain_gauge",
    ],
)
def test_strain_gauge_uses_the_existing_ordered_carrier_pair(name: str) -> None:
    result = parse(name, strict=True)
    assert compose(result.ir) == name


def test_nonordinal_strain_gauge_carrier_is_rejected() -> None:
    with pytest.raises(ParseError):
        parse("x_measurement_direction_unit_vector_of_strain_gauge", strict=True)
