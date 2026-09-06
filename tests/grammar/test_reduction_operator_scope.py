"""A domain reduction leads its name and wraps the projection and the tail.

A reduction collapses its operand over a domain — a flux surface, a volume, a
line, the time axis. The reduced object is therefore the whole projected,
qualified quantity, not one axis component of it, so the operator token sits
first in the canonical spelling and everything to its right is its operand.
"""

import pytest

from imas_standard_names import compose, parse
from imas_standard_names.grammar.ir import OperatorKind

CANONICAL = "flux_surface_averaged_toroidal_lithium_velocity_at_plasma_boundary"
PROJECTION_OUTSIDE = (
    "toroidal_flux_surface_averaged_lithium_velocity_at_plasma_boundary"
)
OPERATOR_TRAILING = "toroidal_lithium_velocity_flux_surface_averaged_at_plasma_boundary"

UNTAILED_CONTROL = "flux_surface_averaged_magnetic_field_magnitude"
RECURSIVE_CONTROL = (
    "flux_surface_averaged_ratio_of_square_of_toroidal_flux_coordinate_gradient"
    "_magnitude_to_square_of_major_radius"
)


def test_reduction_is_an_operator_not_a_qualifier():
    ir = parse(CANONICAL, strict=True).ir
    assert [operator.op for operator in ir.operators] == ["flux_surface_averaged"]
    assert ir.operators[0].kind is OperatorKind.UNARY_PREFIX
    assert ir.operators[0].bare_prefix is True
    assert [qualifier.token for qualifier in ir.qualifiers] == ["lithium"]
    assert ir.projection is not None
    assert ir.projection.axis == "toroidal"
    assert ir.locus is not None
    assert ir.locus.token == "plasma_boundary"


def test_reduction_leads_the_canonical_spelling():
    assert compose(parse(CANONICAL).ir) == CANONICAL


def test_canonical_spelling_round_trips_under_strict_parsing():
    assert compose(parse(CANONICAL, strict=True).ir) == CANONICAL


@pytest.mark.parametrize("name", [PROJECTION_OUTSIDE, OPERATOR_TRAILING])
def test_misplaced_reduction_is_rejected_as_non_canonical(name):
    parse(name)  # parses; only the spelling is wrong
    with pytest.raises(ValueError, match="not canonical"):
        parse(name, strict=True)


@pytest.mark.parametrize("name", [PROJECTION_OUTSIDE, OPERATOR_TRAILING])
def test_every_spelling_of_one_quantity_shares_one_representation(name):
    """Representation equality is the meaning test, so a rename between these
    spellings must read as meaning-preserving rather than meaning-changing."""
    assert parse(name).ir == parse(CANONICAL).ir


@pytest.mark.parametrize("name", [PROJECTION_OUTSIDE, OPERATOR_TRAILING])
def test_a_misplaced_reduction_composes_to_the_canonical_spelling(name):
    assert compose(parse(name).ir) == CANONICAL


@pytest.mark.parametrize("name", [PROJECTION_OUTSIDE, OPERATOR_TRAILING])
def test_misplaced_reduction_names_the_canonical_spelling(name):
    with pytest.raises(ValueError) as excinfo:
        parse(name, strict=True)
    assert CANONICAL in str(excinfo.value)


@pytest.mark.parametrize("name", [UNTAILED_CONTROL, RECURSIVE_CONTROL])
def test_controls_are_byte_identical(name):
    assert compose(parse(name, strict=True).ir) == name


def test_reduction_wraps_a_bare_tail_without_a_projection():
    name = "flux_surface_averaged_lithium_velocity_at_plasma_boundary"
    ir = parse(name, strict=True).ir
    assert [operator.op for operator in ir.operators] == ["flux_surface_averaged"]
    assert compose(ir) == name


def test_reduction_wraps_a_projection_without_a_tail():
    name = "flux_surface_averaged_toroidal_lithium_velocity"
    ir = parse(name, strict=True).ir
    assert [operator.op for operator in ir.operators] == ["flux_surface_averaged"]
    assert compose(ir) == name


def test_a_registered_base_keeps_its_reduction_shaped_prefix():
    """``flux_surface_averaged_metric`` is one base token, not an application."""
    ir = parse("flux_surface_averaged_metric").ir
    assert ir.operators == []
    assert ir.base.token == "flux_surface_averaged_metric"


@pytest.mark.parametrize(
    ("name", "operator"),
    [
        ("volume_averaged_electron_temperature", "volume_averaged"),
        ("line_integrated_electron_density", "line_integrated"),
        ("time_averaged_toroidal_magnetic_field", "time_averaged"),
    ],
)
def test_the_rule_covers_the_whole_reduction_class(name, operator):
    ir = parse(name, strict=True).ir
    assert [application.op for application in ir.operators] == [operator]
    assert compose(ir) == name
