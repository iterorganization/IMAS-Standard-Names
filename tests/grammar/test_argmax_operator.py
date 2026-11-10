"""Indexed argmax locates a profile maximum in its coordinate's units."""

import pytest
from pydantic import ValidationError

from imas_standard_names import compose, parse, validate_round_trip
from imas_standard_names.grammar.ir import OperatorKind
from imas_standard_names.grammar.model import (
    compose_standard_name,
    parse_standard_name,
)
from imas_standard_names.grammar.parser import ParseError
from imas_standard_names.grammar.vocab_loaders import OperatorDef, load_operators
from imas_standard_names.models import StandardNameScalarEntry
from imas_standard_names.validation import run_semantic_checks

COORDINATE = "normalized_toroidal_flux_coordinate"
OPERAND = "deposited_power"
ARGMAX = f"argmax_of_{OPERAND}_with_respect_to_{COORDINATE}"


def _entry(name: str, unit: str) -> StandardNameScalarEntry:
    return StandardNameScalarEntry(
        name=name,
        unit=unit,
        physics_domain="general",
        description="Coordinate of the profile maximum.",
        documentation="The profile maximum is located on the named coordinate.",
    )


def test_argmax_binds_coordinate_in_operator_token() -> None:
    ir = parse(ARGMAX, strict=True).ir
    operator = ir.operators[0]
    assert len(ir.operators) == 1
    assert operator.kind is OperatorKind.UNARY_PREFIX
    assert operator.op == f"argmax_with_respect_to_{COORDINATE}"
    assert operator.args == []
    assert ir.base.token == "power"
    assert [qualifier.token for qualifier in ir.qualifiers] == ["deposited"]
    assert compose(ir) == ARGMAX
    assert validate_round_trip(ARGMAX)


def test_argmax_flat_facade_round_trips() -> None:
    flat = parse_standard_name(ARGMAX)
    assert flat.transformation == f"argmax_with_respect_to_{COORDINATE}"
    assert compose_standard_name(flat) == ARGMAX


@pytest.mark.parametrize(
    "name",
    ["argmax_of_deposited_power", "argmax_of_deposited_power_with_respect_to_banana"],
)
def test_argmax_requires_registered_coordinate(name: str) -> None:
    with pytest.raises(ParseError):
        parse(name, strict=True)


def test_argmax_registry_declares_index_dimension() -> None:
    definition = load_operators().operators["argmax_with_respect_to"]
    assert definition.kind == "unary_prefix"
    assert definition.indexed is True
    assert definition.index_params == ["coord"]
    assert definition.dimension_transforming == "index"
    assert not definition.dimensionless
    assert not definition.semantic_effects


@pytest.mark.parametrize("invalid", [True, False, "coordinate", "arbitrary"])
def test_dimension_transforming_rejects_untyped_values(invalid: object) -> None:
    with pytest.raises(ValidationError, match="dimension_transforming"):
        OperatorDef(kind="unary_prefix", precedence=20, dimension_transforming=invalid)


def test_dimension_preserving_operator_has_no_dimension_transform() -> None:
    assert (
        OperatorDef(kind="unary_prefix", precedence=20).dimension_transforming is None
    )


def test_argmax_accepts_matching_coordinate_unit() -> None:
    entries = {
        OPERAND: _entry(OPERAND, "W"),
        COORDINATE: _entry(COORDINATE, "1"),
        ARGMAX: _entry(ARGMAX, "1"),
    }
    assert run_semantic_checks(entries) == []


def test_argmax_rejects_operand_unit() -> None:
    entries = {
        OPERAND: _entry(OPERAND, "W"),
        COORDINATE: _entry(COORDINATE, "1"),
        ARGMAX: _entry(ARGMAX, "W"),
    }
    issues = run_semantic_checks(entries)
    assert len(issues) == 1
    assert ARGMAX in issues[0]
    assert COORDINATE in issues[0]
    assert "unit 'W'" in issues[0]
    assert "unit '1'" in issues[0]
    assert "WARNING" in issues[0]


def test_argmax_reports_missing_coordinate_entry() -> None:
    entries = {OPERAND: _entry(OPERAND, "W"), ARGMAX: _entry(ARGMAX, "1")}
    issues = run_semantic_checks(entries)
    assert len(issues) == 1
    assert ARGMAX in issues[0]
    assert f"non-existent standard name '{COORDINATE}'" in issues[0]
    assert "WARNING" in issues[0]


def test_semantics_defers_invalid_index_spelling_to_structural_validation() -> None:
    name = "argmax_of_deposited_power_with_respect_to_banana"
    entry = StandardNameScalarEntry.model_construct(name=name, unit="1")
    assert run_semantic_checks({name: entry}) == []


def test_argmax_uses_dimensional_coordinate_unit() -> None:
    coordinate = "toroidal_flux_coordinate"
    name = f"argmax_of_{OPERAND}_with_respect_to_{coordinate}"
    entries = {
        OPERAND: _entry(OPERAND, "W"),
        coordinate: _entry(coordinate, "Wb"),
        name: _entry(name, "Wb"),
    }
    assert run_semantic_checks(entries) == []


def test_outer_dimension_transform_does_not_compare_inner_argmax_unit() -> None:
    name = f"square_of_{ARGMAX.replace(COORDINATE, 'toroidal_flux_coordinate')}"
    entries = {name: _entry(name, "Wb^2")}
    assert run_semantic_checks(entries) == []
