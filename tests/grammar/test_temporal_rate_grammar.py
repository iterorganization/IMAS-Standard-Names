"""Per-time spellings: one rate base, one denominator guard, one operator."""

import pytest

from imas_standard_names import parse
from imas_standard_names.grammar.loader import load_advisory_aliases
from imas_standard_names.grammar.model import (
    compose_standard_name,
    parse_standard_name,
)

_MASS_FLOW_RATE_NAME = "coolant_mass_flow_rate"
_TIME_DENOMINATOR_NAME = "ratio_of_coolant_mass_to_time"


def test_mass_flow_rate_base_composes_and_round_trips() -> None:
    """A kilogram-per-second mass throughput has a base to land on."""
    parsed = parse_standard_name(_MASS_FLOW_RATE_NAME)
    ir = parse(_MASS_FLOW_RATE_NAME, strict=True).ir

    assert ir.base.token == "mass_flow_rate"
    assert [qualifier.token for qualifier in ir.qualifiers] == ["coolant"]
    assert compose_standard_name(parsed) == _MASS_FLOW_RATE_NAME


def test_ratio_over_time_is_refused_with_the_alternatives_named() -> None:
    """Dividing by elapsed time is a temporal derivative, not a ratio."""
    with pytest.raises(ValueError) as excinfo:
        parse(_TIME_DENOMINATOR_NAME, strict=True)

    message = str(excinfo.value)
    assert "temporal derivative" in message
    for alternative in (
        "time_derivative",
        "change_in",
        "mass_flow_rate",
        "source_rate",
    ):
        assert alternative in message


def test_ratio_over_time_is_refused_when_nested_in_an_outer_operator() -> None:
    """The guard reads the ratio wherever it sits in the ordered expression."""
    with pytest.raises(ValueError, match="temporal derivative"):
        parse(f"volume_averaged_{_TIME_DENOMINATOR_NAME}", strict=True)


def test_ratio_of_two_quantities_still_parses() -> None:
    """Only a time denominator is refused; ordinary ratios are untouched."""
    name = "ratio_of_electron_density_to_ion_density"

    assert parse(name, strict=True).ir.operators[0].op == "ratio"


def test_tendency_resolves_through_the_alias_to_time_derivative() -> None:
    """The retired spelling is guidance only and its canonical form parses."""
    alias = load_advisory_aliases()["transformation"]["tendency"]
    assert alias["canonical"] == "time_derivative"

    retired = "tendency_of_electron_density"
    with pytest.raises(ValueError):
        parse(retired, strict=True)

    canonical = retired.replace("tendency", alias["canonical"], 1)
    assert parse(canonical, strict=True).ir.operators[0].op == "time_derivative"
    assert compose_standard_name(parse_standard_name(canonical)) == canonical
