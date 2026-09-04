"""The per-solid-angle spelling: one operator token, every quantity.

A volumetric emission rate resolved into direction carries sr^-1 in its unit
(m^-3.s^-1.sr^-1). ``per_solid_angle`` is what carries that steradian in the
name, so the unit string is not the only place the per-solid-angle character
survives, and it composes onto any quantity rather than needing a base per
emitted species.
"""

import pytest

from imas_standard_names import compose, parse
from imas_standard_names.grammar import parse_name
from imas_standard_names.grammar.vocab_loaders import load_operators

_PER_SOLID_ANGLE_NAMES = [
    "source_rate_per_solid_angle",
    "hard_xray_source_rate_per_solid_angle",
    "electron_energy_source_rate_per_solid_angle",
    "radiance_per_solid_angle",
]


@pytest.mark.parametrize("name", _PER_SOLID_ANGLE_NAMES)
def test_per_solid_angle_names_parse_strictly(name: str) -> None:
    parse(name, strict=True)


@pytest.mark.parametrize("name", _PER_SOLID_ANGLE_NAMES)
def test_per_solid_angle_names_round_trip(name: str) -> None:
    assert compose(parse(name, strict=True).ir) == name


@pytest.mark.parametrize("name", _PER_SOLID_ANGLE_NAMES)
def test_the_operator_leaves_the_base_intact(name: str) -> None:
    """The token normalizes the quantity; it never becomes part of the base."""
    without = name.removesuffix("_per_solid_angle")
    assert parse_name(name).physical_base == parse_name(without).physical_base


def test_the_token_is_a_dimension_transforming_postfix_operator() -> None:
    """sr^-1 joins the unit, so base-implies-unit inference must be suppressed."""
    operator = load_operators().operators["per_solid_angle"]

    assert operator.kind == "unary_postfix"
    assert operator.dimension_transforming is True


def test_one_token_serves_every_emitted_quantity() -> None:
    """No base spelled '<base>_per_solid_angle' — the operator replaces them."""
    from imas_standard_names.grammar.vocab_loaders import load_physical_bases

    assert not [
        token
        for token in load_physical_bases().bases
        if token.endswith("_per_solid_angle") or token.endswith("_per_steradian")
    ]
