"""Equivalent-electron counting composes with species and quantity qualifiers."""

import pytest

from imas_standard_names import compose, parse
from imas_standard_names.grammar.model import (
    StandardName,
    compose_standard_name,
    parse_standard_name,
)
from imas_standard_names.grammar.model_types import Qualifier
from imas_standard_names.grammar.parser import ParseError
from imas_standard_names.grammar.terms import select_conventions
from imas_standard_names.grammar.vocab_loaders import load_concepts

NAMES = [
    "accumulated_argon_equivalent_electron_count",
    "accumulated_ammonia_equivalent_electron_count",
    "neon_prefill_equivalent_electron_count",
    "accumulated_oxygen_prefill_equivalent_electron_count_due_to_gas_injection",
    "lithium_equivalent_electron_source_rate_due_to_gas_injection",
    "total_equivalent_electron_source_rate_at_wall_due_to_gas_injection",
]


@pytest.mark.parametrize("name", NAMES)
def test_equivalent_electron_is_a_distinct_qualifier(name: str) -> None:
    ir = parse(name, strict=True).ir
    convention = [q for q in ir.qualifiers if q.token == "equivalent_electron"]
    assert len(convention) == 1
    assert convention[0].category == "normalized"
    assert ir.qualifiers[-1] == convention[0]
    assert ir.base.token == ("source_rate" if "source_rate" in name else "count")
    assert compose(ir) == name
    assert compose_standard_name(parse_standard_name(name)) == name

    plain_name = name.replace("equivalent_electron_", "")
    plain_ir = parse(plain_name, strict=True).ir
    assert (
        ir.model_copy(
            update={"qualifiers": [q for q in ir.qualifiers if q not in convention]}
        )
        == plain_ir
    )
    assert compose(plain_ir) == plain_name


def test_prefill_keeps_the_convention_next_to_the_base() -> None:
    model = StandardName(
        subject="neon", physical_base="prefill_equivalent_electron_count"
    )
    rendered = compose_standard_name(model)
    assert rendered == "neon_prefill_equivalent_electron_count"
    assert rendered != "neon_equivalent_electron_prefill_count"
    assert compose(parse(rendered, strict=True).ir) == rendered


def test_equivalent_electron_is_public_and_has_no_base_restriction() -> None:
    assert Qualifier("equivalent_electron") == Qualifier.EQUIVALENT_ELECTRON
    name = "equivalent_electron_pressure"
    assert compose(parse(name, strict=True).ir) == name


@pytest.mark.parametrize("name", NAMES)
def test_electron_equivalent_is_not_an_alias(name: str) -> None:
    with pytest.raises(ParseError):
        parse(name.replace("equivalent_electron", "electron_equivalent"), strict=True)


@pytest.mark.parametrize("name", NAMES)
def test_count_and_rate_share_one_convention_definition(name: str) -> None:
    convention = load_concepts().concepts["equivalent_electron"]
    applied = [
        c for c in select_conventions(name).concepts if c.token == "equivalent_electron"
    ]
    assert len(applied) == 1
    assert applied[0].definition == convention.definition
    assert applied[0].equation == convention.equation
    assert applied[0].restates_equation is False
