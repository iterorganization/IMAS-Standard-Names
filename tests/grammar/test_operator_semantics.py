"""Public semantic metadata for grammar operators."""

import json

import pytest

from imas_standard_names import (
    StandardNameIR,
    get_grammar_context,
    get_operator_semantics,
    parse,
)


@pytest.mark.parametrize(
    "token",
    ["change_in", "time_derivative"],
)
def test_temporal_change_operators_expose_the_shared_effect(token: str) -> None:
    assert get_operator_semantics(token) == frozenset({"temporal_change"})


@pytest.mark.parametrize(
    "token",
    ["gradient", "line_averaged", "ratio", "variation", "not_an_operator"],
)
def test_other_and_unknown_tokens_have_no_semantic_effects(token: str) -> None:
    assert get_operator_semantics(token) == frozenset()


@pytest.mark.parametrize(
    "name,token_locations",
    [
        (
            "time_derivative_of_electron_density",
            [("operator", "time_derivative")],
        ),
        (
            "gradient_of_time_derivative_of_electron_temperature",
            [("operator", "gradient"), ("operator", "time_derivative")],
        ),
        (
            "difference_of_time_derivative_of_pressure_and_temperature",
            [
                ("operator", "difference"),
                ("operator", "time_derivative"),
            ],
        ),
        (
            "difference_of_change_in_electron_density_and_ion_density",
            [("operator", "difference"), ("qualifier", "change_in")],
        ),
        ("change_in_electron_density", [("qualifier", "change_in")]),
    ],
)
def test_lookup_is_independent_of_ir_token_location(
    name: str,
    token_locations: list[tuple[str, str]],
) -> None:
    parsed = parse(name, strict=True).ir

    found: list[tuple[str, str]] = []

    def visit(ir: StandardNameIR) -> None:
        for operator in ir.operators:
            found.append(("operator", operator.op))
            for argument in operator.args:
                visit(argument)
        found.extend(("qualifier", qualifier.token) for qualifier in ir.qualifiers)

    visit(parsed)
    assert all(location in found for location in token_locations)
    effects = {token: get_operator_semantics(token) for _, token in token_locations}
    expected_temporal = {
        token
        for _, token in token_locations
        if token in {"change_in", "time_derivative"}
    }
    assert {
        token for token, semantics in effects.items() if "temporal_change" in semantics
    } == expected_temporal


def test_context_exposes_json_safe_operator_semantics_without_registry_drift() -> None:
    context = get_grammar_context()
    operators = context["grammar"]["vocabularies"]["operators"]

    assert json.loads(json.dumps(operators)) == operators
    assert operators["time_derivative"]["semantic_effects"] == ["temporal_change"]
    assert operators["gradient"]["semantic_effects"] == []
    assert {
        token: frozenset(metadata["semantic_effects"])
        for token, metadata in operators.items()
    } == {token: get_operator_semantics(token) for token in operators}


def test_public_lookup_result_is_immutable() -> None:
    effects = get_operator_semantics("time_derivative")
    with pytest.raises(AttributeError):
        effects.add("mutated")  # type: ignore[attr-defined]
