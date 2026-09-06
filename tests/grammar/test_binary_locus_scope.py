"""Lossless locus scope for recursive binary expressions."""

import pytest

from imas_standard_names import ParseError, StandardNameIR, compose, parse
from imas_standard_names.grammar import RenderError


def _binary(
    operator: str,
    separator: str,
    left: StandardNameIR,
    right: StandardNameIR,
) -> StandardNameIR:
    return StandardNameIR.model_validate(
        {
            "operators": [
                {
                    "kind": "binary",
                    "op": operator,
                    "separator": separator,
                    "args": [left, right],
                }
            ],
            "base": {"token": "placeholder", "kind": "quantity"},
        }
    )


def _binary_operands(name: str) -> tuple[StandardNameIR, StandardNameIR]:
    parsed = parse(name, strict=True).ir
    binary = parsed.operators[0]
    return binary.args[0], binary.args[1]


def test_numerator_locus_stays_on_numerator() -> None:
    name = "ratio_of_neutral_density_of_isotope_to_total_neutral_density"

    parsed = parse(name, strict=True).ir
    numerator, denominator = _binary_operands(name)

    assert parsed.locus is None
    assert numerator.locus is not None
    assert numerator.locus.token == "isotope"
    assert denominator.locus is None


def test_each_binary_leaf_retains_its_locus() -> None:
    name = "ratio_of_neutral_density_of_isotope_to_total_neutral_density_of_isotope"

    parsed = parse(name, strict=True).ir
    numerator, denominator = _binary_operands(name)

    assert parsed.locus is None
    assert numerator.locus is not None
    assert denominator.locus is not None
    assert numerator.locus.token == denominator.locus.token == "isotope"


def test_nested_binary_terminal_locus_binds_to_final_leaf() -> None:
    name = (
        "ratio_of_neutral_density_of_isotope_to_difference_of_"
        "total_neutral_density_and_neutral_density_of_isotope"
    )

    parsed = parse(name, strict=True).ir
    numerator, denominator = _binary_operands(name)
    difference = denominator.operators[0]

    assert parsed.locus is None
    assert numerator.locus is not None
    assert denominator.locus is None
    assert difference.args[0].locus is None
    assert difference.args[1].locus is not None
    assert difference.args[1].locus.token == "isotope"


def test_terminal_position_locus_binds_to_final_operand() -> None:
    name = "ratio_of_pressure_to_temperature_at_magnetic_axis"

    parsed = parse(name, strict=True).ir
    numerator, denominator = _binary_operands(name)

    assert parsed.locus is None
    assert numerator.locus is None
    assert denominator.locus is not None
    assert denominator.locus.token == "magnetic_axis"


def test_enclosing_binary_locus_cannot_be_rendered_losslessly() -> None:
    binary = _binary(
        "ratio",
        "to",
        parse("electron_temperature", strict=True).ir,
        parse("ion_temperature", strict=True).ir,
    )
    locus = parse("electron_temperature_at_magnetic_axis", strict=True).ir.locus
    ambiguous = binary.model_copy(update={"locus": locus})

    with pytest.raises(RenderError, match="enclosing binary expression is ambiguous"):
        compose(ambiguous)


def test_recursive_binary_round_trip_preserves_ir() -> None:
    numerator = parse("neutral_density_of_isotope", strict=True).ir
    total_density = parse("total_neutral_density", strict=True).ir
    other_isotope_density = parse("neutral_density_of_isotope", strict=True).ir
    denominator = _binary("difference", "and", total_density, other_isotope_density)
    expected = _binary("ratio", "to", numerator, denominator)

    rendered = compose(expected)

    assert parse(rendered, strict=True).ir == expected


def test_repeated_terminal_locus_is_rejected() -> None:
    malformed = (
        "ratio_of_neutral_density_of_isotope_to_difference_of_"
        "total_neutral_density_and_neutral_density_of_isotope_of_isotope"
    )

    with pytest.raises(ParseError):
        parse(malformed, strict=True)


@pytest.mark.parametrize(
    "name",
    [
        "pressure_time_derivative_at_magnetic_axis",
        "ratio_of_pressure_to_temperature",
        "difference_of_total_pressure_and_electron_pressure",
    ],
)
def test_existing_operator_forms_remain_canonical(name: str) -> None:
    parsed = parse(name, strict=True).ir

    assert compose(parsed) == name
