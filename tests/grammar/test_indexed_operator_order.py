"""Canonical ordering for coordinate-indexed unary operators."""

from imas_standard_names.grammar.ir import (
    BaseKind,
    LocusRef,
    LocusRelation,
    LocusType,
    OperatorApplication,
    OperatorKind,
    QuantityOrCarrier,
    StandardNameIR,
)
from imas_standard_names.grammar.parser import parse
from imas_standard_names.grammar.render import compose


def _flux_surface_quantity(base: str) -> StandardNameIR:
    return StandardNameIR(
        operators=[
            OperatorApplication(
                kind=OperatorKind.UNARY_PREFIX,
                op="derivative_with_respect_to_toroidal_flux_coordinate",
            )
        ],
        base=QuantityOrCarrier(token=base, kind=BaseKind.QUANTITY),
        locus=LocusRef(
            relation=LocusRelation.OF,
            token="flux_surface",
            type=LocusType.POSITION,
        ),
    )


def test_indexed_operator_renders_operand_before_index() -> None:
    ir = _flux_surface_quantity("area")
    rendered = compose(ir)

    assert rendered == (
        "derivative_of_area_of_flux_surface_with_respect_to_toroidal_flux_coordinate"
    )
    assert parse(rendered, strict=True).ir == ir


def test_nested_indexed_operator_round_trips_operand_first() -> None:
    indexed = _flux_surface_quantity("volume")
    ir = indexed.model_copy(
        update={
            "operators": [
                OperatorApplication(
                    kind=OperatorKind.UNARY_PREFIX,
                    op="time_derivative",
                ),
                *indexed.operators,
            ]
        }
    )
    rendered = compose(ir)

    assert rendered == (
        "time_derivative_of_derivative_of_volume_of_flux_surface_"
        "with_respect_to_toroidal_flux_coordinate"
    )
    assert parse(rendered, strict=True).ir == ir


def test_composite_ratio_operands_keep_their_operator_positions() -> None:
    name = (
        "flux_surface_averaged_ratio_of_square_of_"
        "toroidal_flux_coordinate_gradient_magnitude_"
        "to_square_of_major_radius"
    )

    assert compose(parse(name, strict=True).ir) == name
