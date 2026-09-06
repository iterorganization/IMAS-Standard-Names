"""Tests for the grammar IR model and canonical renderer.

These tests validate the IR shape constraints and the canonical strings
produced by :func:`imas_standard_names.grammar.render.compose`.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from imas_standard_names.grammar.ir import (
    AxisProjection,
    BaseKind,
    LocusRef,
    LocusRelation,
    LocusType,
    OperatorApplication,
    OperatorKind,
    Process,
    ProjectionShape,
    Qualifier,
    QuantityOrCarrier,
    StandardNameIR,
    assert_binary_has_separator,
    assert_locus_is_trailing,
    assert_operator_of_form,
)
from imas_standard_names.grammar.render import RenderError, compose

# ---------------------------------------------------------------------------
# Canonical rendering examples
# ---------------------------------------------------------------------------


def test_compose_simple_locus():
    """Render ``elongation_of_plasma_boundary`` with an entity locus."""
    ir = StandardNameIR(
        base=QuantityOrCarrier(token="elongation", kind=BaseKind.QUANTITY),
        locus=LocusRef(
            relation=LocusRelation.OF,
            token="plasma_boundary",
            type=LocusType.ENTITY,
        ),
    )
    assert compose(ir) == "elongation_of_plasma_boundary"


def test_compose_projection_plus_qualifier_plus_locus():
    """Render a radial component, electron qualifier, and position locus."""
    ir = StandardNameIR(
        projection=AxisProjection(axis="radial", shape=ProjectionShape.COMPONENT),
        qualifiers=[Qualifier(token="electron")],
        base=QuantityOrCarrier(token="pressure", kind=BaseKind.QUANTITY),
        locus=LocusRef(
            relation=LocusRelation.AT,
            token="plasma_boundary",
            type=LocusType.POSITION,
        ),
    )
    assert compose(ir) == "radial_electron_pressure_at_plasma_boundary"


def test_compose_nested_prefix_operators():
    """Render nested ``maximum_of`` and ``derivative`` operators with a locus."""
    ir = StandardNameIR(
        operators=[
            OperatorApplication(kind=OperatorKind.UNARY_PREFIX, op="maximum"),
            OperatorApplication(
                kind=OperatorKind.UNARY_PREFIX,
                op="derivative_with_respect_to_normalized_poloidal_flux",
            ),
        ],
        qualifiers=[Qualifier(token="electron")],
        base=QuantityOrCarrier(token="pressure", kind=BaseKind.QUANTITY),
        locus=LocusRef(
            relation=LocusRelation.AT,
            token="pedestal",
            type=LocusType.POSITION,
        ),
    )
    assert compose(ir) == (
        "maximum_of_derivative_of_electron_pressure_at_pedestal_"
        "with_respect_to_normalized_poloidal_flux"
    )


def test_compose_binary_ratio():
    """Binary operator with ``_to_`` separator."""
    ir = StandardNameIR(
        base=QuantityOrCarrier(token="placeholder", kind=BaseKind.QUANTITY),
        operators=[
            OperatorApplication(
                kind=OperatorKind.BINARY,
                op="ratio",
                separator="to",
                args=[
                    StandardNameIR(
                        base=QuantityOrCarrier(
                            token="plasma_current", kind=BaseKind.QUANTITY
                        ),
                    ),
                    StandardNameIR(
                        base=QuantityOrCarrier(
                            token="toroidal_field", kind=BaseKind.QUANTITY
                        ),
                    ),
                ],
            ),
        ],
    )
    assert compose(ir) == "ratio_of_plasma_current_to_toroidal_field"


def test_compose_postfix_operator():
    """Unary postfix operator: ``<inner>_magnitude``."""
    ir = StandardNameIR(
        base=QuantityOrCarrier(token="magnetic_field", kind=BaseKind.QUANTITY),
        operators=[
            OperatorApplication(kind=OperatorKind.UNARY_POSTFIX, op="magnitude"),
        ],
    )
    assert compose(ir) == "magnetic_field_magnitude"


def test_compose_mechanism_trails_locus():
    """Mechanism ``_due_to_…`` is always last, after the locus."""
    ir = StandardNameIR(
        base=QuantityOrCarrier(token="ion_momentum_flux", kind=BaseKind.QUANTITY),
        mechanism=Process(token="diamagnetic_drift"),
    )
    assert compose(ir) == "ion_momentum_flux_due_to_diamagnetic_drift"


def test_compose_coordinate_shape_on_geometry_carrier():
    """Render ``vertical_<geometry_carrier>`` as a coordinate projection."""
    ir = StandardNameIR(
        projection=AxisProjection(axis="vertical", shape=ProjectionShape.COORDINATE),
        base=QuantityOrCarrier(token="position", kind=BaseKind.GEOMETRY),
        locus=LocusRef(
            relation=LocusRelation.OF,
            token="ion_cyclotron_heating_antenna",
            type=LocusType.ENTITY,
        ),
    )
    assert compose(ir) == ("vertical_position_of_ion_cyclotron_heating_antenna")


# ---------------------------------------------------------------------------
# IR validation — rejection cases
# ---------------------------------------------------------------------------


def test_component_shape_requires_quantity_base():
    with pytest.raises(ValidationError):
        StandardNameIR(
            projection=AxisProjection(axis="radial", shape=ProjectionShape.COMPONENT),
            base=QuantityOrCarrier(token="position", kind=BaseKind.GEOMETRY),
        )


def test_coordinate_shape_requires_geometry_base():
    with pytest.raises(ValidationError):
        StandardNameIR(
            projection=AxisProjection(axis="radial", shape=ProjectionShape.COORDINATE),
            base=QuantityOrCarrier(token="pressure", kind=BaseKind.QUANTITY),
        )


def test_locus_rejects_incompatible_relation():
    """``_at_`` is not valid on an entity-typed locus."""
    with pytest.raises(ValidationError):
        LocusRef(
            relation=LocusRelation.AT,
            token="plasma_boundary",
            type=LocusType.ENTITY,
        )


def test_locus_region_requires_over():
    with pytest.raises(ValidationError):
        LocusRef(
            relation=LocusRelation.OF,
            token="plasma_volume",
            type=LocusType.REGION,
        )


def test_binary_operator_requires_separator():
    with pytest.raises(ValidationError):
        OperatorApplication(
            kind=OperatorKind.BINARY,
            op="ratio",
            args=[
                StandardNameIR(
                    base=QuantityOrCarrier(token="a", kind=BaseKind.QUANTITY)
                ),
                StandardNameIR(
                    base=QuantityOrCarrier(token="b", kind=BaseKind.QUANTITY)
                ),
            ],
        )


def test_binary_operator_requires_two_args():
    with pytest.raises(ValidationError):
        OperatorApplication(
            kind=OperatorKind.BINARY,
            op="ratio",
            separator="to",
            args=[
                StandardNameIR(
                    base=QuantityOrCarrier(token="a", kind=BaseKind.QUANTITY)
                ),
            ],
        )


def test_unary_operator_rejects_separator():
    with pytest.raises(ValidationError):
        OperatorApplication(
            kind=OperatorKind.UNARY_PREFIX,
            op="magnitude",
            separator="and",
        )


def test_duplicate_qualifiers_rejected():
    with pytest.raises(ValidationError):
        StandardNameIR(
            qualifiers=[Qualifier(token="electron"), Qualifier(token="electron")],
            base=QuantityOrCarrier(token="pressure", kind=BaseKind.QUANTITY),
        )


def test_invalid_token_shape_rejected():
    with pytest.raises(ValidationError):
        Qualifier(token="Electron")  # uppercase
    with pytest.raises(ValidationError):
        Qualifier(token="_leading")
    with pytest.raises(ValidationError):
        Qualifier(token="")


# ---------------------------------------------------------------------------
# Operator-form assertion helpers
# ---------------------------------------------------------------------------


def test_assert_operator_of_form_shape_only():
    op = OperatorApplication(kind=OperatorKind.UNARY_PREFIX, op="magnitude")
    assert_operator_of_form(op, registry=None)  # no-op, no raise


def test_assert_operator_of_form_with_registry():
    op = OperatorApplication(kind=OperatorKind.UNARY_PREFIX, op="magnitude")
    registry = {"magnitude": {"kind": "unary_prefix"}}
    assert_operator_of_form(op, registry=registry)

    bad_registry = {"magnitude": {"kind": "unary_postfix"}}
    with pytest.raises(ValueError, match="registered with kind"):
        assert_operator_of_form(op, registry=bad_registry)

    with pytest.raises(ValueError, match="not registered"):
        assert_operator_of_form(op, registry={})


def test_assert_operator_of_form_rejects_wrong_kind():
    op = OperatorApplication(kind=OperatorKind.UNARY_POSTFIX, op="magnitude")
    with pytest.raises(ValueError, match="expects a unary_prefix"):
        assert_operator_of_form(op)


def test_assert_binary_has_separator_shape_ok():
    op = OperatorApplication(
        kind=OperatorKind.BINARY,
        op="ratio",
        separator="to",
        args=[
            StandardNameIR(base=QuantityOrCarrier(token="a", kind=BaseKind.QUANTITY)),
            StandardNameIR(base=QuantityOrCarrier(token="b", kind=BaseKind.QUANTITY)),
        ],
    )
    assert_binary_has_separator(op, registry=None)
    assert_binary_has_separator(op, registry={"ratio": {"separator": "to"}})
    with pytest.raises(ValueError, match="separator mismatch"):
        assert_binary_has_separator(op, registry={"ratio": {"separator": "and"}})


def test_assert_locus_is_trailing_ok():
    ir = StandardNameIR(
        base=QuantityOrCarrier(token="elongation", kind=BaseKind.QUANTITY),
        locus=LocusRef(
            relation=LocusRelation.OF,
            token="plasma_boundary",
            type=LocusType.ENTITY,
        ),
    )
    assert_locus_is_trailing("elongation_of_plasma_boundary", ir)


def test_assert_locus_is_trailing_with_mechanism_ok():
    ir = StandardNameIR(
        base=QuantityOrCarrier(token="pressure", kind=BaseKind.QUANTITY),
        locus=LocusRef(
            relation=LocusRelation.AT,
            token="pedestal",
            type=LocusType.POSITION,
        ),
        mechanism=Process(token="collisions"),
    )
    assert_locus_is_trailing("pressure_at_pedestal_due_to_collisions", ir)


def test_assert_locus_is_trailing_rejects_displaced_locus():
    ir = StandardNameIR(
        base=QuantityOrCarrier(token="elongation", kind=BaseKind.QUANTITY),
        locus=LocusRef(
            relation=LocusRelation.OF,
            token="plasma_boundary",
            type=LocusType.ENTITY,
        ),
    )
    with pytest.raises(ValueError, match="is not trailing"):
        assert_locus_is_trailing("something_else_entirely", ir)


# ---------------------------------------------------------------------------
# compose() error propagation
# ---------------------------------------------------------------------------


def test_compose_rejects_non_ir_input():
    with pytest.raises(RenderError):
        compose({"base": "foo"})  # type: ignore[arg-type]
