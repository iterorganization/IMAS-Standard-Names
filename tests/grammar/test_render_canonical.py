"""Canonical rendering tests for the grammar.

Builds IR instances manually and asserts :func:`imas_standard_names.grammar.render.compose`
produces the exact canonical string required by the grammar.

Canonical form rules:
- Operator wrapping: ``unary_prefix`` → ``<op>_of_<inner>``;
  ``unary_postfix`` → ``<inner>_<op>``; ``binary`` → ``<op>_of_<A>_<sep>_<B>``
- A unary operator over an ordinary base with an ``of``, ``at``, or ``due_to``
  tail renders between the base group and that tail
- Operators inside a binary operand tree retain their natural positions
- Projection prefix (canonical): ``<axis>_`` before qualifiers+base
  (component shape) or ``<axis>_`` before carrier (coordinate shape)
- Locus suffix: ``_of_<tok>`` / ``_at_<tok>`` / ``_over_<tok>``
- Mechanism suffix: ``_due_to_<process>``
- Default order: operators(outer→inner) → projection → qualifiers → base → locus
  → mechanism

Each test states the rendering rule it verifies.
"""

import pytest

from imas_standard_names import parse
from imas_standard_names.grammar import parse_standard_name
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
)
from imas_standard_names.grammar.render import RenderError, compose

# ---------------------------------------------------------------------------
# Bare base with no operators or decorators
# ---------------------------------------------------------------------------


def test_render_bare_quantity_base() -> None:
    """A bare quantity base renders as its token unchanged."""
    ir = StandardNameIR(
        base=QuantityOrCarrier(token="pressure", kind=BaseKind.QUANTITY)
    )
    assert compose(ir) == "pressure"


def test_render_bare_geometry_carrier() -> None:
    """A bare geometry carrier renders as its token unchanged."""
    ir = StandardNameIR(
        base=QuantityOrCarrier(token="normalized_minor_radius", kind=BaseKind.GEOMETRY)
    )
    assert compose(ir) == "normalized_minor_radius"


# ---------------------------------------------------------------------------
# Unary prefix operator
# ---------------------------------------------------------------------------


def test_render_unary_prefix_operator() -> None:
    """Render ``unary_prefix`` as ``<op>_of_<inner>``."""
    ir = StandardNameIR(
        operators=[OperatorApplication(kind=OperatorKind.UNARY_PREFIX, op="maximum")],
        base=QuantityOrCarrier(token="pressure", kind=BaseKind.QUANTITY),
    )
    assert compose(ir) == "maximum_of_pressure"


def test_render_unary_prefix_derivative() -> None:
    """Render a derivative prefix around its operand."""
    ir = StandardNameIR(
        operators=[
            OperatorApplication(kind=OperatorKind.UNARY_PREFIX, op="derivative")
        ],
        base=QuantityOrCarrier(token="pressure", kind=BaseKind.QUANTITY),
    )
    assert compose(ir) == "derivative_of_pressure"


def test_render_unary_prefix_flux_surface_averaged() -> None:
    """``flux_surface_averaged`` prefix operator."""
    ir = StandardNameIR(
        operators=[
            OperatorApplication(
                kind=OperatorKind.UNARY_PREFIX, op="flux_surface_averaged"
            )
        ],
        base=QuantityOrCarrier(token="temperature", kind=BaseKind.QUANTITY),
    )
    assert compose(ir) == "flux_surface_averaged_of_temperature"


# ---------------------------------------------------------------------------
# Unary postfix operator
# ---------------------------------------------------------------------------


def test_render_unary_postfix_magnitude() -> None:
    """Render ``unary_postfix`` as ``<inner>_<op>``."""
    ir = StandardNameIR(
        operators=[
            OperatorApplication(kind=OperatorKind.UNARY_POSTFIX, op="magnitude")
        ],
        base=QuantityOrCarrier(token="pressure", kind=BaseKind.QUANTITY),
    )
    assert compose(ir) == "pressure_magnitude"


def test_render_unary_postfix_gyroaveraged() -> None:
    """Render the ``gyroaveraged`` postfix operator."""
    ir = StandardNameIR(
        operators=[
            OperatorApplication(kind=OperatorKind.UNARY_POSTFIX, op="gyroaveraged")
        ],
        base=QuantityOrCarrier(token="temperature", kind=BaseKind.QUANTITY),
    )
    assert compose(ir) == "temperature_gyroaveraged"


def test_render_unary_postfix_moment() -> None:
    """``moment`` postfix operator."""
    ir = StandardNameIR(
        operators=[OperatorApplication(kind=OperatorKind.UNARY_POSTFIX, op="moment")],
        base=QuantityOrCarrier(token="temperature", kind=BaseKind.QUANTITY),
    )
    assert compose(ir) == "temperature_moment"


# ---------------------------------------------------------------------------
# Binary operators
# ---------------------------------------------------------------------------


def test_render_binary_ratio() -> None:
    """Render a binary ratio with the ``to`` separator."""
    ir = StandardNameIR(
        operators=[
            OperatorApplication(
                kind=OperatorKind.BINARY,
                op="ratio",
                separator="to",
                args=[
                    StandardNameIR(
                        base=QuantityOrCarrier(token="pressure", kind=BaseKind.QUANTITY)
                    ),
                    StandardNameIR(
                        base=QuantityOrCarrier(
                            token="temperature", kind=BaseKind.QUANTITY
                        )
                    ),
                ],
            )
        ],
        base=QuantityOrCarrier(token="placeholder", kind=BaseKind.QUANTITY),
    )
    assert compose(ir) == "ratio_of_pressure_to_temperature"


def test_render_binary_product() -> None:
    """Render a binary product with the ``and`` separator."""
    ir = StandardNameIR(
        operators=[
            OperatorApplication(
                kind=OperatorKind.BINARY,
                op="product",
                separator="and",
                args=[
                    StandardNameIR(
                        base=QuantityOrCarrier(token="pressure", kind=BaseKind.QUANTITY)
                    ),
                    StandardNameIR(
                        base=QuantityOrCarrier(
                            token="temperature", kind=BaseKind.QUANTITY
                        )
                    ),
                ],
            )
        ],
        base=QuantityOrCarrier(token="placeholder", kind=BaseKind.QUANTITY),
    )
    assert compose(ir) == "product_of_pressure_and_temperature"


# ---------------------------------------------------------------------------
# Axis projection prefix
# ---------------------------------------------------------------------------


def test_render_projection_component_prefix() -> None:
    """Render a component projection as ``<axis>_<base>``."""
    ir = StandardNameIR(
        projection=AxisProjection(axis="radial", shape=ProjectionShape.COMPONENT),
        base=QuantityOrCarrier(token="pressure", kind=BaseKind.QUANTITY),
    )
    assert compose(ir) == "radial_pressure"


def test_render_projection_toroidal_component() -> None:
    """Render a toroidal component projection."""
    ir = StandardNameIR(
        projection=AxisProjection(axis="toroidal", shape=ProjectionShape.COMPONENT),
        base=QuantityOrCarrier(token="magnetic_field", kind=BaseKind.QUANTITY),
    )
    assert compose(ir) == "toroidal_magnetic_field"


def test_render_projection_coordinate_prefix() -> None:
    """Render a coordinate projection as short form ``<axis>_<carrier>``."""
    ir = StandardNameIR(
        projection=AxisProjection(axis="vertical", shape=ProjectionShape.COORDINATE),
        base=QuantityOrCarrier(token="normalized_minor_radius", kind=BaseKind.GEOMETRY),
    )
    assert compose(ir) == "vertical_normalized_minor_radius"


def test_render_projection_normalized_toroidal_coordinate() -> None:
    """Render ``normalized_toroidal_flux_coordinate`` with an axis projection."""
    ir = StandardNameIR(
        projection=AxisProjection(
            axis="normalized_toroidal", shape=ProjectionShape.COORDINATE
        ),
        base=QuantityOrCarrier(
            token="normalized_toroidal_flux_coordinate",
            kind=BaseKind.GEOMETRY,
        ),
    )
    # Axis 'normalized_toroidal' + coordinate shape → short form
    assert compose(ir) == ("normalized_toroidal_normalized_toroidal_flux_coordinate")


# ---------------------------------------------------------------------------
# Locus suffix
# ---------------------------------------------------------------------------


def test_render_locus_of_entity() -> None:
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


def test_render_locus_at_position() -> None:
    """Position locus with ``_at_`` suffix."""
    ir = StandardNameIR(
        base=QuantityOrCarrier(token="temperature", kind=BaseKind.QUANTITY),
        locus=LocusRef(
            relation=LocusRelation.AT,
            token="magnetic_axis",
            type=LocusType.POSITION,
        ),
    )
    assert compose(ir) == "temperature_at_magnetic_axis"


def test_render_locus_of_position() -> None:
    """Render a position locus with the ``_of_`` relation."""
    ir = StandardNameIR(
        base=QuantityOrCarrier(token="major_radius", kind=BaseKind.QUANTITY),
        locus=LocusRef(
            relation=LocusRelation.OF,
            token="x_point",
            type=LocusType.POSITION,
        ),
    )
    assert compose(ir) == "major_radius_of_x_point"


def test_render_locus_entity_of() -> None:
    """Render an entity locus with the ``_of_`` relation."""
    ir = StandardNameIR(
        base=QuantityOrCarrier(token="pressure", kind=BaseKind.QUANTITY),
        locus=LocusRef(
            relation=LocusRelation.OF,
            token="bolometer",
            type=LocusType.ENTITY,
        ),
    )
    assert compose(ir) == "pressure_of_bolometer"


# ---------------------------------------------------------------------------
# Mechanism suffix
# ---------------------------------------------------------------------------


def test_render_mechanism_due_to() -> None:
    """``_due_to_<process>`` appended after all other suffixes."""
    ir = StandardNameIR(
        base=QuantityOrCarrier(token="pressure", kind=BaseKind.QUANTITY),
        mechanism=Process(token="conduction"),
    )
    assert compose(ir) == "pressure_due_to_conduction"


def test_render_mechanism_with_locus() -> None:
    """Locus before mechanism: ``X_at_L_due_to_P``."""
    ir = StandardNameIR(
        base=QuantityOrCarrier(token="pressure", kind=BaseKind.QUANTITY),
        locus=LocusRef(
            relation=LocusRelation.AT,
            token="plasma_boundary",
            type=LocusType.POSITION,
        ),
        mechanism=Process(token="conduction"),
    )
    assert compose(ir) == "pressure_at_plasma_boundary_due_to_conduction"


# ---------------------------------------------------------------------------
# Combined operator, projection, locus, and mechanism ordering
# ---------------------------------------------------------------------------


def test_render_operator_between_projected_base_and_locus() -> None:
    """Render ``<axis>_<qual>_<base>_<op>_<locus>`` canonically."""
    ir = StandardNameIR(
        operators=[
            OperatorApplication(kind=OperatorKind.UNARY_PREFIX, op="root_mean_square")
        ],
        projection=AxisProjection(axis="radial", shape=ProjectionShape.COMPONENT),
        qualifiers=[Qualifier(token="electron")],
        base=QuantityOrCarrier(token="pressure", kind=BaseKind.QUANTITY),
        locus=LocusRef(
            relation=LocusRelation.AT,
            token="plasma_boundary",
            type=LocusType.POSITION,
        ),
    )
    assert compose(ir) == (
        "radial_electron_pressure_root_mean_square_at_plasma_boundary"
    )


def test_render_nested_operators_outer_first() -> None:
    """Render the outermost operator nearest the trailing locus."""
    ir = StandardNameIR(
        operators=[
            OperatorApplication(kind=OperatorKind.UNARY_PREFIX, op="maximum"),
            OperatorApplication(kind=OperatorKind.UNARY_PREFIX, op="derivative"),
        ],
        base=QuantityOrCarrier(token="pressure", kind=BaseKind.QUANTITY),
        locus=LocusRef(
            relation=LocusRelation.AT,
            token="pedestal",
            type=LocusType.POSITION,
        ),
    )
    # The derivative is inner and nearest the base; maximum remains outer.
    assert compose(ir) == "pressure_derivative_maximum_at_pedestal"


def test_render_nested_indexed_operator_at_pedestal() -> None:
    """Render a nested maximum and indexed derivative at the pedestal."""
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


def test_render_maximum_energy_flux_at_target() -> None:
    """Render the peak deposited-energy surface load at a divertor target."""
    ir = StandardNameIR(
        operators=[OperatorApplication(kind=OperatorKind.UNARY_PREFIX, op="maximum")],
        qualifiers=[Qualifier(token="energy")],
        base=QuantityOrCarrier(token="flux", kind=BaseKind.QUANTITY),
        locus=LocusRef(
            relation=LocusRelation.AT,
            token="inner_divertor_target",
            type=LocusType.POSITION,
        ),
    )
    rendered = compose(ir)
    assert rendered == "energy_flux_maximum_at_inner_divertor_target"

    parsed = parse(rendered, strict=True)
    assert compose(parsed.ir) == rendered

    flat = parse_standard_name(rendered)
    assert flat.physical_base == "flux"
    assert flat.channel is not None
    assert flat.channel.value == "energy"


# ---------------------------------------------------------------------------
# Qualifier rendering
# ---------------------------------------------------------------------------


def test_render_single_qualifier() -> None:
    """Qualifier token is prefixed directly before the base."""
    ir = StandardNameIR(
        qualifiers=[Qualifier(token="electron")],
        base=QuantityOrCarrier(token="pressure", kind=BaseKind.QUANTITY),
    )
    assert compose(ir) == "electron_pressure"


def test_render_multiple_qualifiers() -> None:
    """Multiple qualifiers are concatenated in list order before base."""
    ir = StandardNameIR(
        qualifiers=[Qualifier(token="fast"), Qualifier(token="ion")],
        base=QuantityOrCarrier(token="pressure", kind=BaseKind.QUANTITY),
    )
    assert compose(ir) == "fast_ion_pressure"


# ---------------------------------------------------------------------------
# Render errors (invalid IR)
# ---------------------------------------------------------------------------


def test_render_error_on_missing_separator_for_binary() -> None:
    """Binary operator without separator raises ValidationError at construction."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        # separator required for binary operators — pydantic should catch this
        OperatorApplication(
            kind=OperatorKind.BINARY,
            op="ratio",
            # separator intentionally missing
            args=[
                StandardNameIR(
                    base=QuantityOrCarrier(token="pressure", kind=BaseKind.QUANTITY)
                ),
                StandardNameIR(
                    base=QuantityOrCarrier(token="temperature", kind=BaseKind.QUANTITY)
                ),
            ],
        )
