"""Unit tests for the grammar parser.

These tests exercise the parser against fixture-injected closed vocabularies;
the parser itself does not depend on YAML contents.

For every canonical grammar name the test asserts ``compose(parse(name).ir)
== name`` (round-trip). For non-canonical and error cases, the test
asserts specific diagnostic / exception behaviour.
"""

import pytest

from imas_standard_names.grammar.ir import (
    BaseKind,
    LocusRelation,
    LocusType,
    OperatorKind,
    ProjectionShape,
)
from imas_standard_names.grammar.parser import (
    Diagnostic,
    ParseError,
    ParseResult,
    Vocabularies,
    parse,
    validate_round_trip,
)
from imas_standard_names.grammar.render import compose

# ---------------------------------------------------------------------------
# Fixture: a minimal grammar vocabulary bundle
# ---------------------------------------------------------------------------


@pytest.fixture
def vocabs() -> Vocabularies:
    """Inject a minimal vocabulary covering all test names.

    This fixture bypasses YAML loading. Every base, locus, operator, qualifier,
    and axis referenced in the tests below is declared here.
    """

    loci: dict[str, tuple[LocusType, frozenset[LocusRelation]]] = {
        "plasma_boundary": (
            LocusType.POSITION,
            frozenset({LocusRelation.AT, LocusRelation.OF}),
        ),
        "pedestal": (
            LocusType.POSITION,
            frozenset({LocusRelation.AT, LocusRelation.OF}),
        ),
        "magnetic_axis": (
            LocusType.POSITION,
            frozenset({LocusRelation.AT, LocusRelation.OF}),
        ),
        "ferritic_element_centroid": (
            LocusType.ENTITY,
            frozenset({LocusRelation.OF}),
        ),
        "flux_loop": (
            LocusType.ENTITY,
            frozenset({LocusRelation.OF}),
        ),
    }

    operators: dict[str, dict] = {
        "magnitude": {
            "kind": OperatorKind.UNARY_POSTFIX.value,
            "precedence": 10,
            "separator": None,
            "indexed": False,
            "index_params": None,
            "returns": "scalar",
            "arg_types": ["vector", "complex"],
        },
        "maximum": {
            "kind": OperatorKind.UNARY_PREFIX.value,
            "precedence": 30,
            "separator": None,
            "indexed": False,
            "index_params": None,
            "returns": "scalar",
            "arg_types": None,
        },
        "derivative_with_respect_to_normalized_poloidal_flux": {
            "kind": OperatorKind.UNARY_PREFIX.value,
            "precedence": 20,
            "separator": None,
            "indexed": False,
            "index_params": None,
            "returns": "rate",
            "arg_types": None,
        },
        "time_derivative": {
            "kind": OperatorKind.UNARY_PREFIX.value,
            "precedence": 20,
            "separator": None,
            "indexed": False,
            "index_params": None,
            "returns": "rate",
            "arg_types": None,
        },
        "ratio": {
            "kind": OperatorKind.BINARY.value,
            "precedence": 5,
            "separator": "to",
            "indexed": False,
            "index_params": None,
            "returns": None,
            "arg_types": None,
        },
        "product": {
            "kind": OperatorKind.BINARY.value,
            "precedence": 5,
            "separator": "and",
            "indexed": False,
            "index_params": None,
            "returns": None,
            "arg_types": None,
        },
    }

    bases = frozenset(
        {
            "pressure",
            "temperature",
            "magnetic_field",
            "magnetic_field_magnitude",  # intentionally absent; derived via operator
            "magnetic_moment",
            "plasma_current",
            "toroidal_field",
            "elongation",
            "ion_momentum_flux",
        }
        - {"magnetic_field_magnitude"}
    )

    carriers = frozenset({"position"})

    qualifiers = frozenset({"electron", "ion"})

    axes = frozenset(
        {"radial", "vertical", "toroidal", "poloidal", "parallel", "perpendicular"}
    )

    return Vocabularies(
        axes=axes,
        loci=loci,
        operators=operators,
        bases=bases,
        carriers=carriers,
        qualifiers=qualifiers,
    )


# ---------------------------------------------------------------------------
# Canonical parses (must round-trip)
# ---------------------------------------------------------------------------


def test_parse_simple_locus(vocabs: Vocabularies):
    """Parse ``elongation_of_plasma_boundary`` with an entity locus."""
    name = "elongation_of_plasma_boundary"
    result = parse(name, vocabs=vocabs)
    assert isinstance(result, ParseResult)
    assert result.ir.base.token == "elongation"
    assert result.ir.base.kind is BaseKind.QUANTITY
    assert result.ir.locus is not None
    assert result.ir.locus.token == "plasma_boundary"
    assert result.ir.locus.relation is LocusRelation.OF
    assert compose(result.ir) == name


def test_parse_projection_plus_qualifier_plus_locus(vocabs: Vocabularies):
    """Parse a radial component, electron qualifier, and position locus."""
    name = "radial_electron_pressure_at_plasma_boundary"
    result = parse(name, vocabs=vocabs)
    assert result.ir.projection is not None
    assert result.ir.projection.axis == "radial"
    assert result.ir.projection.shape is ProjectionShape.COMPONENT
    assert [q.token for q in result.ir.qualifiers] == ["electron"]
    assert result.ir.base.token == "pressure"
    assert result.ir.locus is not None
    assert result.ir.locus.relation is LocusRelation.AT
    assert compose(result.ir) == name


def test_parse_nested_prefix_operators(vocabs: Vocabularies):
    """Parse the canonical nested maximum and indexed derivative decomposition."""
    name = (
        "maximum_of_derivative_of_electron_pressure_at_pedestal_"
        "with_respect_to_normalized_poloidal_flux"
    )
    result = parse(name, vocabs=vocabs)
    assert len(result.ir.operators) == 2
    assert result.ir.operators[0].op == "maximum"
    assert result.ir.operators[0].kind is OperatorKind.UNARY_PREFIX
    assert (
        result.ir.operators[1].op
        == "derivative_with_respect_to_normalized_poloidal_flux"
    )
    assert [q.token for q in result.ir.qualifiers] == ["electron"]
    assert result.ir.base.token == "pressure"
    assert result.ir.locus is not None
    assert result.ir.locus.token == "pedestal"
    assert compose(result.ir) == name


def test_parse_binary_ratio(vocabs: Vocabularies):
    """Binary operator with ``_to_`` separator."""
    name = "ratio_of_plasma_current_to_toroidal_field"
    result = parse(name, vocabs=vocabs)
    assert len(result.ir.operators) == 1
    op = result.ir.operators[0]
    assert op.kind is OperatorKind.BINARY
    assert op.op == "ratio"
    assert op.separator == "to"
    assert len(op.args) == 2
    assert op.args[0].base.token == "plasma_current"
    assert op.args[1].base.token == "toroidal_field"
    assert compose(result.ir) == name


def test_parse_binary_product_and_separator(vocabs: Vocabularies):
    """Binary operator with ``_and_`` separator."""
    name = "product_of_plasma_current_and_magnetic_field"
    result = parse(name, vocabs=vocabs)
    op = result.ir.operators[0]
    assert op.kind is OperatorKind.BINARY
    assert op.separator == "and"
    assert op.args[0].base.token == "plasma_current"
    assert op.args[1].base.token == "magnetic_field"
    assert compose(result.ir) == name


def test_parse_postfix_operator(vocabs: Vocabularies):
    """Unary postfix: ``magnetic_field_magnitude``."""
    name = "magnetic_field_magnitude"
    result = parse(name, vocabs=vocabs)
    assert len(result.ir.operators) == 1
    assert result.ir.operators[0].kind is OperatorKind.UNARY_POSTFIX
    assert result.ir.operators[0].op == "magnitude"
    assert result.ir.base.token == "magnetic_field"
    assert compose(result.ir) == name


def test_parse_mechanism(vocabs: Vocabularies):
    """Trailing ``_due_to_<process>`` mechanism attaches to the IR."""
    name = "ion_momentum_flux_due_to_diamagnetic_drift"
    result = parse(name, vocabs=vocabs)
    assert result.ir.mechanism is not None
    assert result.ir.mechanism.token == "diamagnetic_drift"
    assert result.ir.base.token == "ion_momentum_flux"
    assert compose(result.ir) == name


def test_parse_time_derivative_with_locus(vocabs: Vocabularies):
    """Prefix op + locus — combined round-trip."""
    name = "pressure_time_derivative_at_magnetic_axis"
    result = parse(name, vocabs=vocabs)
    assert len(result.ir.operators) == 1
    assert result.ir.operators[0].op == "time_derivative"
    assert result.ir.locus is not None and result.ir.locus.token == "magnetic_axis"
    assert compose(result.ir) == name


def test_validate_round_trip_helper(vocabs: Vocabularies):
    assert validate_round_trip("elongation_of_plasma_boundary", vocabs=vocabs)
    assert validate_round_trip(
        "ratio_of_plasma_current_to_toroidal_field", vocabs=vocabs
    )


# ---------------------------------------------------------------------------
# Component and locus disambiguation
# ---------------------------------------------------------------------------


def test_parse_row25_two_of_disambiguation(vocabs: Vocabularies):
    """Distinguish the projection-like suffix from the locus ``_of_``.

    The postfix-component form ``<base>_<axis>_component_of_<locus>`` is
    non-canonical; the parser resolves it without ambiguity:

      - trailing ``_of_ferritic_element_centroid`` -> locus (registry hit)
      - remaining ``_toroidal_component`` is a postfix projection... but
        we accept PREFIX projection only in this parser (compose emits
        prefix). This form is therefore non-canonical; we assert that the
        parser still reaches the ambiguity correctly — specifically, the
        locus is stripped first and the remainder fails base lookup
        with a suggestion rather than silently mis-parsing.
    """

    name = "magnetic_moment_toroidal_component_of_ferritic_element_centroid"
    # The remainder after stripping locus is
    # "magnetic_moment_toroidal_component" which is not a valid base; the
    # parser must raise rather than silently accept. The _of_ that binds to the locus is
    # consumed exclusively for the locus relation.
    with pytest.raises(ParseError) as excinfo:
        parse(name, vocabs=vocabs)
    # The residue in the error must be the non-locus tail, proving the
    # locus `_of_` was correctly bound to ferritic_element_centroid.
    assert excinfo.value.residue == "magnetic_moment_toroidal_component"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_parse_unknown_base_raises_with_suggestions(vocabs: Vocabularies):
    with pytest.raises(ParseError) as excinfo:
        parse("pressuer", vocabs=vocabs)  # typo of 'pressure'
    assert "pressure" in excinfo.value.suggestions


def test_parse_unknown_axis_falls_through_to_base_error(vocabs: Vocabularies):
    """An unknown axis in projection prefix cannot be peeled and the
    remainder fails base lookup."""
    with pytest.raises(ParseError):
        parse("banana_component_of_pressure", vocabs=vocabs)


def test_parse_random_invented_base_rejected(vocabs: Vocabularies):
    """rc20 would open-fallback this; the parser must reject."""
    with pytest.raises(ParseError):
        parse("quokka_density", vocabs=vocabs)


def test_parse_invalid_token_pattern_raises(vocabs: Vocabularies):
    with pytest.raises(ParseError):
        parse("NotSnakeCase", vocabs=vocabs)


def test_parse_empty_string_raises(vocabs: Vocabularies):
    with pytest.raises(ParseError):
        parse("", vocabs=vocabs)


# ---------------------------------------------------------------------------
# Diagnostics — vocab-gap surface
# ---------------------------------------------------------------------------


def test_parse_unregistered_at_locus_emits_vocab_gap(vocabs: Vocabularies):
    """A trailing ``_at_<unknown>`` strips with a vocab_gap diagnostic."""
    # 'novel_position' not in registry; still accepted with diagnostic.
    name = "pressure_at_novel_position"
    result = parse(name, vocabs=vocabs)
    assert result.ir.locus is not None
    assert result.ir.locus.token == "novel_position"
    assert any(
        isinstance(d, Diagnostic) and d.category == "vocab_gap"
        for d in result.diagnostics
    )


# ---------------------------------------------------------------------------
# Canonical bare-quantity and qualifier handling
# ---------------------------------------------------------------------------


def test_parse_bare_quantity(vocabs: Vocabularies):
    result = parse("pressure", vocabs=vocabs)
    assert result.ir.base.token == "pressure"
    assert result.ir.operators == []
    assert result.ir.qualifiers == []
    assert result.ir.locus is None
    assert compose(result.ir) == "pressure"


def test_parse_qualifier_plus_base(vocabs: Vocabularies):
    result = parse("electron_pressure", vocabs=vocabs)
    assert [q.token for q in result.ir.qualifiers] == ["electron"]
    assert result.ir.base.token == "pressure"
    assert compose(result.ir) == "electron_pressure"


def test_parse_coordinate_projection_on_carrier(vocabs: Vocabularies):
    """Parse the short-form ``vertical_position_of_flux_loop`` projection."""
    name = "vertical_position_of_flux_loop"
    result = parse(name, vocabs=vocabs)
    assert result.ir.projection is not None
    assert result.ir.projection.shape is ProjectionShape.COORDINATE
    assert result.ir.base.token == "position"
    assert result.ir.base.kind is BaseKind.GEOMETRY
    assert result.ir.locus is not None and result.ir.locus.token == "flux_loop"
    assert compose(result.ir) == name
