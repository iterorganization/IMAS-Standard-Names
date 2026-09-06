"""Parser tests for two grammar fixes.

1. Indexed flux-coordinate derivative prefix operator.

   ``derivative_with_respect_to`` is declared in ``operators.yml`` as a
   ``unary_prefix`` with ``index_params: [coord]``. Its canonical prefix
   form places the coordinate index after the complete operand::

       derivative_of_<base>_with_respect_to_<coord>

   where ``<coord>`` is a registered coordinate / flux-coordinate token
   (geometry-carrier vocabulary, e.g. ``radial_coordinate``,
   ``toroidal_flux_coordinate``). The bound form must parse AND round-trip:
   ``compose(parse(s).ir) == s``.

2. ``over_<X>`` locus validation against the region vocabulary.

   The ``over`` relation is only valid for region-typed loci. An
   ``over_<X>`` suffix whose ``<X>`` is not a registered region must NOT
   strip as a spurious region locus — it must raise :class:`ParseError`
   so the author is forced toward the correct construction (e.g.
   ``ratio_of_velocity_to_magnetic_field`` rather than
   ``velocity_over_magnetic_field``).
"""

from __future__ import annotations

import pytest

from imas_standard_names.grammar.ir import OperatorKind
from imas_standard_names.grammar.parser import (
    ParseError,
    Vocabularies,
    load_default_vocabularies,
    parse,
    validate_round_trip,
)
from imas_standard_names.grammar.render import compose


@pytest.fixture(scope="module")
def vocabs() -> Vocabularies:
    return load_default_vocabularies()


# ---------------------------------------------------------------------------
# Indexed flux-coordinate derivative prefix
# ---------------------------------------------------------------------------

# Coordinate tokens drawn from the registered coordinate / flux-coordinate
# (geometry-carrier) vocabulary, paired with a registered physical base.
_FLUX_DERIVATIVE_NAMES = [
    "derivative_of_volume_with_respect_to_radial_coordinate",
    "derivative_of_pressure_with_respect_to_toroidal_flux_coordinate",
    "derivative_of_pressure_with_respect_to_normalized_toroidal_flux_coordinate",
    "derivative_of_volume_with_respect_to_normalized_poloidal_flux_coordinate",
]


@pytest.mark.parametrize("name", _FLUX_DERIVATIVE_NAMES)
def test_indexed_flux_derivative_parses(name: str, vocabs: Vocabularies) -> None:
    """The declared prefix indexed form parses to a single prefix operator."""
    result = parse(name, vocabs=vocabs)
    assert len(result.ir.operators) == 1
    op = result.ir.operators[0]
    assert op.kind is OperatorKind.UNARY_PREFIX
    assert op.op.startswith("derivative_with_respect_to_")


@pytest.mark.parametrize("name", _FLUX_DERIVATIVE_NAMES)
def test_indexed_flux_derivative_round_trips(name: str, vocabs: Vocabularies) -> None:
    """``compose(parse(s).ir) == s`` for the indexed flux-derivative form."""
    result = parse(name, vocabs=vocabs)
    assert compose(result.ir) == name
    assert validate_round_trip(name, vocabs=vocabs)


def test_indexed_flux_derivative_binds_coordinate(vocabs: Vocabularies) -> None:
    """The coordinate index is fused into the operator token; base is the inner."""
    name = "derivative_of_volume_with_respect_to_radial_coordinate"
    result = parse(name, vocabs=vocabs)
    op = result.ir.operators[0]
    assert op.op == "derivative_with_respect_to_radial_coordinate"
    assert result.ir.base.token == "volume"


def test_indexed_flux_derivative_rejects_unregistered_coordinate(
    vocabs: Vocabularies,
) -> None:
    """A ``<coord>`` not in the coordinate/flux-coordinate vocabulary fails.

    ``banana`` is not a registered coordinate, so the operator cannot bind
    its index and the residue does not resolve to a base.
    """
    with pytest.raises(ParseError):
        parse("derivative_of_volume_with_respect_to_banana", vocabs=vocabs)


def test_indexed_flux_derivative_nested_under_maximum(
    vocabs: Vocabularies,
) -> None:
    """Indexed derivative nests under an outer prefix operator and round-trips."""
    name = "maximum_of_derivative_of_pressure_with_respect_to_radial_coordinate"
    result = parse(name, vocabs=vocabs)
    assert [op.op for op in result.ir.operators] == [
        "maximum",
        "derivative_with_respect_to_radial_coordinate",
    ]
    assert compose(result.ir) == name


# ---------------------------------------------------------------------------
# over_<X> must validate against the region vocabulary
# ---------------------------------------------------------------------------

# Names whose over_<X> target is NOT a registered region; must be rejected.
_INVALID_OVER_NAMES = [
    "velocity_over_magnetic_field",
    "vorticity_over_r",
    "temperature_over_some_region",
    "temperature_over_unknown_region_token",
]


@pytest.mark.parametrize("name", _INVALID_OVER_NAMES)
def test_over_unregistered_region_rejected(name: str, vocabs: Vocabularies) -> None:
    """``over_<X>`` with an unregistered region raises ParseError."""
    with pytest.raises(ParseError):
        parse(name, vocabs=vocabs)


def test_over_registered_region_still_parses(vocabs: Vocabularies) -> None:
    """A genuine ``over_<region>`` (registered region) still parses + round-trips."""
    name = "pressure_over_core_region"
    result = parse(name, vocabs=vocabs)
    assert result.ir.locus is not None
    assert result.ir.locus.token == "core_region"
    assert result.ir.locus.relation.value == "over"
    assert compose(result.ir) == name
