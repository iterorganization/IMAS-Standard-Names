"""Model-layer round-trip for the fused indexed flux-coordinate derivative.

``derivative_with_respect_to`` is an indexed unary-prefix operator
(``index_params: [coord]``). The parser fuses the coordinate index into the
operator token (``derivative_with_respect_to_<coord>``) so the canonical form
``derivative_of_<base>_with_respect_to_<coord>`` round-trips. The downstream
composer goes through the flat ``StandardName`` model
(``parse_standard_name`` / ``compose_standard_name``), where the fused token
must be accepted in the ``transformation`` slot even though it is not a bare
member of the closed ``Transformation`` StrEnum.

This complements ``test_indexed_derivative_and_over_locus.py`` (which only
exercises the PARSER path) by asserting the MODEL path.
"""

from __future__ import annotations

import itertools

import pytest

from imas_standard_names.grammar.model import (
    compose_standard_name,
    parse_standard_name,
)

# Coordinate indices drawn from the registered coordinate / flux-coordinate
# vocabulary, paired with representative bases.
_COORDS = [
    "radial_coordinate",
    "toroidal_flux_coordinate",
    "normalized_poloidal_flux_coordinate",
]
_BASES = ["volume", "pressure", "area"]

_INDEXED_DERIVATIVE_NAMES = [
    f"derivative_of_{base}_with_respect_to_{coord}"
    for coord, base in itertools.product(_COORDS, _BASES)
]


@pytest.mark.parametrize("name", _INDEXED_DERIVATIVE_NAMES)
def test_indexed_derivative_model_round_trips(name: str) -> None:
    """The fused indexed-derivative name round-trips through the model layer."""
    assert compose_standard_name(parse_standard_name(name)) == name


@pytest.mark.parametrize("name", _INDEXED_DERIVATIVE_NAMES)
def test_indexed_derivative_carries_fused_transformation(name: str) -> None:
    """The fused ``<op>_<coord>`` token lands in the model ``transformation``
    slot (not split or dropped)."""
    model = parse_standard_name(name)
    assert model.transformation is not None
    assert model.transformation.startswith("derivative_with_respect_to_")
    # The coordinate index is preserved verbatim in the fused token.
    _, _, coord = name.partition("_with_respect_to_")
    assert model.transformation == f"derivative_with_respect_to_{coord}"


def test_indexed_derivative_rejects_unregistered_coordinate() -> None:
    """A fused token whose index is not a registered coordinate is rejected by
    the model field validator (it is neither a bare operator nor a valid fused
    indexed form)."""
    from imas_standard_names.grammar.model import StandardName

    with pytest.raises(ValueError):
        StandardName(
            physical_base="pressure",
            transformation="derivative_with_respect_to_banana",
        )


def test_bare_registered_transformation_still_valid() -> None:
    """A bare registered operator token (enum member) still validates."""
    from imas_standard_names.grammar.model import StandardName

    model = StandardName(physical_base="pressure", transformation="gradient")
    assert model.transformation == "gradient"
