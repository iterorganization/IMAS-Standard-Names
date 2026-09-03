"""Model-level round-trip for a transformation stacked with a projection.

A directional **component** projection (``toroidal``, ``radial``, ...) now
coexists with a transformation (prefix operator). The two occupy distinct,
unambiguous positions in the canonical string, and the form depends on how the
transformation renders:

* an ``_of_``-form transformation renders OUTERMOST, wrapping the projected
  base — ``time_derivative_of_toroidal_current_density``,
  ``gradient_of_radial_electron_temperature``. The parser peels the leading
  ``<op>_of_`` first, then resolves the projection inside the residue.
* a BARE-prefix transformation (``change_in``, ``volume_averaged``, ...) folds
  into the qualifier run, so the component stays outermost —
  ``poloidal_change_in_ion_velocity``, ``radial_change_in_ion_temperature``.

Either way the model carries the projection in ``component`` and the operator
in ``transformation``; the renderer picks the single canonical spelling, so the
compose model emits the two as separate segments and never spells the surface
order itself.

``change_in`` is the new operator added alongside this relaxation: a
finite/discrete increment (Δ) that preserves the base unit (NOT a per-time
rate like ``time_derivative``) and renders bare.

Retained limits (must still RAISE — never drop a token):

* ``transformation`` + ``geometric_base`` — a geometry carrier has no
  field/operator structure to transform.
* a coordinate projection with a transformation — ``coordinate`` requires a
  ``geometric_base``, which the transformation rule forbids, so the pairing is
  unrepresentable even though the exclusivity validator no longer blocks it.
* two stacked PREFIX operators (``change_in`` of a ``volume_averaged`` base) —
  the flat model has a single transformation slot.
* the component-OUTERMOST spelling of an ``_of_`` operator
  (``toroidal_time_derivative_of_current_density``) is NOT canonical; the canonical is
  ``time_derivative_of_toroidal_current_density``.
"""

from __future__ import annotations

import pytest

from imas_standard_names.grammar.model import (
    compose_standard_name,
    parse_standard_name,
)

# Transformation + component projection — these raised before the
# transformation×component exclusivity was relaxed. Every token is registered.
COEXISTENCE_ROUND_TRIP = [
    # _of_-form transformation wraps the component (operator outermost)
    "time_derivative_of_toroidal_current_density",
    "gradient_of_radial_electron_temperature",
    "time_derivative_of_radial_magnetic_field",
    "gradient_of_parallel_electron_pressure",
    # bare-prefix transformation folds in; component stays outermost
    "poloidal_change_in_ion_velocity",
    "radial_change_in_ion_temperature",
    "toroidal_change_in_current_density",
]

# change_in operator (with and without a projection / subject).
CHANGE_IN_ROUND_TRIP = [
    "change_in_electron_density",
    "change_in_ion_temperature",
    "poloidal_change_in_ion_velocity",
]

# Simpler forms that already round-tripped and must keep doing so (the
# relaxation must not disturb a lone projection or a lone transformation).
ALREADY_WORKING = [
    "toroidal_current_density",
    "time_derivative_of_current_density",
    "time_derivative_of_electron_density",
    "volume_averaged_electron_density",
    "radial_magnetic_field",
]

# Forms that must still RAISE — never silently drop a token.
MUST_RAISE = [
    # component-outermost spelling of an _of_ operator is NOT canonical
    "toroidal_time_derivative_of_current_density",
    "radial_gradient_of_electron_temperature",
    # two stacked PREFIX operators (single transformation slot)
    "change_in_volume_averaged_electron_density",
]


@pytest.mark.parametrize("name", COEXISTENCE_ROUND_TRIP)
def test_transformation_projection_coexistence_round_trips(name: str) -> None:
    model = parse_standard_name(name)
    assert compose_standard_name(model) == name


@pytest.mark.parametrize("name", CHANGE_IN_ROUND_TRIP)
def test_change_in_round_trips(name: str) -> None:
    model = parse_standard_name(name)
    assert compose_standard_name(model) == name


@pytest.mark.parametrize("name", ALREADY_WORKING)
def test_simpler_forms_still_round_trip(name: str) -> None:
    model = parse_standard_name(name)
    assert compose_standard_name(model) == name


@pytest.mark.parametrize("name", MUST_RAISE)
def test_non_canonical_or_unrepresentable_forms_raise(name: str) -> None:
    # UnknownBaseTokenError / NonCanonicalNameError / ParseError all derive
    # from ValueError; the contract is that these RAISE rather than silently
    # round-tripping a dropped token.
    with pytest.raises(ValueError):
        parse_standard_name(name)


def test_change_in_renders_bare_in_transformation_slot() -> None:
    """``change_in`` sits in the transformation slot and renders bare.

    Unlike ``time_derivative`` (``_of_`` form), ``change_in``
    attaches directly to the base (``change_in_electron_density``) — it must
    NOT render ``change_in_of_...``.
    """
    model = parse_standard_name("change_in_electron_density")
    dump = model.model_dump_compact()
    assert dump.get("transformation") == "change_in"
    assert "change_in_of_" not in compose_standard_name(model)


def test_of_form_transformation_with_component_populates_both_slots() -> None:
    """``time_derivative_of_toroidal_current_density`` carries BOTH slots.

    The projection lives in ``component`` and the operator in
    ``transformation``; they no longer collide.
    """
    model = parse_standard_name("time_derivative_of_toroidal_current_density")
    dump = model.model_dump_compact()
    assert dump.get("component") == "toroidal"
    assert dump.get("transformation") == "time_derivative"
    assert dump.get("physical_base") == "current_density"


def test_bare_transformation_with_component_populates_both_slots() -> None:
    """``poloidal_change_in_ion_velocity`` carries component + transformation."""
    model = parse_standard_name("poloidal_change_in_ion_velocity")
    dump = model.model_dump_compact()
    assert dump.get("component") == "poloidal"
    assert dump.get("transformation") == "change_in"
    assert dump.get("subject") == "ion"
    assert dump.get("physical_base") == "velocity"


def test_transformation_with_geometric_base_still_raises() -> None:
    """A transformation cannot act on a geometry carrier (retained limit)."""
    with pytest.raises(ValueError, match="transformation.*geometric_base"):
        compose_standard_name(
            {
                "transformation": "gradient",
                "geometric_base": "position",
                "geometry": "magnetic_axis",
            }
        )
