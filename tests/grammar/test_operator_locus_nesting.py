"""Model-level round-trip for postfix operators stacked with a locus,
mechanism, or a prefix transformation.

These real-catalog forms combine a **postfix** decomposition operator
(``magnitude``) with a locus/mechanism suffix, or nest a prefix transformation
with a postfix decomposition. The renderer places the operator immediately
after the base group and before the tail.

Mechanism and locus stripping must preserve the postfix token as an operator,
rather than absorbing it into a fabricated process or locus token. The flat
:class:`StandardName` model also retains prefix and postfix operators in their
distinct transformation and decomposition slots.

Canonical-order decisions encoded here:

* postfix-before-locus / postfix-before-mechanism is canonical
  (``..._magnitude_of_iron_core_segment``, ``..._magnitude_due_to_<process>``);
* prefix transformation is outermost, postfix decomposition innermost
  (``maximum_of_<base>_magnitude``);
* an extremum transformation qualifier (``maximum_<base>``) coexists with a
  locus and a postfix operator (``maximum_<base>_magnitude_of_<locus>``).

The lossless-canonical guard remains the safety net: any name the flat
model cannot represent must raise, never drop a token.
"""

import pytest

from imas_standard_names.grammar.model import (
    compose_standard_name,
    parse_standard_name,
)

# Postfix decomposition stacked with a locus / mechanism / prefix operator.
# Every token is registered (verified against the live vocabularies):
#   iron_core_segment -> object (entity locus, _of_)
#   flux_surface      -> position/geometry locus (_of_)
#   pellet_injection  -> process (mechanism, _due_to_)
#   magnitude         -> unary_postfix operator
#   maximum           -> unary_prefix operator (and bare-prefix qualifier)
IN_SCOPE = [
    # Postfix operator before an entity locus.
    "magnetic_field_magnitude_of_iron_core_segment",
    # Postfix operator before a mechanism.
    "velocity_magnitude_due_to_pellet_injection",
    # prefix transformation (of-form) wrapping a postfix decomposition
    "maximum_of_magnetic_field_magnitude",
    # bare-prefix transformation qualifier + locus + postfix decomposition
    "maximum_magnetic_field_magnitude_of_flux_surface",
]

# Component pieces that already round-tripped in isolation and must keep
# doing so (regression guard — the parser reorder must not disturb
# the simpler forms).
ALREADY_WORKING = [
    "magnetic_field_magnitude",
    "magnetic_field_of_iron_core_segment",
    "velocity_due_to_pellet_injection",
    "maximum_magnetic_field",
    "maximum_magnetic_field_magnitude",
    "maximum_magnetic_field_of_flux_surface",
    "safety_factor_at_magnetic_axis",
]


@pytest.mark.parametrize("name", IN_SCOPE)
def test_in_scope_operator_locus_round_trips(name: str) -> None:
    model = parse_standard_name(name)
    assert compose_standard_name(model) == name


@pytest.mark.parametrize("name", ALREADY_WORKING)
def test_already_working_names_still_round_trip(name: str) -> None:
    model = parse_standard_name(name)
    assert compose_standard_name(model) == name


def test_mechanism_postfix_does_not_fabricate_process_token() -> None:
    """The postfix operator must not be absorbed into the process token.

    The danger case: a greedy mechanism strip swallows ``_magnitude`` into a
    fabricated ``pellet_injection_magnitude`` process token (not in the closed
    process vocabulary). The IR-level parse happened to re-render the same
    string, masking the lost operator. Assert the model carries the real
    process token and a real decomposition slot.
    """
    model = parse_standard_name("velocity_magnitude_due_to_pellet_injection")
    dump = model.model_dump_compact()
    assert dump.get("process") == "pellet_injection"
    assert dump.get("decomposition") == "magnitude"


def test_prefix_postfix_nest_populates_both_slots() -> None:
    """``maximum_of_magnetic_field_magnitude`` carries both operator slots.

    Prefix transformation and postfix decomposition occupy distinct,
    non-ambiguous slots (prefix renders ``maximum_of_<...>``, postfix renders
    ``<...>_magnitude``). The flat model must accept both together.
    """
    model = parse_standard_name("maximum_of_magnetic_field_magnitude")
    dump = model.model_dump_compact()
    assert dump.get("transformation") == "maximum"
    assert dump.get("decomposition") == "magnitude"
    assert dump.get("physical_base") == "magnetic_field"
