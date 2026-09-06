"""Model and ordered-IR round-trip for nested operator names.

These names carry an outer unary operator (a prefix transformation like
``time_derivative``/``gradient`` or a postfix ``magnitude``) wrapping an
inner expression that itself contains a bare-prefix transformation
qualifier (``volume_averaged``, ``flux_surface_averaged``,
``normalized``, ...).

The flat :class:`StandardName` model has a single ``transformation`` and
a single ``decomposition`` slot. A compatible inner bare-prefix expression
therefore occupies one ``physical_base`` compound string, while the ordered IR
retains every operator explicitly.

A single transformation can coexist with a projection axis
(``time_derivative_of_radial_electric_field`` — the projection lives in
``component``, the operator in ``transformation``; see
``test_operator_projection_coexistence.py``). The flat model still cannot
represent two structurally distinct prefix operators
(``gradient_of_time_derivative_of_...``); those names are a documented
limitation and must keep raising rather than silently dropping tokens.
"""

import pytest

from imas_standard_names.grammar.model import (
    compose_standard_name,
    parse_standard_name,
)

# Names that must survive a full model-level round-trip:
# outer unary operator (prefix or postfix) wrapping an inner expression
# whose canonical spelling contains a bare-prefix transformation qualifier,
# optionally with a trailing locus on the outer model.
FLAT_MODEL_FORMS = [
    # of-prefix outer operator + inner bare-transformation + species subject
    "maximum_of_volume_averaged_electron_density",
    "maximum_of_flux_surface_averaged_electron_density",
    "gradient_of_normalized_electron_temperature",
    # postfix outer operator + inner bare-transformation
    "volume_averaged_magnetic_field_magnitude",
    # outer operator + inner bare-transformation + trailing locus
    "electron_density_volume_averaged_maximum_at_magnetic_axis",
    # of-prefix transformation wrapping a projection axis: the projection
    # stays in `component`, no fold, so it round-trips (transformation ×
    # component coexistence).
    "time_derivative_of_radial_electric_field",
]

# Single-operator and binary baseline forms.
BASELINE_FORMS = [
    "time_derivative_of_electron_density",
    "gradient_of_electron_pressure",
    "volume_averaged_electron_density",
    "magnetic_field_magnitude",
    "ratio_of_electron_to_ion_temperature",
]

# Names the flat model provably cannot represent. Folding the inner
# expression into physical_base loses a token (a projection axis, or a
# second structurally-distinct unary operator), so the strict
# lossless-canonical guard must reject them rather than silently emit a
# token-dropped name. Documented limitation; tracked for a future nested
# model. Each entry is (name, the token folding would drop).
IR_ONLY_FORMS = [
    # operator-of-operator: two structurally-distinct prefix operators;
    # folding drops the inner 'time_derivative'
    "gradient_of_time_derivative_of_electron_temperature",
    # Higher-precedence bare prefixes canonically wrap these explicit
    # operators, and the flat model has no second transformation slot.
    "volume_averaged_time_derivative_of_electron_density",
]


@pytest.mark.parametrize("name", FLAT_MODEL_FORMS)
def test_flat_model_nested_operator_round_trips(name: str):
    model = parse_standard_name(name)
    assert compose_standard_name(model) == name


@pytest.mark.parametrize("name", BASELINE_FORMS)
def test_single_operator_and_binary_names_round_trip(name: str):
    model = parse_standard_name(name)
    assert compose_standard_name(model) == name


@pytest.mark.parametrize("name", IR_ONLY_FORMS)
def test_flat_model_rejects_multi_prefix_chains(name: str):
    """The flat model cannot represent these; they must raise, never
    silently drop a token. The lossless-canonical guard is the safety net.
    """
    with pytest.raises(ValueError):
        parse_standard_name(name)


@pytest.mark.parametrize("name", IR_ONLY_FORMS)
def test_lossless_ir_strictly_validates_nested_prefix_chains(name: str) -> None:
    """The ordered IR validates structures the flat facade cannot project."""
    from imas_standard_names import compose, parse

    assert compose(parse(name, strict=True).ir) == name


class TestUnaryOverBinary:
    """A unary operator wrapping a binary expression round-trips at the IR
    layer and raises an explicit (not token-drop) error at the model layer.

    Guards the parser path that terminates operator peeling at a binary:
    prefix operators peeled before the binary terminator must stay on the
    IR operator stack instead of being silently discarded.
    """

    NAME = "gradient_of_ratio_of_electron_pressure_to_magnetic_pressure"

    def test_ir_round_trip_preserves_outer_operator(self):
        from imas_standard_names.grammar.parser import parse
        from imas_standard_names.grammar.render import compose

        assert compose(parse(self.NAME).ir) == self.NAME

    def test_validate_round_trip(self):
        from imas_standard_names.grammar.parser import validate_round_trip

        assert validate_round_trip(self.NAME)

    def test_postfix_over_binary_ir_round_trip(self):
        from imas_standard_names.grammar.parser import parse
        from imas_standard_names.grammar.render import compose

        name = "ratio_of_electron_pressure_to_magnetic_pressure_magnitude"
        assert compose(parse(name).ir) == name

    def test_flat_model_raises_explicit_error(self):
        with pytest.raises(ValueError, match="not representable"):
            parse_standard_name(self.NAME)
