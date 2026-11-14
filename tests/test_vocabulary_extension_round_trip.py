"""Round-trip tests for extended segment vocabulary tokens.

Validates parse + compose round-trip for extension tokens across the
process, subject, position, component, and transformation segments, plus
forbidden-pattern error hints and the deferred coordinate-prefix split.
"""

import pytest

from imas_standard_names.grammar.model import (
    compose_standard_name,
    parse_standard_name,
)
from imas_standard_names.grammar.model_types import (
    Component,
    Position,
    Process,
    Subject,
    Transformation,
)
from imas_standard_names.grammar.support import validate_forbidden_patterns

# ---------------------------------------------------------------------------
# Process tokens
# ---------------------------------------------------------------------------


class TestProcessTokens:
    """Verify process vocabulary additions."""

    @pytest.mark.parametrize(
        "token,enum_member",
        [
            ("e_cross_b_drift", Process.E_CROSS_B_DRIFT),
            ("thermal_fusion", Process.THERMAL_FUSION),
            # thermalization is the canonical fast-particle thermalisation token;
            # its reworded synonyms are gated (test_process_synonym_gate.py).
            ("thermalization", Process.THERMALIZATION),
            ("halo_current", Process.HALO_CURRENT),
            ("non_inductive_current_drive", Process.NON_INDUCTIVE_CURRENT_DRIVE),
        ],
    )
    def test_process_enum_membership(self, token, enum_member):
        assert Process(token) == enum_member

    @pytest.mark.parametrize(
        "process",
        [
            "e_cross_b_drift",
            "thermal_fusion",
            "thermalization",
            "halo_current",
            "non_inductive_current_drive",
        ],
    )
    def test_process_round_trip(self, process):
        name = f"power_due_to_{process}"
        parsed = parse_standard_name(name)
        assert parsed.process == Process(process)
        assert parsed.physical_base == "power"
        assert compose_standard_name(parsed) == name

    def test_e_cross_b_drift_with_subject(self):
        name = "electron_heat_flux_due_to_e_cross_b_drift"
        parsed = parse_standard_name(name)
        assert parsed.subject == Subject.ELECTRON
        assert parsed.process == Process.E_CROSS_B_DRIFT
        assert compose_standard_name(parsed) == name


# ---------------------------------------------------------------------------
# Subject tokens
# ---------------------------------------------------------------------------


class TestSubjectTokens:
    """Modifier-subjects are decomposed into population + species.

    thermal_electron / thermal_ion / suprathermal_electrons are no longer
    compound Subject tokens; the modifier is the orthogonal ``population``
    segment and the species is the ``subject``. (The old ``suprathermal_
    electrons`` plural is gone — the singular ``electron`` species is used.)
    """

    @pytest.mark.parametrize(
        "name,population,subject",
        [
            ("suprathermal_electron_temperature", "suprathermal", "electron"),
            ("thermal_electron_temperature", "thermal", "electron"),
            ("thermal_ion_temperature", "thermal", "ion"),
        ],
    )
    def test_population_decomposition_round_trip(self, name, population, subject):
        parsed = parse_standard_name(name)
        assert parsed.population == population
        assert parsed.subject == Subject(subject)
        assert parsed.physical_base == "temperature"
        assert compose_standard_name(parsed) == name

    def test_thermal_electron_density_at_position(self):
        name = "thermal_electron_density_at_magnetic_axis"
        parsed = parse_standard_name(name)
        assert parsed.population == "thermal"
        assert parsed.subject == Subject.ELECTRON
        assert parsed.physical_base == "density"
        assert parsed.position == Position.MAGNETIC_AXIS
        assert compose_standard_name(parsed) == name


# ---------------------------------------------------------------------------
# Position tokens
# ---------------------------------------------------------------------------


class TestPositionTokens:
    """Verify position vocabulary additions."""

    @pytest.mark.parametrize(
        "token,enum_member",
        [
            ("ferritic_element_centroid", Position.FERRITIC_ELEMENT_CENTROID),
            # neutron_detector is an entity rather than a position locus.
            pytest.param(
                "neutron_detector",
                getattr(Position, "NEUTRON_DETECTOR", None),
                marks=pytest.mark.xfail(
                    reason="neutron_detector reclassified as entity (Object) in the current grammar",
                    strict=True,
                ),
            ),
            ("measurement_position", Position.MEASUREMENT_POSITION),
        ],
    )
    def test_position_enum_membership(self, token, enum_member):
        assert Position(token) == enum_member

    @pytest.mark.parametrize(
        "position",
        [
            "ferritic_element_centroid",
            pytest.param(
                "neutron_detector",
                marks=pytest.mark.xfail(
                    reason="neutron_detector reclassified as entity (Object) in the current grammar",
                    strict=True,
                ),
            ),
            "measurement_position",
        ],
    )
    def test_position_at_round_trip(self, position):
        name = f"electron_temperature_at_{position}"
        parsed = parse_standard_name(name)
        assert parsed.position == Position(position)
        assert parsed.subject == Subject.ELECTRON
        assert parsed.physical_base == "temperature"
        assert compose_standard_name(parsed) == name

    @pytest.mark.parametrize(
        "position",
        [
            "ferritic_element_centroid",
            pytest.param(
                "neutron_detector",
                marks=pytest.mark.xfail(
                    reason="neutron_detector reclassified as entity (Object) in the current grammar",
                    strict=True,
                ),
            ),
            "measurement_position",
        ],
    )
    def test_position_of_geometry_round_trip(self, position):
        # A point's R coordinate is radial_coordinate_of_<X>, not
        # major_radius_of_<X> (which is now rejected). This still exercises the
        # position-locus round-trip via geometry.
        name = f"radial_coordinate_of_{position}"
        parsed = parse_standard_name(name)
        assert parsed.geometry == Position(position)
        assert compose_standard_name(parsed) == name


# ---------------------------------------------------------------------------
# Component tokens
# ---------------------------------------------------------------------------


class TestComponentTokens:
    """Verify the normalized coordinate-frame component tokens.

    ``normalized_parallel`` / ``normalized_perpendicular`` were removed
    Parallel/perpendicular are field-relative
    directions with no distinct normalized variant, so a ``normalized_<dir>``
    prefix on them could only ever encode the GyroBohm-normalization OPERATOR,
    which was being shadowed by the component token via longest-match. Only the
    normalized COORDINATE-frame directions (radial→ρ, vertical, toroidal→φ,
    poloidal→θ) survive as genuine component tokens.
    """

    @pytest.mark.parametrize(
        "token,enum_member",
        [
            ("normalized_radial", Component.NORMALIZED_RADIAL),
            ("normalized_vertical", Component.NORMALIZED_VERTICAL),
            ("normalized_toroidal", Component.NORMALIZED_TOROIDAL),
            ("normalized_poloidal", Component.NORMALIZED_POLOIDAL),
        ],
    )
    def test_component_enum_membership(self, token, enum_member):
        assert Component(token) == enum_member

    @pytest.mark.parametrize(
        "component",
        [
            "normalized_radial",
            "normalized_vertical",
            "normalized_toroidal",
            "normalized_poloidal",
        ],
    )
    def test_component_round_trip(self, component):
        name = f"{component}_magnetic_field"
        parsed = parse_standard_name(name)
        assert parsed.component == Component(component)
        assert parsed.physical_base == "magnetic_field"
        assert compose_standard_name(parsed) == name

    @pytest.mark.parametrize(
        "removed_token",
        ["normalized_parallel", "normalized_perpendicular"],
    )
    def test_removed_normalized_field_relative_directions(self, removed_token):
        """The field-relative normalized directions are no longer components.

        ``normalized`` peels as the operator and the projection resolves to the
        bare field-relative direction (``parallel`` / ``perpendicular``). The
        ``normalized_<dir>`` spelling is now non-canonical — it canonicalizes to
        ``<dir>_normalized_...`` (component leads the normalization qualifier).
        """
        assert not hasattr(Component, removed_token.upper())
        direction = removed_token.removeprefix("normalized_")
        canonical = f"{direction}_normalized_momentum_flux"
        parsed = parse_standard_name(canonical)
        assert parsed.component == Component(direction)


# ---------------------------------------------------------------------------
# Transformation tokens
# ---------------------------------------------------------------------------


class TestTransformationTokens:
    """Verify transformation vocabulary membership."""

    @pytest.mark.parametrize(
        "token,enum_member",
        [
            # These tokens belong to different operator roles.
            pytest.param(
                "ratio_of",
                getattr(Transformation, "RATIO_OF", None),
                marks=pytest.mark.xfail(
                    reason="ratio_of is a binary operator, not a transformation",
                    strict=True,
                ),
            ),
        ],
    )
    def test_transformation_enum_membership(self, token, enum_member):
        assert Transformation(token) == enum_member

    @pytest.mark.xfail(
        reason="ratio_of is a binary operator, not a transformation",
        strict=True,
    )
    def test_ratio_of_round_trip(self):
        name = "ratio_of_density"
        parsed = parse_standard_name(name)
        assert parsed.transformation == Transformation.RATIO_OF
        assert parsed.physical_base == "density"
        assert compose_standard_name(parsed) == name


class TestForbiddenPatterns:
    """Verify parser error hints for common mistakes."""

    def test_diamagnetic_component_of_forbidden(self):
        with pytest.raises(ValueError, match="diamagnetic.*drift qualifier"):
            validate_forbidden_patterns("diamagnetic_component_of_velocity")

    def test_diamagnetic_component_of_suggests_canonical(self):
        with pytest.raises(
            ValueError, match="radial_<subject>_velocity_due_to_diamagnetic_drift"
        ):
            validate_forbidden_patterns(
                "diamagnetic_component_of_electron_drift_velocity"
            )

    def test_density_ratio_forbidden(self):
        with pytest.raises(ValueError, match="ratio_of_<A>_density_to_<B>_density"):
            validate_forbidden_patterns("deuterium_to_tritium_density_ratio")

    def test_density_ratio_canonical_form_passes(self):
        """The canonical ratio_of_ form should not trigger the pattern."""
        # This should not raise
        validate_forbidden_patterns("ratio_of_deuterium_density_to_tritium_density")

    def test_normal_names_pass(self):
        """Regular names should not trigger any forbidden pattern."""
        validate_forbidden_patterns("electron_temperature")
        validate_forbidden_patterns("radial_magnetic_field")
        validate_forbidden_patterns("power_due_to_e_cross_b_drift")


# ---------------------------------------------------------------------------
# Regression — existing names still work
# ---------------------------------------------------------------------------


class TestExistingRoundTripRegression:
    """Verify representative established names still round-trip."""

    @pytest.mark.parametrize(
        "name",
        [
            "electron_temperature",
            "radial_magnetic_field",
            "flux_loop_voltage",
            "area_of_flux_loop",
            "electron_temperature_at_magnetic_axis",
            "radial_coordinate_of_plasma_boundary",
            "power_due_to_ohmic_heating",
            "flux_surface_averaged_density",
            # real_part is a postfix scalar-extraction operator: <base>_real_part
            "magnetic_field_real_part",
            "cumulative_inside_flux_surface_power",
            "power_due_to_collisions",
            "electron_temperature_at_ferritic_insert_centroid",
        ],
    )
    def test_round_trip(self, name):
        parsed = parse_standard_name(name)
        assert compose_standard_name(parsed) == name
