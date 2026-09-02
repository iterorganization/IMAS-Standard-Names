"""Tests for dimensionless unit validation in semantic checks."""

from __future__ import annotations

from imas_standard_names.models import (
    StandardNameMetadataEntry,
    StandardNameScalarEntry,
    StandardNameVectorEntry,
)
from imas_standard_names.validation.semantic import (
    _check_dimensionless_physical_quantity,
    _check_none_unit_with_quantitative_kind,
)


def _scalar(name: str, unit: str = "eV") -> StandardNameScalarEntry:
    return StandardNameScalarEntry(
        name=name,
        kind="scalar",
        physics_domain="general",
        unit=unit,
        description="Test entry",
        documentation="Test documentation for validation.",
        status="draft",
    )


def _vector(name: str, unit: str = "m.s^-1") -> StandardNameVectorEntry:
    return StandardNameVectorEntry(
        name=name,
        kind="vector",
        physics_domain="general",
        unit=unit,
        description="Test entry",
        documentation="Test documentation for validation.",
        status="draft",
    )


def _metadata(name: str) -> StandardNameMetadataEntry:
    return StandardNameMetadataEntry(
        name=name,
        kind="metadata",
        physics_domain="general",
        description="Test entry",
        documentation="Test documentation for validation.",
        status="draft",
    )


class TestDimensionlessPhysicalQuantity:
    """Test _check_dimensionless_physical_quantity semantic check."""

    def test_warns_temperature_with_unit_one(self):
        entry = _scalar("electron_temperature", unit="1")
        issues = _check_dimensionless_physical_quantity("electron_temperature", entry)
        assert len(issues) == 1
        assert "WARNING" in issues[0]
        assert "temperature" in issues[0]
        assert "dimensionless" in issues[0]

    def test_warns_density_with_unit_one(self):
        entry = _scalar("electron_density", unit="1")
        issues = _check_dimensionless_physical_quantity("electron_density", entry)
        assert len(issues) == 1
        assert "density" in issues[0]

    def test_warns_pressure_with_unit_one(self):
        entry = _scalar("electron_pressure", unit="1")
        issues = _check_dimensionless_physical_quantity("electron_pressure", entry)
        assert len(issues) == 1
        assert "pressure" in issues[0]

    def test_warns_velocity_with_unit_one(self):
        entry = _scalar("ion_velocity", unit="1")
        issues = _check_dimensionless_physical_quantity("ion_velocity", entry)
        assert len(issues) == 1
        assert "velocity" in issues[0]

    def test_no_warning_for_proper_unit(self):
        entry = _scalar("electron_temperature", unit="eV")
        issues = _check_dimensionless_physical_quantity("electron_temperature", entry)
        assert issues == []

    def test_no_warning_for_non_dimensional_base(self):
        """Names with bases not in the inherently-dimensional set should pass."""
        entry = _scalar("safety_factor", unit="1")
        issues = _check_dimensionless_physical_quantity("safety_factor", entry)
        assert issues == []

    def test_no_warning_for_ratio_binary_operator(self):
        """Binary operator names (ratios) can legitimately be dimensionless."""
        entry = _scalar("ratio_of_electron_temperature_to_ion_temperature", unit="1")
        issues = _check_dimensionless_physical_quantity(
            "ratio_of_electron_temperature_to_ion_temperature", entry
        )
        assert issues == []

    def test_no_warning_for_product_binary_operator(self):
        entry = _scalar("product_of_density_and_velocity", unit="1")
        issues = _check_dimensionless_physical_quantity(
            "product_of_density_and_velocity", entry
        )
        assert issues == []

    def test_no_warning_for_metadata_entry(self):
        entry = _metadata("plasma_boundary")
        issues = _check_dimensionless_physical_quantity("plasma_boundary", entry)
        assert issues == []

    def test_vector_with_unit_one_warns(self):
        entry = _vector("magnetic_field", unit="1")
        issues = _check_dimensionless_physical_quantity("magnetic_field", entry)
        assert len(issues) == 1
        assert "magnetic_field" in issues[0]

    def test_warns_for_area(self):
        entry = _scalar("area_of_flux_loop", unit="1")
        issues = _check_dimensionless_physical_quantity("area_of_flux_loop", entry)
        assert len(issues) == 1
        assert "area" in issues[0]

    def test_warns_for_energy(self):
        entry = _scalar("electron_energy", unit="1")
        issues = _check_dimensionless_physical_quantity("electron_energy", entry)
        assert len(issues) == 1
        assert "energy" in issues[0]

    def test_no_warning_for_different_unit(self):
        """Non-'1' units should never trigger this check."""
        entry = _scalar("electron_temperature", unit="K")
        issues = _check_dimensionless_physical_quantity("electron_temperature", entry)
        assert issues == []

    def test_no_warning_for_dimension_transforming_operator(self):
        """Integral operators change the base's dimensions — a volume-integrated
        density is a count, so unit '1' is legitimate."""
        name = "volume_integrated_runaway_electron_density"
        entry = _scalar(name, unit="1")
        assert _check_dimensionless_physical_quantity(name, entry) == []

    def test_no_warning_for_derivative_operator(self):
        name = "derivative_of_electron_temperature"
        entry = _scalar(name, unit="1")
        assert _check_dimensionless_physical_quantity(name, entry) == []

    def test_dimension_preserving_operator_still_warns(self):
        """Averaging preserves dimensions — a volume-averaged density with
        unit '1' is still suspicious."""
        name = "volume_averaged_electron_density"
        entry = _scalar(name, unit="1")
        issues = _check_dimensionless_physical_quantity(name, entry)
        assert len(issues) == 1
        assert "density" in issues[0]

    def test_no_warning_for_normalized_operator_on_projection(self):
        """A GyroBohm-normalized projected flux is legitimately dimensionless.

        ``normalized`` peels as the dimensionless operator (the projection
        resolves to ``parallel``), so it is exempted via the dimensionless-
        operator/qualifier path — no unit-suspicion warning."""
        name = (
            "normalized_parallel_momentum_flux_due_to_"
            "perturbed_parallel_vector_potential"
        )
        entry = _scalar(name, unit="1")
        assert _check_dimensionless_physical_quantity(name, entry) == []


class TestNoneUnitWithQuantitativeKind:
    """Test _check_none_unit_with_quantitative_kind semantic check."""

    def test_none_unit_with_metadata_is_valid(self):
        """Metadata entries are allowed to have no unit."""
        entry = _metadata("plasma_boundary")
        issues = _check_none_unit_with_quantitative_kind("plasma_boundary", entry)
        assert issues == []

    def test_scalar_with_proper_unit_is_valid(self):
        """Scalar entries with explicit units are fine."""
        entry = _scalar("electron_temperature", unit="eV")
        issues = _check_none_unit_with_quantitative_kind("electron_temperature", entry)
        assert issues == []

    def test_scalar_with_dimensionless_unit_is_valid(self):
        """Scalar entries with unit='1' (dimensionless) are fine."""
        entry = _scalar("safety_factor", unit="1")
        issues = _check_none_unit_with_quantitative_kind("safety_factor", entry)
        assert issues == []

    def test_vector_with_proper_unit_is_valid(self):
        """Vector entries with explicit units are fine."""
        entry = _vector("magnetic_field", unit="T")
        issues = _check_none_unit_with_quantitative_kind("magnetic_field", entry)
        assert issues == []

    def test_scalar_with_none_unit_warns(self):
        """Scalar entries with unit=None should produce a warning."""
        entry = StandardNameScalarEntry.model_construct(
            name="electron_temperature",
            kind="scalar",
            physics_domain="core_plasma_physics",
            unit=None,
            description="Test entry",
            documentation="Test docs.",
            status="draft",
        )
        issues = _check_none_unit_with_quantitative_kind("electron_temperature", entry)
        assert len(issues) == 1
        assert "WARNING" in issues[0]
        assert "unit" in issues[0].lower()

    def test_vector_with_none_unit_warns(self):
        """Vector entries with unit=None should produce a warning."""
        entry = StandardNameVectorEntry.model_construct(
            name="magnetic_field",
            kind="vector",
            physics_domain="equilibrium",
            unit=None,
            description="Test entry",
            documentation="Test docs.",
            status="draft",
        )
        issues = _check_none_unit_with_quantitative_kind("magnetic_field", entry)
        assert len(issues) == 1
        assert "WARNING" in issues[0]
        assert "vector" in issues[0].lower()
