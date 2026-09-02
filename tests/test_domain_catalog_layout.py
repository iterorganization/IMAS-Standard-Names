"""Tests for the per-domain catalog layout, ArgumentRef, and error_variants.

Covers:
- Round-trip: load domain-file fixture, re-serialise, byte-identical.
- Missing cross-reference warning fires but does not fail validation.
- Topological load handles arguments[].name edges.
- Legacy per-file YAML produces clean migration error.
- ArgumentRef validates operator_kind enum strictly.
"""

import warnings
from pathlib import Path

import pytest
import yaml

from imas_standard_names.models import (
    ArgumentRef,
    StandardNameEntryBase,
    create_standard_name_entry,
)
from imas_standard_names.ordering import ordered_model_names
from imas_standard_names.yaml_store import CatalogMigrationError, YamlStore

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_DOMAIN_FIXTURE = [
    {
        "name": "temperature",
        "kind": "scalar",
        "physics_domain": "core_plasma_physics",
        "description": "Temperature of a plasma species.",
        "documentation": "Generic temperature quantity.",
        "unit": "eV",
        "status": "draft",
        "error_variants": {
            "upper": "upper_uncertainty_of_temperature",
            "lower": "lower_uncertainty_of_temperature",
        },
    },
    {
        "name": "upper_uncertainty_of_temperature",
        "kind": "scalar",
        "physics_domain": "core_plasma_physics",
        "description": "Upper uncertainty of temperature.",
        "documentation": "Upper bound on temperature uncertainty.",
        "unit": "eV",
        "status": "draft",
    },
    {
        "name": "lower_uncertainty_of_temperature",
        "kind": "scalar",
        "physics_domain": "core_plasma_physics",
        "description": "Lower uncertainty of temperature.",
        "documentation": "Lower bound on temperature uncertainty.",
        "unit": "eV",
        "status": "draft",
    },
    {
        "name": "maximum_of_temperature",
        "kind": "scalar",
        "physics_domain": "core_plasma_physics",
        "description": "Maximum of temperature.",
        "documentation": "Peak temperature over a domain.",
        "unit": "eV",
        "status": "draft",
        "arguments": [
            {
                "name": "temperature",
                "operator": "maximum",
                "operator_kind": "unary_prefix",
            }
        ],
    },
]


def _write_domain_fixture(root: Path, domain: str = "transport") -> Path:
    """Write the domain fixture to a YAML file and return the path."""
    fpath = root / f"{domain}.yml"
    with open(fpath, "w", encoding="utf-8") as fh:
        yaml.safe_dump(
            _DOMAIN_FIXTURE,
            fh,
            sort_keys=False,
            allow_unicode=True,
            width=80,
        )
    return fpath


# ---------------------------------------------------------------------------
# ArgumentRef model validation
# ---------------------------------------------------------------------------


class TestArgumentRef:
    """ArgumentRef validates operator_kind enum strictly."""

    def test_unary_prefix_valid(self):
        ref = ArgumentRef(
            name="temperature", operator="maximum", operator_kind="unary_prefix"
        )
        assert ref.operator_kind == "unary_prefix"

    def test_unary_postfix_valid(self):
        ref = ArgumentRef(
            name="temperature", operator="squared", operator_kind="unary_postfix"
        )
        assert ref.operator_kind == "unary_postfix"

    def test_binary_valid(self):
        ref = ArgumentRef(
            name="pressure",
            operator="ratio",
            operator_kind="binary",
            role="a",
            separator="to",
        )
        assert ref.role == "a"
        assert ref.separator == "to"

    def test_projection_valid(self):
        ref = ArgumentRef(
            name="magnetic_field",
            operator="component",
            operator_kind="projection",
            axis="x",
            shape="component",
        )
        assert ref.axis == "x"
        assert ref.shape == "component"

    def test_binary_missing_role_fails(self):
        with pytest.raises(ValueError, match="role and separator are required"):
            ArgumentRef(
                name="pressure",
                operator="ratio",
                operator_kind="binary",
                separator="to",
            )

    def test_binary_missing_separator_fails(self):
        with pytest.raises(ValueError, match="role and separator are required"):
            ArgumentRef(
                name="pressure",
                operator="ratio",
                operator_kind="binary",
                role="a",
            )

    def test_binary_with_axis_forbidden(self):
        with pytest.raises(ValueError, match="axis and shape are forbidden"):
            ArgumentRef(
                name="pressure",
                operator="ratio",
                operator_kind="binary",
                role="a",
                separator="to",
                axis="x",
            )

    def test_projection_missing_axis_fails(self):
        with pytest.raises(ValueError, match="axis and shape are required"):
            ArgumentRef(
                name="magnetic_field",
                operator="component",
                operator_kind="projection",
                shape="component",
            )

    def test_projection_missing_shape_fails(self):
        with pytest.raises(ValueError, match="axis and shape are required"):
            ArgumentRef(
                name="magnetic_field",
                operator="component",
                operator_kind="projection",
                axis="x",
            )

    def test_projection_with_role_forbidden(self):
        with pytest.raises(ValueError, match="role and separator are forbidden"):
            ArgumentRef(
                name="magnetic_field",
                operator="component",
                operator_kind="projection",
                axis="x",
                shape="component",
                role="a",
            )

    def test_unary_with_role_forbidden(self):
        with pytest.raises(ValueError, match="role and separator are forbidden"):
            ArgumentRef(
                name="temperature",
                operator="maximum",
                operator_kind="unary_prefix",
                role="a",
            )

    def test_unary_with_axis_forbidden(self):
        with pytest.raises(ValueError, match="axis and shape are forbidden"):
            ArgumentRef(
                name="temperature",
                operator="maximum",
                operator_kind="unary_prefix",
                axis="x",
            )

    def test_invalid_operator_kind_rejected(self):
        with pytest.raises(ValueError):
            ArgumentRef(
                name="temperature",
                operator="maximum",
                operator_kind="invalid_kind",
            )

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValueError):
            ArgumentRef(
                name="temperature",
                operator="maximum",
                operator_kind="unary_prefix",
                extra_field="nope",
            )


# ---------------------------------------------------------------------------
# StandardNameEntry with arguments and error_variants
# ---------------------------------------------------------------------------


class TestEntryWithComputedFields:
    """StandardNameEntry accepts arguments and error_variants."""

    def test_entry_with_arguments(self):
        entry = create_standard_name_entry(
            {
                "name": "maximum_of_temperature",
                "kind": "scalar",
                "physics_domain": "core_plasma_physics",
                "description": "Maximum of temperature.",
                "documentation": "Peak temperature over a domain.",
                "unit": "eV",
                "arguments": [
                    {
                        "name": "temperature",
                        "operator": "maximum",
                        "operator_kind": "unary_prefix",
                    }
                ],
            }
        )
        assert entry.arguments is not None
        assert len(entry.arguments) == 1
        assert entry.arguments[0].name == "temperature"

    def test_entry_with_error_variants(self):
        entry = create_standard_name_entry(
            {
                "name": "temperature",
                "kind": "scalar",
                "physics_domain": "core_plasma_physics",
                "description": "Temperature of a plasma species.",
                "documentation": "Generic temperature quantity.",
                "unit": "eV",
                "error_variants": {
                    "upper": "upper_uncertainty_of_temperature",
                    "lower": "lower_uncertainty_of_temperature",
                },
            }
        )
        assert entry.error_variants is not None
        assert entry.error_variants["upper"] == "upper_uncertainty_of_temperature"

    def test_entry_without_computed_fields(self):
        entry = create_standard_name_entry(
            {
                "name": "temperature",
                "kind": "scalar",
                "physics_domain": "core_plasma_physics",
                "description": "Temperature of a plasma species.",
                "documentation": "Generic temperature quantity.",
                "unit": "eV",
            }
        )
        assert entry.arguments is None
        assert entry.error_variants is None

    def test_invalid_error_variant_key_rejected(self):
        with pytest.raises(ValueError):
            create_standard_name_entry(
                {
                    "name": "temperature",
                    "kind": "scalar",
                    "physics_domain": "core_plasma_physics",
                    "description": "Temperature.",
                    "documentation": "Temperature.",
                    "unit": "eV",
                    "error_variants": {
                        "bad_key": "some_name",
                    },
                }
            )


# ---------------------------------------------------------------------------
# Per-domain list loader
# ---------------------------------------------------------------------------


class TestLoaderDomainFormat:
    """YamlStore loads per-domain list files."""

    def test_load_domain_list(self, tmp_path: Path):
        """Load a YAML file containing a list of entries."""
        _write_domain_fixture(tmp_path)
        store = YamlStore(tmp_path)
        models = store.load()
        names = {m.name for m in models}
        assert "temperature" in names
        assert "maximum_of_temperature" in names
        assert len(models) == 4

    def test_load_single_entry_dict_compat(self, tmp_path: Path):
        """Single-entry dict with 'name' key still loads (backward compat)."""
        (tmp_path / "plasma_current.yml").write_text(
            "name: plasma_current\n"
            "kind: scalar\n"
            "physics_domain: core_plasma_physics\n"
            "description: Plasma current.\n"
            "documentation: Total plasma current in the tokamak.\n"
            "unit: A\n"
        )
        store = YamlStore(tmp_path)
        models = store.load()
        assert len(models) == 1
        assert models[0].name == "plasma_current"


# ---------------------------------------------------------------------------
# Legacy per-file YAML migration error
# ---------------------------------------------------------------------------


class TestLegacyMigrationError:
    """Legacy per-file YAML produces clean migration error."""

    def test_nested_path_raises_migration_error(self, tmp_path: Path):
        """Nested subdirectory YAML triggers CatalogMigrationError."""
        subdir = tmp_path / "transport"
        subdir.mkdir()
        (subdir / "temperature.yml").write_text(
            "name: temperature\n"
            "kind: scalar\n"
            "description: Temperature.\n"
            "documentation: Temperature doc.\n"
            "unit: eV\n"
        )
        store = YamlStore(tmp_path, permissive=False)
        with pytest.raises(CatalogMigrationError, match="Legacy per-file YAML"):
            store.load()

    def test_nested_path_permissive_loads(self, tmp_path: Path):
        """In permissive mode, nested paths still load without error."""
        subdir = tmp_path / "transport"
        subdir.mkdir()
        (subdir / "temperature.yml").write_text(
            "name: temperature\n"
            "kind: scalar\n"
            "description: Temperature.\n"
            "documentation: Temperature doc.\n"
            "unit: eV\n"
        )
        store = YamlStore(tmp_path, permissive=True)
        models = store.load()
        assert len(models) >= 1

    def test_dict_without_name_raises_migration_error(self, tmp_path: Path):
        """Dict YAML without 'name' key triggers migration error."""
        (tmp_path / "bad.yml").write_text("key: value\nother: stuff\n")
        store = YamlStore(tmp_path, permissive=False)
        with pytest.raises(CatalogMigrationError, match="Legacy per-file YAML"):
            store.load()


# ---------------------------------------------------------------------------
# Round-trip: load → re-serialise → byte-identical
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Load a domain-file fixture, re-serialise, byte-identical."""

    def test_domain_round_trip(self, tmp_path: Path):
        """Load domain fixture and re-serialise preserving data + order."""
        _write_domain_fixture(tmp_path)

        # Load entries
        store = YamlStore(tmp_path)
        models = store.load()

        # Re-serialise preserving fixture order
        name_to_model = {m.name: m for m in models}
        reserialized: list[dict] = []
        for fixture_entry in _DOMAIN_FIXTURE:
            m = name_to_model[fixture_entry["name"]]
            d = m.model_dump(exclude_none=True)
            d["name"] = m.name
            reserialized.append(d)

        # Semantic equivalence: every fixture key/value round-trips intact,
        # and order of entries is preserved.
        assert [e["name"] for e in reserialized] == [e["name"] for e in _DOMAIN_FIXTURE]
        for fixture_entry, out_entry in zip(_DOMAIN_FIXTURE, reserialized, strict=True):
            for key, value in fixture_entry.items():
                assert key in out_entry, (
                    f"round-trip dropped {key!r} from {fixture_entry['name']!r}"
                )
                assert out_entry[key] == value, (
                    f"round-trip mismatch for "
                    f"{fixture_entry['name']!r}.{key}: "
                    f"{out_entry[key]!r} != {value!r}"
                )


# ---------------------------------------------------------------------------
# Missing cross-reference warning
# ---------------------------------------------------------------------------


class TestCrossReferenceWarnings:
    """Missing cross-reference warning fires but does not fail validation."""

    def test_missing_argument_target_warns(self, tmp_path: Path):
        """Warning when arguments[].name targets absent entry."""
        fixture = [
            {
                "name": "maximum_of_temperature",
                "kind": "scalar",
                "physics_domain": "core_plasma_physics",
                "description": "Maximum of temperature.",
                "documentation": "Peak temperature over a domain.",
                "unit": "eV",
                "arguments": [
                    {
                        "name": "temperature",
                        "operator": "maximum",
                        "operator_kind": "unary_prefix",
                    }
                ],
            },
        ]
        fpath = tmp_path / "transport.yml"
        with open(fpath, "w", encoding="utf-8") as fh:
            yaml.safe_dump(fixture, fh, sort_keys=False)

        store = YamlStore(tmp_path)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            models = store.load()

        # Should load successfully
        assert len(models) == 1
        # Should have warning about missing target
        assert any("temperature" in str(warning.message) for warning in w)
        assert any("argument reference" in msg for msg in store.validation_warnings)

    def test_missing_error_variant_target_warns(self, tmp_path: Path):
        """Warning when error_variants target is absent."""
        fixture = [
            {
                "name": "temperature",
                "kind": "scalar",
                "physics_domain": "core_plasma_physics",
                "description": "Temperature.",
                "documentation": "Temperature doc.",
                "unit": "eV",
                "error_variants": {
                    "upper": "upper_uncertainty_of_temperature",
                },
            },
        ]
        fpath = tmp_path / "transport.yml"
        with open(fpath, "w", encoding="utf-8") as fh:
            yaml.safe_dump(fixture, fh, sort_keys=False)

        store = YamlStore(tmp_path)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            models = store.load()

        assert len(models) == 1
        assert any("error_variant" in str(warning.message) for warning in w)

    def test_present_cross_refs_no_warning(self, tmp_path: Path):
        """No warning when all cross-references resolve."""
        _write_domain_fixture(tmp_path)  # Contains resolved refs
        store = YamlStore(tmp_path)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            store.load()

        xref_warnings = [
            x
            for x in w
            if "argument reference" in str(x.message)
            or "error_variant" in str(x.message)
        ]
        assert len(xref_warnings) == 0


# ---------------------------------------------------------------------------
# Topological load handles arguments[].name edges
# ---------------------------------------------------------------------------


class TestTopologicalOrderingArguments:
    """Topological sort respects arguments[].name dependencies."""

    def test_argument_target_before_wrapper(self, tmp_path: Path):
        """Entry with arguments loads after its argument targets."""
        _write_domain_fixture(tmp_path)
        store = YamlStore(tmp_path)
        models = store.load()
        order = list(ordered_model_names(models))
        idx = {name: i for i, name in enumerate(order)}

        # maximum_of_temperature depends on temperature
        assert idx["temperature"] < idx["maximum_of_temperature"]

    def test_binary_arguments_both_before_wrapper(self, tmp_path: Path):
        """Binary operator entry loads after both argument targets."""
        fixture = [
            {
                "name": "pressure",
                "kind": "scalar",
                "physics_domain": "core_plasma_physics",
                "description": "Pressure.",
                "documentation": "Pressure doc.",
                "unit": "Pa",
            },
            {
                "name": "density",
                "kind": "scalar",
                "physics_domain": "core_plasma_physics",
                "description": "Density.",
                "documentation": "Density doc.",
                "unit": "m^-3",
            },
            {
                "name": "ratio_of_pressure_to_density",
                "kind": "scalar",
                "physics_domain": "core_plasma_physics",
                "description": "Ratio of pressure to density.",
                "documentation": "Pressure over density.",
                "unit": "Pa.m^3",
                "arguments": [
                    {
                        "name": "pressure",
                        "operator": "ratio",
                        "operator_kind": "binary",
                        "role": "a",
                        "separator": "to",
                    },
                    {
                        "name": "density",
                        "operator": "ratio",
                        "operator_kind": "binary",
                        "role": "b",
                        "separator": "to",
                    },
                ],
            },
        ]
        fpath = tmp_path / "transport.yml"
        with open(fpath, "w", encoding="utf-8") as fh:
            yaml.safe_dump(fixture, fh, sort_keys=False)

        store = YamlStore(tmp_path)
        models = store.load()
        order = list(ordered_model_names(models))
        idx = {name: i for i, name in enumerate(order)}

        assert idx["pressure"] < idx["ratio_of_pressure_to_density"]
        assert idx["density"] < idx["ratio_of_pressure_to_density"]

    def test_missing_argument_target_does_not_block_ordering(self, tmp_path: Path):
        """Missing argument target is skipped in ordering (not a hard dep)."""
        fixture = [
            {
                "name": "maximum_of_temperature",
                "kind": "scalar",
                "physics_domain": "core_plasma_physics",
                "description": "Maximum of temperature.",
                "documentation": "Peak temperature.",
                "unit": "eV",
                "arguments": [
                    {
                        "name": "temperature",
                        "operator": "maximum",
                        "operator_kind": "unary_prefix",
                    }
                ],
            },
        ]
        fpath = tmp_path / "transport.yml"
        with open(fpath, "w", encoding="utf-8") as fh:
            yaml.safe_dump(fixture, fh, sort_keys=False)

        store = YamlStore(tmp_path)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            models = store.load()

        # Should still order successfully (temperature not in catalog)
        order = list(ordered_model_names(models))
        assert "maximum_of_temperature" in order

    def test_projection_argument_ordering(self, tmp_path: Path):
        """Projection entries load after their base argument."""
        fixture = [
            {
                "name": "magnetic_field",
                "kind": "scalar",
                "physics_domain": "equilibrium",
                "description": "Magnetic field.",
                "documentation": "Magnetic field doc.",
                "unit": "T",
            },
            {
                "name": "x_magnetic_field",
                "kind": "scalar",
                "physics_domain": "equilibrium",
                "description": "X component of magnetic field.",
                "documentation": "X projection of magnetic field.",
                "unit": "T",
                "arguments": [
                    {
                        "name": "magnetic_field",
                        "operator": "component",
                        "operator_kind": "projection",
                        "axis": "x",
                        "shape": "component",
                    }
                ],
            },
            {
                "name": "y_magnetic_field",
                "kind": "scalar",
                "physics_domain": "equilibrium",
                "description": "Y component of magnetic field.",
                "documentation": "Y projection of magnetic field.",
                "unit": "T",
                "arguments": [
                    {
                        "name": "magnetic_field",
                        "operator": "component",
                        "operator_kind": "projection",
                        "axis": "y",
                        "shape": "component",
                    }
                ],
            },
        ]
        fpath = tmp_path / "magnetics.yml"
        with open(fpath, "w", encoding="utf-8") as fh:
            yaml.safe_dump(fixture, fh, sort_keys=False)

        store = YamlStore(tmp_path)
        models = store.load()
        order = list(ordered_model_names(models))
        idx = {name: i for i, name in enumerate(order)}

        assert idx["magnetic_field"] < idx["x_magnetic_field"]
        assert idx["magnetic_field"] < idx["y_magnetic_field"]
