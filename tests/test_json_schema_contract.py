"""Tests for StandardNameEntry JSON schema generation and validation."""

import json
from pathlib import Path

from imas_standard_names.schemas.generate import (
    _SCHEMA_PATH,
    generate_entry_schema,
    write_entry_schema,
)
from imas_standard_names.schemas.validate import validate_against_schema

# ---------------------------------------------------------------------------
# Schema generation
# ---------------------------------------------------------------------------


def test_generate_entry_schema_returns_dict():
    """generate_entry_schema() produces a JSON-serializable dictionary."""
    schema = generate_entry_schema()
    assert isinstance(schema, dict)
    # Must be JSON-serializable
    json.dumps(schema)


def test_schema_includes_all_three_variants():
    """Schema covers scalar, vector, and metadata entry kinds."""
    schema = generate_entry_schema()
    schema_text = json.dumps(schema)
    assert "StandardNameScalarEntry" in schema_text
    assert "StandardNameVectorEntry" in schema_text
    assert "StandardNameMetadataEntry" in schema_text


def test_schema_excludes_package_version_metadata():
    """Schema generation is independent of the local package version."""
    schema = generate_entry_schema()
    assert "$schema_version" not in schema


def test_schema_has_defs():
    """Schema includes $defs with model definitions."""
    schema = generate_entry_schema()
    assert "$defs" in schema
    assert len(schema["$defs"]) > 0


# ---------------------------------------------------------------------------
# Validation utility
# ---------------------------------------------------------------------------


def test_validate_accepts_valid_scalar_entry():
    """Valid scalar entry data passes validation."""
    data = {
        "name": "electron_temperature",
        "kind": "scalar",
        "physics_domain": "core_plasma_physics",
        "unit": "eV",
        "description": "Electron temperature.",
        "documentation": "",
    }
    errors = validate_against_schema(data)
    assert errors == []


def test_validate_accepts_valid_vector_entry():
    """Valid vector entry data passes validation."""
    data = {
        "name": "plasma_velocity",
        "kind": "vector",
        "physics_domain": "core_plasma_physics",
        "unit": "m.s^-1",
        "description": "Plasma velocity vector.",
        "documentation": "",
    }
    errors = validate_against_schema(data)
    assert errors == []


def test_validate_accepts_valid_metadata_entry():
    """Valid metadata entry data passes validation."""
    data = {
        "name": "plasma_boundary",
        "kind": "metadata",
        "physics_domain": "equilibrium",
        "description": "Definition of the plasma boundary.",
        "documentation": "",
    }
    errors = validate_against_schema(data)
    assert errors == []


def test_validate_rejects_missing_kind():
    """Missing 'kind' field produces validation errors."""
    data = {"name": "bad_entry"}
    errors = validate_against_schema(data)
    assert len(errors) > 0


def test_validate_rejects_invalid_kind():
    """An unrecognized 'kind' value produces validation errors."""
    data = {
        "name": "bad_entry",
        "kind": "imaginary",
        "physics_domain": "general",
        "description": "Not real.",
        "documentation": "",
    }
    errors = validate_against_schema(data)
    assert len(errors) > 0


def test_validate_rejects_missing_required_fields():
    """Missing required fields (description, name) produce errors."""
    data = {"kind": "scalar"}
    errors = validate_against_schema(data)
    assert len(errors) > 0


def test_validate_errors_are_readable_strings():
    """Validation errors are human-readable strings."""
    data = {"kind": "scalar"}
    errors = validate_against_schema(data)
    for err in errors:
        assert isinstance(err, str)
        assert len(err) > 0


# ---------------------------------------------------------------------------
# Schema file on disk
# ---------------------------------------------------------------------------


def test_schema_file_exists():
    """The committed entry_schema.json file exists."""
    assert _SCHEMA_PATH.is_file(), f"Schema file missing at {_SCHEMA_PATH}"


def test_schema_file_is_valid_json():
    """The on-disk schema file is valid JSON."""
    content = _SCHEMA_PATH.read_text()
    schema = json.loads(content)
    assert isinstance(schema, dict)
    assert "$schema_version" not in schema


def test_schema_file_matches_generated():
    """The committed schema file matches a fresh generation.

    This catches drift between the committed artifact and the current
    model definitions.
    """
    committed = json.loads(_SCHEMA_PATH.read_text())
    fresh = generate_entry_schema()
    assert committed == fresh


# ---------------------------------------------------------------------------
# Schema file can be loaded via importlib.resources
# ---------------------------------------------------------------------------


def test_schema_loadable_via_importlib():
    """Schema file is discoverable via importlib.resources."""
    import importlib.resources

    package = importlib.resources.files("imas_standard_names.schemas")
    schema_file = package / "entry_schema.json"
    content = schema_file.read_text()
    schema = json.loads(content)
    assert "$schema_version" not in schema


# ---------------------------------------------------------------------------
# Write utility
# ---------------------------------------------------------------------------


def test_write_entry_schema_to_custom_path(tmp_path):
    """write_entry_schema() can write to a specified path."""
    schema = generate_entry_schema()
    target = tmp_path / "entry_schema.json"
    written = write_entry_schema(target)
    assert written == target
    reloaded = json.loads(written.read_text())
    assert reloaded == schema


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def test_cli_main_generates_schema(tmp_path):
    """The CLI main() function generates the schema file."""
    from click.testing import CliRunner

    from imas_standard_names.schemas.generate import main

    runner = CliRunner()
    target = tmp_path / "entry_schema.json"
    result = runner.invoke(main, ["--output", str(target)])
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert "Schema written to" in result.output

    # The schema file should exist and be valid
    assert target.is_file()
    content = json.loads(target.read_text())
    assert content == generate_entry_schema()


def test_cli_main_default_output(monkeypatch, tmp_path):
    """CLI with no --output flag writes to an isolated default location."""
    from click.testing import CliRunner

    from imas_standard_names.schemas import generate
    from imas_standard_names.schemas.generate import main

    target = tmp_path / "entry_schema.json"
    monkeypatch.setattr(generate, "_SCHEMA_PATH", target)
    runner = CliRunner()
    result = runner.invoke(main, [])
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert "Schema written to" in result.output
    assert target.is_file()
