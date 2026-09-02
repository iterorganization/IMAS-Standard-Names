import json
from pathlib import Path

from click.testing import CliRunner

from imas_standard_names.cli import standard_names
from imas_standard_names.validation.cli import validate_catalog_cli


def _make_minimal_vector(tmp_path: Path):  # now simplified to scalars
    (tmp_path / "electron_temperature.yml").write_text(
        """name: electron_temperature
kind: scalar
physics_domain: core_plasma_physics
status: active
unit: eV
description: Electron temperature.
documentation: |
  Electron temperature for CLI command testing.
""",
        encoding="utf-8",
    )
    (tmp_path / "ion_temperature.yml").write_text(
        """name: ion_temperature
kind: scalar
physics_domain: core_plasma_physics
status: active
unit: eV
description: Ion temperature.
documentation: |
  Ion temperature for CLI command testing.
""",
        encoding="utf-8",
    )


def test_catalog_build_subcommand(tmp_path: Path):
    _make_minimal_vector(tmp_path)
    runner = CliRunner()
    result = runner.invoke(standard_names, ["build", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "Built catalog" in result.output


def test_catalog_build_with_verify(tmp_path: Path):
    """Test build --verify outputs file size and entry count."""
    _make_minimal_vector(tmp_path)
    runner = CliRunner()
    result = runner.invoke(standard_names, ["build", str(tmp_path), "--verify"])
    assert result.exit_code == 0, result.output
    assert "✓ Built" in result.output
    assert "2 entries" in result.output
    # Check for size format (KB or B)
    assert "KB" in result.output or "B" in result.output


def test_validate_catalog_cli(tmp_path: Path):
    _make_minimal_vector(tmp_path)
    runner = CliRunner()
    result = runner.invoke(validate_catalog_cli, [str(tmp_path)])
    assert result.exit_code == 0, result.output


def test_validate_catalog_summary_text(tmp_path: Path):
    """Test validate_catalog --summary text outputs machine-readable summary."""
    _make_minimal_vector(tmp_path)
    runner = CliRunner()
    result = runner.invoke(validate_catalog_cli, [str(tmp_path), "--summary", "text"])
    assert result.exit_code == 0, result.output
    assert "✓ Validated" in result.output
    assert "2 entries" in result.output
    assert "0 errors" in result.output


def test_validate_catalog_summary_json(tmp_path: Path):
    """Test validate_catalog --summary json outputs JSON format."""
    _make_minimal_vector(tmp_path)
    runner = CliRunner()
    result = runner.invoke(validate_catalog_cli, [str(tmp_path), "--summary", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["passed"] is True
    assert data["entries"] == 2
    assert data["errors"] == 0


def test_validate_catalog_no_quality_check(tmp_path: Path):
    """Test validate_catalog --no-quality-check skips quality checks."""
    _make_minimal_vector(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        validate_catalog_cli, [str(tmp_path), "--no-quality-check", "--summary", "json"]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["passed"] is True
    # Quality checks disabled, so no info/warnings counted
    assert data["info"] == 0


def test_validate_catalog_quality_check_default(tmp_path: Path):
    """Test that quality checks are enabled by default."""
    _make_minimal_vector(tmp_path)
    runner = CliRunner()
    # Run without --no-quality-check, quality checks should run
    result = runner.invoke(validate_catalog_cli, [str(tmp_path), "--summary", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["passed"] is True


def test_validate_catalog_strict_fails_on_warnings(tmp_path: Path):
    """Test that --strict fails validation when warnings are present."""
    # Create an entry with a description that triggers warnings (too short)
    (tmp_path / "short_desc.yml").write_text(
        """name: short_desc
kind: scalar
physics_domain: general
status: active
unit: '1'
description: Short.
documentation: |
  Entry with intentionally short description to trigger warning.
""",
        encoding="utf-8",
    )
    runner = CliRunner()
    # Without --strict, should pass despite warnings
    result = runner.invoke(validate_catalog_cli, [str(tmp_path), "--summary", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["warnings"] > 0, "Expected warnings from short description"
    assert data["passed"] is True

    # With --strict, should fail due to warnings
    result = runner.invoke(
        validate_catalog_cli, [str(tmp_path), "--strict", "--summary", "json"]
    )
    assert result.exit_code == 1, result.output
    data = json.loads(result.output)
    assert data["passed"] is False


def test_serve_command_exists():
    """Test that the top-level serve command is registered."""
    runner = CliRunner()
    result = runner.invoke(standard_names, ["serve", "--help"])
    assert result.exit_code == 0, result.output
    assert "preview" in result.output.lower()


def test_site_deploy_command_help():
    """Test site-deploy command has expected options."""
    runner = CliRunner()
    result = runner.invoke(standard_names, ["site-deploy", "--help"])
    assert result.exit_code == 0, result.output
    assert "--version" in result.output
    assert "--push" in result.output
    assert "--set-default" in result.output
