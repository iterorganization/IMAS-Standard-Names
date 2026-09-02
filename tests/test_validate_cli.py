import time
from pathlib import Path

from click.testing import CliRunner

from imas_standard_names.cli import standard_names
from imas_standard_names.validation.cli import validate_catalog_cli


def _seed(root: Path):
    (root / "a.yml").write_text(
        "name: a\nkind: scalar\nphysics_domain: general\nstatus: active\nunit: keV\ndescription: A desc.\ndocumentation: |\n  A description for CLI validation testing.\n"
    )
    (root / "b.yml").write_text(
        "name: b\nkind: scalar\nphysics_domain: general\nstatus: draft\nunit: keV\ndescription: B desc.\ndocumentation: |\n  B description for CLI validation testing.\n"
    )


def test_validate_memory_mode(tmp_path: Path):
    _seed(tmp_path)
    runner = CliRunner()
    res = runner.invoke(validate_catalog_cli, [str(tmp_path), "--mode", "memory"])
    assert res.exit_code == 0, res.output
    assert "PASSED" in res.output


def test_validate_file_mode_without_verify(tmp_path: Path):
    _seed(tmp_path)
    runner = CliRunner()
    build_res = runner.invoke(standard_names, ["build", str(tmp_path)])
    assert build_res.exit_code == 0
    res = runner.invoke(validate_catalog_cli, [str(tmp_path), "--mode", "file"])
    assert res.exit_code == 0, res.output
    assert "PASSED" in res.output


def test_validate_file_mode_with_verify_clean(tmp_path: Path):
    _seed(tmp_path)
    runner = CliRunner()
    runner.invoke(standard_names, ["build", str(tmp_path)])
    res = runner.invoke(
        validate_catalog_cli, [str(tmp_path), "--mode", "file", "--verify"]
    )
    assert res.exit_code == 0, res.output
    assert "PASSED" in res.output


def test_validate_file_mode_with_integrity_mismatch(tmp_path: Path):
    _seed(tmp_path)
    runner = CliRunner()
    runner.invoke(standard_names, ["build", str(tmp_path)])
    # Modify one file after build
    time.sleep(0.02)
    (tmp_path / "b.yml").write_text(
        "name: b\nkind: scalar\nphysics_domain: general\nstatus: draft\nunit: keV\ndescription: B changed.\ndocumentation: |\n  B description changed for integrity mismatch testing.\n"
    )
    res = runner.invoke(
        validate_catalog_cli,
        [str(tmp_path), "--mode", "file", "--verify"],
    )
    # Integrity mismatch returns exit code 2
    assert res.exit_code == 2, res.output
    assert "Integrity issues" in res.output


def test_validate_auto_prefers_file(tmp_path: Path):
    _seed(tmp_path)
    runner = CliRunner()
    runner.invoke(standard_names, ["build", str(tmp_path)])
    res = runner.invoke(validate_catalog_cli, [str(tmp_path), "--mode", "auto"])
    assert res.exit_code == 0
    assert "PASSED" in res.output
