import json
from pathlib import Path

from click.testing import CliRunner

from imas_standard_names.cli import standard_names


def _seed(root: Path):
    (root / "a.yml").write_text(
        "name: electron_temperature\nkind: scalar\nphysics_domain: core_plasma_physics\nstatus: active\nunit: keV\ndescription: Electron temperature.\ndocumentation: |\n  Electron temperature for search mode testing.\n"
    )
    (root / "b.yml").write_text(
        "name: ion_temperature\nkind: scalar\nphysics_domain: core_plasma_physics\nstatus: draft\nunit: keV\ndescription: Ion temperature.\ndocumentation: |\n  Ion temperature for search mode testing.\n"
    )


def test_search_mode_memory(tmp_path: Path):
    _seed(tmp_path)
    runner = CliRunner()
    res = runner.invoke(
        standard_names,
        ["search", "electron", str(tmp_path), "--mode", "memory", "--meta"],
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert any(r["name"] == "electron_temperature" for r in payload)
    assert all(r["source"] == "memory" for r in payload)


def test_search_mode_file(tmp_path: Path):
    _seed(tmp_path)
    runner = CliRunner()
    # Build file-backed catalog
    build_res = runner.invoke(standard_names, ["build", str(tmp_path)])
    assert build_res.exit_code == 0, build_res.output
    res = runner.invoke(
        standard_names,
        ["search", "ion", str(tmp_path), "--mode", "file", "--meta"],
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert any(r["name"] == "ion_temperature" for r in payload)
    assert all(r["source"] == "file" for r in payload)


def test_search_mode_auto_prefers_file(tmp_path: Path):
    _seed(tmp_path)
    runner = CliRunner()
    # Build file-backed catalog
    runner.invoke(standard_names, ["build", str(tmp_path)])
    res = runner.invoke(
        standard_names,
        ["search", "temperature", str(tmp_path), "--mode", "auto", "--meta"],
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    # Should include both names and mark file as source
    names = {r["name"] for r in payload}
    assert {"electron_temperature", "ion_temperature"}.issubset(names)
    assert all(r["source"] == "file" for r in payload)
