"""Tests for the required physics-domain entry field."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from imas_standard_names.catalog import build_site_dataset
from imas_standard_names.grammar.tag_types import PhysicsDomain
from imas_standard_names.models import create_standard_name_entry
from imas_standard_names.yaml_store import YamlStore, write_catalog_yaml


def _entry_data(**overrides: str) -> dict[str, str]:
    data = {
        "name": "electron_temperature",
        "kind": "scalar",
        "physics_domain": "core_plasma_physics",
        "description": "Electron temperature.",
        "documentation": "Kinetic temperature of the electron population.",
        "unit": "eV",
        "status": "active",
    }
    data.update(overrides)
    return data


def test_physics_domain_round_trips_through_model_and_yaml_store(
    tmp_path: Path,
) -> None:
    entry = create_standard_name_entry(_entry_data())
    assert entry.physics_domain is PhysicsDomain.CORE_PLASMA_PHYSICS

    write_catalog_yaml(
        tmp_path / "core_plasma_physics.yml",
        [entry.model_dump(mode="json", exclude_none=True)],
    )

    loaded = YamlStore(tmp_path).load()
    assert len(loaded) == 1
    assert loaded[0].physics_domain is PhysicsDomain.CORE_PLASMA_PHYSICS


def test_unknown_physics_domain_names_enum_in_validation_error() -> None:
    with pytest.raises(ValidationError, match="PhysicsDomain"):
        create_standard_name_entry(_entry_data(physics_domain="unknown_domain"))


def test_missing_physics_domain_is_rejected() -> None:
    data = _entry_data()
    del data["physics_domain"]

    with pytest.raises(ValidationError) as exc_info:
        create_standard_name_entry(data)

    error = exc_info.value.errors()[0]
    assert error["loc"][-1] == "physics_domain"
    assert error["type"] == "missing"


def test_catalog_domain_grouping_uses_declared_model_field(tmp_path: Path) -> None:
    write_catalog_yaml(tmp_path / "core_plasma_physics.yml", [_entry_data()])

    dataset = build_site_dataset(tmp_path)

    assert dataset["NAMES"][0]["category"] == "core_plasma_physics"
    assert dataset["CATEGORIES"] == [
        {"id": "core_plasma_physics", "label": "Core Plasma Physics", "count": 1}
    ]


def test_repository_catalog_fixtures_declare_physics_domain() -> None:
    example_root = Path("imas_standard_names/resources/standard_name_examples")
    fixture_paths = (
        sorted(Path("tests").rglob("*.yml"))
        + sorted(Path("tests").rglob("*.yaml"))
        + sorted(example_root.glob("*.yml"))
    )
    missing: list[str] = []
    mismatched: list[str] = []
    for path in fixture_paths:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
        entries = content if isinstance(content, list) else [content]
        for entry in entries:
            if not isinstance(entry, dict) or "name" not in entry:
                continue
            if "physics_domain" not in entry:
                missing.append(f"{path}:{entry['name']}")
            elif path.parent == example_root and entry["physics_domain"] != path.stem:
                mismatched.append(f"{path}:{entry['name']}")

    assert missing == []
    assert mismatched == []
