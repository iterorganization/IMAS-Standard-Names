import importlib.resources as resources
from pathlib import Path

import yaml

from imas_standard_names.models import create_standard_name_entry
from imas_standard_names.yaml_store import (
    YamlStore,
    dump_catalog_yaml,
    write_catalog_yaml,
)


def _entry_with_source(source: dict[str, str]) -> dict:
    return {
        "name": "plasma_current",
        "kind": "scalar",
        "physics_domain": "core_plasma_physics",
        "description": "Plasma current.",
        "documentation": "Total plasma current in the tokamak.",
        "unit": "A",
        "sources": [source],
    }


def test_yaml_store_load(tmp_path: Path):
    store = YamlStore(tmp_path)
    # Write a YAML file directly to test load
    (tmp_path / "plasma_current.yml").write_text(
        "name: plasma_current\n"
        "kind: scalar\n"
        "physics_domain: core_plasma_physics\n"
        "description: Plasma current.\n"
        "documentation: Total plasma current in the tokamak.\n"
        "unit: A\n"
        ""
    )
    loaded = {mm.name: mm for mm in store.load()}
    assert "plasma_current" in loaded


def test_dd_source_binding_round_trips_in_generic_shape() -> None:
    entry = create_standard_name_entry(
        _entry_with_source(
            {
                "kind": "imas-dd",
                "ref": "summary/global_quantities/ip/value",
                "version": "4.0.0",
            }
        )
    )

    dumped = entry.model_dump(mode="json", exclude_none=True)["sources"]

    assert dumped == [
        {
            "kind": "imas-dd",
            "ref": "summary/global_quantities/ip/value",
            "version": "4.0.0",
        }
    ]


def test_facility_source_binding_round_trips_without_mapping_facet() -> None:
    entry = create_standard_name_entry(
        _entry_with_source(
            {
                "kind": "west-uda",
                "ref": "MAI/PLASMA/IP",
                "version": "62253",
                "semantic_facet": "measured",
            }
        )
    )

    dumped = entry.model_dump(mode="json", exclude_none=True)["sources"]

    assert dumped == [{"kind": "west-uda", "ref": "MAI/PLASMA/IP", "version": "62253"}]


def test_legacy_dd_source_binding_loads_as_generic_shape() -> None:
    entry = create_standard_name_entry(
        _entry_with_source(
            {
                "dd_path": "summary/global_quantities/ip/value",
                "dd_version": "4.0.0",
            }
        )
    )

    dumped = entry.model_dump(mode="json", exclude_none=True)["sources"]

    assert dumped == [
        {
            "kind": "imas-dd",
            "ref": "summary/global_quantities/ip/value",
            "version": "4.0.0",
        }
    ]


_LONG_PARAGRAPH = " ".join(["Poloidal flux through a reviewed magnetic surface."] * 7)


def test_catalog_yaml_wraps_prose_and_preserves_authored_structure() -> None:
    equation = "\\psi = " + " + ".join(f"B_{{{index}}}" for index in range(20))
    bullet = "- " + " ".join(["Keep this review item intact."] * 4)
    entries = [
        {
            "name": "poloidal_flux",
            "description": _LONG_PARAGRAPH,
            "documentation": (
                f"{_LONG_PARAGRAPH}\n\n$$\n{equation}\n$$\n\n{bullet}\n"
                "- Second item\n\nPositive \u03c8 follows the chosen orientation."
            ),
        },
        {
            "name": "safety_factor",
            "description": "Safety factor q for a magnetic surface.",
            "documentation": (
                "The winding ratio is dimensionless.\n\n"
                "$$q = \\frac{d\\Phi}{d\\psi}$$\n\n"
                "The toroidal flux is \u03a6."
            ),
        },
    ]

    rendered = dump_catalog_yaml(entries)
    lines = rendered.splitlines()

    assert len(_LONG_PARAGRAPH) > 300
    assert rendered.count("description: >-") == 2
    assert rendered.count("documentation: >-") == 2
    assert rendered.count("\n\n- name:") == 1
    assert "\n\n\n- name:" not in rendered
    # The paragraph is wrapped rather than emitted as one unreadable line.
    assert sum(1 for line in lines if line.strip().startswith("Poloidal flux")) > 1
    # Equations, fences and list items keep their authored layout.
    assert f"    {equation}" in lines
    assert f"    {bullet}" in lines
    assert "    $$" in lines
    assert "    $$q = \\frac{d\\Phi}{d\\psi}$$" in lines
    assert "\u03c8" in rendered
    assert "\u03a6" in rendered
    assert "\\u03c8" not in rendered.lower()
    assert "\\u03a6" not in rendered.lower()
    assert yaml.safe_load(rendered) == entries


def test_wrapped_prose_lines_stay_within_the_review_width() -> None:
    entries = [
        {
            "name": "poloidal_flux",
            "description": _LONG_PARAGRAPH,
            "documentation": _LONG_PARAGRAPH,
        }
    ]

    rendered = dump_catalog_yaml(entries)

    assert len(_LONG_PARAGRAPH) > 300
    assert max(len(line) for line in rendered.splitlines()) <= 88
    assert yaml.safe_load(rendered) == entries


def test_wrapped_prose_reloads_unchanged_through_the_store(tmp_path: Path) -> None:
    entry = {
        "name": "plasma_current",
        "kind": "scalar",
        "physics_domain": "core_plasma_physics",
        "unit": "A",
        "description": _LONG_PARAGRAPH,
        "documentation": (
            f"{_LONG_PARAGRAPH}\n\n$$\nI_p = \\int_S J \\cdot dS\n$$\n\n"
            f"{_LONG_PARAGRAPH}"
        ),
    }

    write_catalog_yaml(tmp_path / "core_plasma_physics.yml", [entry])
    loaded = {model.name: model for model in YamlStore(tmp_path).load()}

    assert loaded["plasma_current"].description == entry["description"]
    assert loaded["plasma_current"].documentation == entry["documentation"]


def test_catalog_yaml_round_trips_input_structure() -> None:
    entries = [
        {
            "name": "safety_factor",
            "description": "Safety factor q at normalized poloidal flux ψ.",
            "documentation": "The angle φ is measured in radians.",
            "status": "draft",
            "kind": "scalar",
            "physics_domain": "equilibrium",
            "unit": "1",
            "links": ["magnetic_field", "poloidal_flux"],
        },
        {
            "name": "magnetic_axis",
            "description": "Magnetic axis position.",
            "status": "active",
            "kind": "metadata",
            "physics_domain": "equilibrium",
        },
    ]

    assert yaml.safe_load(dump_catalog_yaml(entries)) == entries


def test_existing_catalog_rewrite_preserves_data(tmp_path: Path) -> None:
    source = (
        resources.files("imas_standard_names")
        / "resources"
        / "standard_name_examples"
        / "equilibrium.yml"
    )
    existing = yaml.safe_load(source.read_text(encoding="utf-8"))
    rewritten = tmp_path / "equilibrium.yml"

    write_catalog_yaml(rewritten, existing)

    assert yaml.safe_load(rewritten.read_text(encoding="utf-8")) == existing
