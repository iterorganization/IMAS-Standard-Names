import importlib.resources as resources
from pathlib import Path

import yaml

from imas_standard_names.models import create_standard_name_entry
from imas_standard_names.yaml_store import (
    YamlStore,
    dump_catalog_yaml,
    unwrap_catalog_prose,
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
_SECOND_PARAGRAPH = " ".join(
    [
        "The sign follows the orientation chosen for the poloidal circuit,",
        "see [magnetic_field](magnetic_field.md) for the companion convention",
        "and for the toroidal counterpart of this quantity.",
    ]
)
_DISPLAY_EQUATION = "$$\n\\psi = \\int_S \\mathbf{B} \\cdot d\\mathbf{S}\n$$"
_PROSE_FIELDS = ("description", "documentation")


def _reviewed_entries() -> list[dict]:
    """Entries whose prose carries two paragraphs, display math and a link."""
    return [
        {
            "name": "poloidal_flux",
            "kind": "scalar",
            "physics_domain": "equilibrium",
            "unit": "Wb",
            "description": _LONG_PARAGRAPH,
            "documentation": (
                f"{_LONG_PARAGRAPH}\n\n{_DISPLAY_EQUATION}\n\n{_SECOND_PARAGRAPH}"
            ),
        },
        {
            "name": "safety_factor",
            "kind": "scalar",
            "physics_domain": "equilibrium",
            "unit": "1",
            "description": "Safety factor q for a magnetic surface.",
            "documentation": f"The winding ratio is dimensionless.\n\n{_SECOND_PARAGRAPH}",
        },
    ]


def _rejoin_prose(entry: dict) -> dict:
    return {
        key: unwrap_catalog_prose(value) if key in _PROSE_FIELDS else value
        for key, value in entry.items()
    }


def test_literal_block_prose_reloads_unchanged_through_the_store(
    tmp_path: Path,
) -> None:
    entries = _reviewed_entries()

    write_catalog_yaml(tmp_path / "equilibrium.yml", entries)
    loaded = {model.name: model for model in YamlStore(tmp_path).load()}

    for entry in entries:
        model = loaded[entry["name"]]
        assert model.description == entry["description"]
        assert model.documentation == entry["documentation"]


def test_reemitting_the_loaded_entries_reproduces_the_same_bytes() -> None:
    entries = _reviewed_entries()

    rendered = dump_catalog_yaml(entries)
    reloaded = [_rejoin_prose(item) for item in yaml.safe_load(rendered)]

    assert reloaded == entries
    assert dump_catalog_yaml(reloaded) == rendered


def test_emitted_prose_is_wrapped_without_doubled_blank_lines() -> None:
    entries = _reviewed_entries()

    rendered = dump_catalog_yaml(entries)
    lines = rendered.splitlines()

    assert len(_LONG_PARAGRAPH) > 300
    assert max(len(line) for line in lines) <= 88
    assert "\n\n\n" not in rendered
    assert rendered.count("description: |-") == 2
    assert rendered.count("documentation: |-") == 2
    # One blank line separates the two entries and each pair of paragraphs.
    assert rendered.count("\n\n- name:") == 1
    # The paragraph is wrapped rather than emitted as one unreadable line.
    assert sum(1 for line in lines if line.strip().startswith("Poloidal flux")) > 1


def test_catalog_yaml_keeps_authored_structure_and_unicode() -> None:
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

    # Equations, fences and list items keep their authored layout.
    assert f"    {equation}" in lines
    assert f"    {bullet}" in lines
    assert "    $$" in lines
    assert "    $$q = \\frac{d\\Phi}{d\\psi}$$" in lines
    assert "\u03c8" in rendered
    assert "\u03a6" in rendered
    assert "\\u03c8" not in rendered.lower()
    assert "\\u03a6" not in rendered.lower()
    assert [_rejoin_prose(item) for item in yaml.safe_load(rendered)] == entries


def test_catalog_yaml_round_trips_input_structure() -> None:
    entries = [
        {
            "name": "safety_factor",
            "description": "Safety factor q at normalized poloidal flux \u03c8.",
            "documentation": "The angle \u03c6 is measured in radians.",
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

    rewritten_entries = [
        _rejoin_prose(item)
        for item in yaml.safe_load(rewritten.read_text(encoding="utf-8"))
    ]

    assert rewritten_entries == existing
