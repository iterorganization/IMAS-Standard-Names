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


def test_catalog_yaml_uses_literal_blocks_with_authored_equation_spacing() -> None:
    entries = [
        {
            "name": "poloidal_flux",
            "description": "Poloidal flux ψ through a magnetic surface.",
            "documentation": (
                "The flux follows the surface convention.\n\n"
                "$$\\psi = \\int_S \\mathbf{B} \\cdot d\\mathbf{S}$$\n\n"
                "Positive ψ follows the chosen orientation."
            ),
        },
        {
            "name": "safety_factor",
            "description": "Safety factor q for a magnetic surface.",
            "documentation": (
                "The winding ratio is dimensionless.\n\n"
                "$$q = \\frac{d\\Phi}{d\\psi}$$\n\n"
                "The toroidal flux is Φ."
            ),
        },
    ]

    rendered = dump_catalog_yaml(entries)

    assert rendered.count("description: |-") == 2
    assert rendered.count("documentation: |-") == 2
    assert rendered.count("\n\n- name:") == 1
    assert "\n\n\n- name:" not in rendered
    assert (
        "    The flux follows the surface convention.\n\n"
        "    $$\\psi = \\int_S \\mathbf{B} \\cdot d\\mathbf{S}$$\n\n"
        "    Positive ψ follows the chosen orientation."
    ) in rendered
    assert (
        "    The winding ratio is dimensionless.\n\n"
        "    $$q = \\frac{d\\Phi}{d\\psi}$$\n\n"
        "    The toroidal flux is Φ."
    ) in rendered
    assert "ψ" in rendered
    assert "Φ" in rendered
    assert "\\u03c8" not in rendered.lower()
    assert "\\u03a6" not in rendered.lower()
    assert yaml.safe_load(rendered) == entries


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
