"""Tests for the catalog manifest's per-name metadata sidecar.

The reviewable catalog entry is reduced to the four fields a physicist
edits — ``name``, ``description``, ``documentation`` and ``unit``. Every
machine-owned field (``kind``, ``status``, ``physics_domain``, ``sources``
with their pinned versions, ``links`` and ``arguments``) moves into a
per-name block of the manifest sidebar at ``catalog.yml``. These tests
pin the three consequences: the reduced entry still validates and
resolves its kind through the entry union, the SPA dataset reads the
moved fields from the sidecar rather than from the entry, and the
manifest model declares the block while still forbidding stray keys.
"""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from imas_standard_names.catalog import build_site_dataset
from imas_standard_names.models import (
    CATALOG_EDGE_MODEL_VERSION,
    CatalogNameMetadata,
    StandardNameCatalogManifest,
    StandardNameEntryBase,
    StandardNameVectorEntry,
    create_standard_name_entry,
)

# The four fields review touches. Everything else is machine-owned.
EDITABLE_FIELDS = ("name", "description", "documentation", "unit")

_DOC = "Body documentation paragraph for the sidecar fixture entry."

# One reviewable entry: prose, a unit, and nothing a physicist should not
# be asked to maintain by hand.
REDUCED_ENTRY: dict = {
    "name": "radial_magnetic_field",
    "description": "Radial component of the magnetic flux density.",
    "documentation": _DOC,
    "unit": "T",
}

# Its machine-owned counterpart, held once in the manifest.
SIDECAR_BLOCK: dict = {
    "kind": "vector",
    "status": "active",
    "physics_domain": "equilibrium",
    "links": ["name:magnetic_field"],
    "sources": [
        {
            "kind": "imas-dd",
            "ref": "equilibrium/time_slice/profiles_2d/b_field_r",
            "version": "4.0.0",
        }
    ],
    "arguments": [
        {
            "name": "magnetic_field",
            "operator": "radial",
            "operator_kind": "projection",
            "axis": "radial",
            "shape": "component",
        }
    ],
}

BASE_MANIFEST: dict = {
    "catalog_name": "sidecar-fixture",
    "cocos_convention": 11,
    "grammar_version": "0.1.0",
    "isn_model_version": "0.1.0",
    "dd_version_lineage": ["4.0.0"],
    "generated_by": "test",
    "generated_at": "2026-09-03T00:00:00Z",
    "candidate_count": 1,
    "published_count": 1,
    "domains_included": ["equilibrium"],
}


def _manifest(names: dict | None = None) -> StandardNameCatalogManifest:
    """Validate the fixture manifest, optionally carrying a sidecar."""
    data = dict(BASE_MANIFEST)
    if names is not None:
        data["names"] = names
    return StandardNameCatalogManifest.model_validate(data)


@pytest.fixture
def catalog_dir(tmp_path: Path) -> Path:
    """Write a catalog whose only entry carries just the editable fields."""
    names_dir = tmp_path / "standard_names"
    names_dir.mkdir()
    (names_dir / "equilibrium.yml").write_text(
        yaml.safe_dump([REDUCED_ENTRY]), encoding="utf-8"
    )
    manifest = dict(BASE_MANIFEST)
    manifest["names"] = {REDUCED_ENTRY["name"]: SIDECAR_BLOCK}
    (tmp_path / "catalog.yml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return names_dir


class TestReducedEntryValidates:
    """A four-field entry plus its sidecar block resolves the entry union."""

    def test_entry_carries_only_the_editable_fields(self) -> None:
        assert tuple(REDUCED_ENTRY) == EDITABLE_FIELDS

    def test_reduced_entry_alone_cannot_discriminate(self) -> None:
        """Without the sidecar there is no ``kind`` for the union tag."""
        with pytest.raises(ValidationError):
            create_standard_name_entry(REDUCED_ENTRY)

    def test_sidecar_resolves_kind_and_validates(self) -> None:
        entry = create_standard_name_entry(
            REDUCED_ENTRY, manifest=_manifest({REDUCED_ENTRY["name"]: SIDECAR_BLOCK})
        )
        assert isinstance(entry, StandardNameVectorEntry)
        assert entry.kind == "vector"
        assert entry.name == "radial_magnetic_field"
        assert entry.unit == "T"

    def test_sidecar_supplies_every_moved_field(self) -> None:
        entry = create_standard_name_entry(
            REDUCED_ENTRY, manifest=_manifest({REDUCED_ENTRY["name"]: SIDECAR_BLOCK})
        )
        assert entry.status == "active"
        assert str(entry.physics_domain) == "equilibrium"
        assert entry.links == ["name:magnetic_field"]
        assert entry.sources is not None
        assert entry.sources[0].version == "4.0.0"
        assert entry.arguments is not None
        assert entry.arguments[0].name == "magnetic_field"

    def test_resolve_leaves_an_unlisted_name_untouched(self) -> None:
        manifest = _manifest({REDUCED_ENTRY["name"]: SIDECAR_BLOCK})
        other = {"name": "electron_temperature", "unit": "eV"}
        assert manifest.resolve_entry(other) == other

    def test_entry_is_unmutated_by_resolution(self) -> None:
        manifest = _manifest({REDUCED_ENTRY["name"]: SIDECAR_BLOCK})
        manifest.resolve_entry(REDUCED_ENTRY)
        assert tuple(REDUCED_ENTRY) == EDITABLE_FIELDS


class TestDatasetReadsSidecar:
    """The SPA dataset resolves the moved fields through the manifest."""

    @pytest.fixture
    def record(self, catalog_dir: Path) -> dict:
        dataset = build_site_dataset(catalog_dir)
        assert len(dataset["NAMES"]) == 1
        return dataset["NAMES"][0]

    def test_physics_domain_from_sidecar(self, record: dict) -> None:
        assert record["category"] == "equilibrium"

    def test_links_from_sidecar(self, record: dict) -> None:
        assert record["seeAlso"] == ["magnetic_field"]

    def test_sources_from_sidecar(self, record: dict) -> None:
        assert len(record["sources"]) == 1
        assert record["sources"][0]["version"] == "4.0.0"

    def test_arguments_parent_from_sidecar(self, record: dict) -> None:
        assert record["parent"] == "magnetic_field"
        assert record["arguments"] == ["magnetic_field"]

    def test_kind_and_status_from_sidecar(self, record: dict) -> None:
        assert record["algebra"] == "vector"
        assert record["status"] == "active"

    def test_sidecar_overrides_a_stale_inline_value(self, tmp_path: Path) -> None:
        """When both carry the field, the sidecar is authoritative."""
        names_dir = tmp_path / "standard_names"
        names_dir.mkdir()
        stale = dict(REDUCED_ENTRY) | {
            "kind": "scalar",
            "physics_domain": "core_plasma_physics",
        }
        (names_dir / "equilibrium.yml").write_text(
            yaml.safe_dump([stale]), encoding="utf-8"
        )
        manifest = dict(BASE_MANIFEST)
        manifest["names"] = {REDUCED_ENTRY["name"]: SIDECAR_BLOCK}
        (tmp_path / "catalog.yml").write_text(
            yaml.safe_dump(manifest), encoding="utf-8"
        )
        record = build_site_dataset(names_dir)["NAMES"][0]
        assert record["algebra"] == "vector"
        assert record["category"] == "equilibrium"


class TestManifestDeclaresTheBlock:
    """The block is declared; undeclared keys are still refused."""

    def test_block_is_accepted(self) -> None:
        manifest = _manifest({REDUCED_ENTRY["name"]: SIDECAR_BLOCK})
        block = manifest.names[REDUCED_ENTRY["name"]]
        assert isinstance(block, CatalogNameMetadata)
        assert block.kind == "vector"

    def test_names_defaults_to_empty(self) -> None:
        assert _manifest().names == {}

    def test_undeclared_manifest_key_is_forbidden(self) -> None:
        data = dict(BASE_MANIFEST) | {"per_name_metadata": {}}
        with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
            StandardNameCatalogManifest.model_validate(data)

    def test_undeclared_block_key_is_forbidden(self) -> None:
        bad = {REDUCED_ENTRY["name"]: SIDECAR_BLOCK | {"provenance_mode": "operator"}}
        with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
            _manifest(bad)

    def test_block_carries_no_editable_field(self) -> None:
        """The sidecar must never become a second home for the prose."""
        declared = set(CatalogNameMetadata.model_fields)
        assert declared.isdisjoint({"description", "documentation", "unit"})

    def test_block_declares_every_moved_field(self) -> None:
        assert set(CatalogNameMetadata.model_fields) == {
            "kind",
            "status",
            "physics_domain",
            "links",
            "arguments",
            "sources",
        }

    def test_moved_fields_still_live_on_the_entry_model(self) -> None:
        """The sidecar feeds the entry model; it does not replace it."""
        fields = set(StandardNameEntryBase.model_fields)
        assert {"physics_domain", "links", "arguments", "sources"} <= fields

    def test_edge_model_version_has_moved_off_its_first_value(self) -> None:
        assert CATALOG_EDGE_MODEL_VERSION != "v1"
        assert _manifest().edge_model_version == CATALOG_EDGE_MODEL_VERSION
