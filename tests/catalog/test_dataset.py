"""Tests for the SPA dataset builder (``imas_standard_names.catalog.dataset``).

These tests exercise the conversion from the published per-domain YAML
catalog into the SPA's flat-JSON shape (``CATALOG_VERSION`` +
``CATEGORIES`` + ``GRAMMAR_VOCAB`` + ``NAMES``). End-to-end assertions
run against a small, self-contained synthetic catalog written to a tmp
dir (see ``catalog_dir`` / ``site_dataset``), so they are deterministic
and need no external checkout. We deliberately avoid asserting against
the live generated ISNC catalog — its data drifts as the upstream
vocabulary evolves, which made name-specific assertions flaky.
"""

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from imas_standard_names.catalog import (
    build_site_dataset,
    dataset as dataset_module,
    write_site_dataset,
)
from imas_standard_names.catalog.dataset import (
    _arguments_parent,
    _build_grammar_vocab,
    _derive_grammar_facets,
    _extract_sign,
    _GrammarFacets,
    _humanise_domain,
    _local_ir_peel,
    _normalise_see_also,
    _normalise_sources,
    _normalise_status,
    _parent_token,
    _sort_tier,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# A small, curated catalog covering the behaviours the dataset tests
# assert: scalar vs vector algebra, axis-projection components + magnitude,
# locus metadata, subject extraction, see-also / source normalisation, and
# canonical sort tiers. Every name is a real, parseable Standard Name, so
# the grammar parser produces genuine facets — but the corpus is fixed in
# this file, so the dataset never depends on the drifting live catalog.
_DOC = "Body documentation paragraph for the synthetic fixture entry."

SYNTHETIC_EQUILIBRIUM: list[dict] = [
    {
        "name": "magnetic_field",
        "kind": "vector",
        "status": "active",
        "description": "Magnetic flux density vector.",
        "documentation": _DOC,
        "unit": "T",
        "physics_domain": "equilibrium",
        "links": [],
        "sources": [],
    },
    {
        "name": "radial_magnetic_field",
        "kind": "scalar",
        "status": "active",
        "description": "Radial component of the magnetic field.",
        "documentation": _DOC,
        "unit": "T",
        "physics_domain": "equilibrium",
        "arguments": [
            {"name": "magnetic_field", "operator_kind": "projection", "axis": "radial"}
        ],
    },
    {
        "name": "poloidal_magnetic_field",
        "kind": "scalar",
        "status": "active",
        "description": "Poloidal component of the magnetic field.",
        "documentation": _DOC,
        "unit": "T",
        "physics_domain": "equilibrium",
        "arguments": [
            {
                "name": "magnetic_field",
                "operator_kind": "projection",
                "axis": "poloidal",
            }
        ],
    },
    {
        "name": "magnetic_field_magnitude",
        "kind": "scalar",
        "status": "active",
        "description": "Magnitude of the magnetic field.",
        "documentation": _DOC,
        "unit": "T",
        "physics_domain": "equilibrium",
        "arguments": [{"name": "magnetic_field", "operator_kind": "magnitude"}],
    },
    {
        "name": "safety_factor",
        "kind": "scalar",
        "status": "active",
        "description": "Safety factor q.",
        "documentation": _DOC,
        "unit": "1",
        "physics_domain": "equilibrium",
    },
    {
        "name": "safety_factor_at_magnetic_axis",
        "kind": "scalar",
        "status": "active",
        "description": "Safety factor at the magnetic axis.",
        "documentation": _DOC,
        "unit": "1",
        "physics_domain": "equilibrium",
        "arguments": [{"name": "safety_factor", "operator_kind": "locus"}],
    },
    {
        "name": "toroidal_magnetic_field_at_magnetic_axis",
        "kind": "scalar",
        "status": "active",
        "description": "Toroidal magnetic field at the magnetic axis.",
        "documentation": _DOC,
        "unit": "T",
        "physics_domain": "equilibrium",
    },
    {
        "name": "vertical_coordinate_of_geometric_axis",
        "kind": "scalar",
        "status": "active",
        "description": "Vertical coordinate of the geometric axis.",
        "documentation": _DOC,
        "unit": "m",
        "physics_domain": "equilibrium",
    },
    {
        "name": "plasma_inductance",
        "kind": "scalar",
        "status": "active",
        "description": "Plasma self-inductance.",
        "documentation": (
            "Self-inductance of the plasma current.\n\n"
            "Sign convention: Positive for a current in the +phi direction."
        ),
        "unit": "H",
        "physics_domain": "equilibrium",
    },
]

SYNTHETIC_CORE: list[dict] = [
    {
        "name": "electron_temperature",
        "kind": "scalar",
        "status": "active",
        "description": "Electron temperature.",
        "documentation": _DOC,
        "unit": "eV",
        "physics_domain": "core_plasma_physics",
        "links": ["name:ion_temperature"],
        "sources": [
            {
                "id": "dd:core_profiles/profiles_1d/electrons/temperature",
                "dd_path": "core_profiles/profiles_1d/electrons/temperature",
                "status": "composed",
            }
        ],
    },
    {
        "name": "ion_temperature",
        "kind": "scalar",
        "status": "active",
        "description": "Ion temperature.",
        "documentation": _DOC,
        "unit": "eV",
        "physics_domain": "core_plasma_physics",
    },
]

SYNTHETIC_MANIFEST: dict = {
    "catalog_name": "synthetic-test-catalog",
    "cocos_convention": 11,
    "grammar_version": "test-grammar-1.0",
    "isn_model_version": "test-model-1.0",
    "dd_version_lineage": ["4.0.0"],
    "generated_by": "test",
    "generated_at": "2026-01-01T00:00:00Z",
    "candidate_count": 11,
    "published_count": 11,
    "domains_included": ["equilibrium", "core_plasma_physics"],
}


@pytest.fixture
def catalog_dir(tmp_path: Path) -> Path:
    """Write the synthetic catalog (entries + manifest) to a tmp dir and
    return its ``standard_names`` directory. Deterministic — no live
    checkout, so the dataset tests never depend on drifting catalog data."""
    names_dir = tmp_path / "standard_names"
    names_dir.mkdir()
    (names_dir / "equilibrium.yml").write_text(
        yaml.safe_dump(SYNTHETIC_EQUILIBRIUM), encoding="utf-8"
    )
    (names_dir / "core_plasma_physics.yml").write_text(
        yaml.safe_dump(SYNTHETIC_CORE), encoding="utf-8"
    )
    (tmp_path / "catalog.yml").write_text(
        yaml.safe_dump(SYNTHETIC_MANIFEST), encoding="utf-8"
    )
    return names_dir


@pytest.fixture
def site_dataset(catalog_dir: Path) -> dict:
    """Built dataset from the synthetic catalog (one parse pass per test)."""
    return build_site_dataset(catalog_dir)


@pytest.fixture
def manifest() -> dict:
    """The synthetic manifest dict (mirrors the on-disk ``catalog.yml``)."""
    return dict(SYNTHETIC_MANIFEST)


# ---------------------------------------------------------------------------
# Pure-helper tests (no catalog dependency)
# ---------------------------------------------------------------------------


class TestHumaniseDomain:
    """``_humanise_domain`` matches the SPA prototype's compact labels."""

    def test_simple_two_word_slug(self) -> None:
        assert _humanise_domain("auxiliary_heating") == "Auxiliary Heating"

    def test_single_word_slug(self) -> None:
        assert _humanise_domain("equilibrium") == "Equilibrium"

    def test_measurement_abbreviated(self) -> None:
        assert (
            _humanise_domain("particle_measurement_diagnostics")
            == "Particle Meas. Diagnostics"
        )

    def test_electromagnetic_abbreviated(self) -> None:
        assert (
            _humanise_domain("electromagnetic_wave_diagnostics")
            == "EM Wave Diagnostics"
        )

    def test_empty_returns_empty(self) -> None:
        assert _humanise_domain("") == ""


class TestExtractSign:
    """``_extract_sign`` separates the ``Sign convention:`` paragraph."""

    def test_extracts_trailing_paragraph(self) -> None:
        doc = (
            "Body of documentation explaining the quantity.\n\n"
            "Sign convention: Positive when the field points outward."
        )
        long, sign = _extract_sign(doc)
        assert long == "Body of documentation explaining the quantity."
        assert sign == "Positive when the field points outward."

    def test_returns_none_when_absent(self) -> None:
        long, sign = _extract_sign("Plain documentation only.")
        assert long == "Plain documentation only."
        assert sign is None

    def test_empty_returns_empty(self) -> None:
        long, sign = _extract_sign("")
        assert long == ""
        assert sign is None


class TestNormaliseSeeAlso:
    """``_normalise_see_also`` strips the ``name:`` prefix."""

    def test_strips_name_prefix(self) -> None:
        assert _normalise_see_also(["name:foo", "name:bar_baz"]) == ["foo", "bar_baz"]

    def test_drops_external_urls(self) -> None:
        assert _normalise_see_also(["https://example.org", "name:foo"]) == ["foo"]

    def test_handles_none(self) -> None:
        assert _normalise_see_also(None) == []


class TestNormaliseSources:
    """``_normalise_sources`` dispatches generic source-system bindings."""

    def test_extracts_dd_path_and_status(self) -> None:
        sources = [
            {
                "id": "dd:equilibrium/time_slice/ip",
                "dd_path": "equilibrium/time_slice/ip",
                "status": "composed",
            }
        ]
        assert _normalise_sources(sources) == [
            {"kind": "imas-dd", "ref": "equilibrium/time_slice/ip"}
        ]

    def test_falls_back_to_id_minus_prefix(self) -> None:
        sources = [{"id": "dd:foo/bar", "status": "attached"}]
        assert _normalise_sources(sources) == [{"kind": "imas-dd", "ref": "foo/bar"}]

    def test_handles_none(self) -> None:
        assert _normalise_sources(None) == []

    def test_pinned_source_resolves_through_imas_python(self) -> None:
        source = {
            "kind": "imas-dd",
            "ref": "summary/global_quantities/energy_thermal/value",
            "version": "4.0.0",
            "dd_documentation": {
                "leaf": "Inline content must not override imas-python.",
            },
            "model": "must-not-leak",
            "status": "composed",
            "source_id": "must-not-leak",
        }
        projected = _normalise_sources([source])[0]

        assert projected["kind"] == "imas-dd"
        assert projected["ref"] == "summary/global_quantities/energy_thermal/value"
        assert projected["version"] == "4.0.0"
        assert projected["leaf_definition"] == "Value"
        assert projected["parent_path"] == ("summary/global_quantities/energy_thermal")
        assert "Thermal plasma energy content" in projected["parent_definition"]
        assert projected["data_type"] == "FLT"
        assert projected["unit"] == "J"
        assert projected["coordinates"] == ["time"]
        assert projected["resolution_source"] == "imas-python"
        assert projected["resolution_status"] == "resolved"
        assert "model" not in projected
        assert "source_id" not in projected
        assert "dd_path" not in projected
        assert "dd_version" not in projected
        assert "semantic_facet" not in projected

    def test_facility_binding_is_not_sent_to_dd_resolver(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def reject_dd_dispatch(path: str, version: str) -> dict:
            pytest.fail(f"facility binding was sent to DD resolver: {path} {version}")

        monkeypatch.setattr(dataset_module, "_resolve_dd_source", reject_dd_dispatch)
        source = {
            "kind": "west-uda",
            "ref": "MAI/PLASMA/IP",
            "version": "62253",
            "semantic_facet": "measured",
        }

        projected = _normalise_sources([source])[0]

        assert projected == {
            "kind": "west-uda",
            "ref": "MAI/PLASMA/IP",
            "version": "62253",
            "resolution_status": "unresolved",
            "resolution_error": "No resolver registered for source kind 'west-uda'",
        }

    def test_unresolvable_path_is_recorded_without_raising(self) -> None:
        source = {
            "kind": "imas-dd",
            "ref": "summary/global_quantities/not_a_dd_leaf",
            "version": "4.0.0",
        }

        projected = _normalise_sources([source])[0]

        assert projected["resolution_source"] == "imas-python"
        assert projected["resolution_status"] == "unresolved"
        assert "not_a_dd_leaf" in projected["resolution_error"]
        assert "leaf_definition" not in projected

    def test_legacy_inline_source_is_still_accepted(self) -> None:
        source = {
            "dd_path": "summary/global_quantities/energy_thermal/value",
            "dd_documentation": {
                "leaf": "Value",
                "parent_path": "summary/global_quantities/energy_thermal",
                "parent": "Thermal plasma energy.",
                "data_type": "FLT_0D",
                "lifecycle_status": "active",
            },
            "enhanced_context": {
                "description": "Context inferred from the parent hierarchy.",
                "kind": "generated",
            },
        }

        projected = _normalise_sources([source])[0]

        assert projected["leaf_definition"] == "Value"
        assert projected["parent_definition"] == "Thermal plasma energy."
        assert projected["resolution_source"] == "catalog-inline"
        assert projected["resolution_status"] == "legacy-inline"

    def test_enhanced_context_without_description_never_leaks_dict(self) -> None:
        source = {
            "dd_path": "summary/global_quantities/energy_thermal/value",
            "enhanced_context": {"kind": "generated"},
        }
        projected = _normalise_sources([source])
        assert "enhanced_context" not in projected[0]

    def test_enhanced_context_empty_description_dropped(self) -> None:
        source = {
            "dd_path": "summary/global_quantities/energy_thermal/value",
            "enhanced_context": {"description": "", "kind": "generated"},
        }
        projected = _normalise_sources([source])
        assert "enhanced_context" not in projected[0]


class TestGrammarFacets:
    """``_derive_grammar_facets`` extracts the IR signals the emitter needs."""

    def test_locus_detected(self) -> None:
        facets = _derive_grammar_facets("safety_factor_at_magnetic_axis")
        assert facets.has_locus is True
        assert facets.locus_token == "magnetic_axis"

    def test_projection_detected(self) -> None:
        facets = _derive_grammar_facets("poloidal_magnetic_field")
        assert facets.has_projection is True
        assert facets.axis == "poloidal"

    def test_pure_base_has_no_facets(self) -> None:
        facets = _derive_grammar_facets("safety_factor")
        assert facets.has_projection is False
        assert facets.has_locus is False
        assert facets.base_token == "safety_factor"


# ---------------------------------------------------------------------------
# Real-catalog tests — auto-skip when ISNC checkout is missing
# ---------------------------------------------------------------------------


class TestDatasetShape:
    """Top-level dataset keys and counts derive from the actual YAML files.

    The CATALOG_VERSION label and category counts use ``len(names)`` —
    the actual number of records emitted — rather than the manifest's
    ``published_count``. (Manifest counts can lag if the codex export
    filter excluded entries after the manifest was written, and we
    want the SPA to show the truth.)
    """

    def test_loads_catalog(self, site_dataset: dict) -> None:
        assert "NAMES" in site_dataset
        assert "CATEGORIES" in site_dataset
        assert "GRAMMAR_VOCAB" in site_dataset
        assert "STANDARD_TERMS" in site_dataset
        assert "CATALOG_VERSION" in site_dataset

        names = site_dataset["NAMES"]
        assert len(names) > 0, "real ISNC catalog must yield at least one record"

    def test_catalog_version_includes_grammar_and_count(
        self, site_dataset: dict, manifest: dict
    ) -> None:
        version = site_dataset["CATALOG_VERSION"]
        names = site_dataset["NAMES"]
        # Manifest's grammar_version always appears verbatim.
        assert manifest["grammar_version"] in version
        # The displayed count is the actual emitted count, not the
        # manifest's possibly-stale ``published_count`` claim.
        assert str(len(names)) in version

    def test_categories_derived(self, site_dataset: dict, manifest: dict) -> None:
        cats = site_dataset["CATEGORIES"]
        # Every domain in the manifest should appear in CATEGORIES.
        category_ids = {c["id"] for c in cats}
        for domain in manifest["domains_included"]:
            assert domain in category_ids, f"domain {domain} missing from CATEGORIES"

        # Counts sum to total NAMES length.
        total = sum(c["count"] for c in cats)
        assert total == len(site_dataset["NAMES"])

        # Counts decrease (descending).
        counts = [c["count"] for c in cats]
        assert counts == sorted(counts, reverse=True)


def _find_record(dataset: dict, name: str) -> dict:
    """Helper — fetch a NAMES record by name, asserting it exists."""
    for record in dataset["NAMES"]:
        if record["name"] == name:
            return record
    pytest.fail(f"record {name!r} not found in dataset")


class TestRecordShape:
    """A representative NAMES record carries all SPA-expected fields."""

    def test_plasma_inductance_full_shape(self, site_dataset: dict) -> None:
        record = _find_record(site_dataset, "plasma_inductance")

        # Identity
        assert record["name"] == "plasma_inductance"
        assert record["category"] == "equilibrium"

        # Required keys all present. ``display_kind`` / ``kind`` have
        # been retired; ``algebra`` (scalar/vector/tensor/complex/metadata)
        # is the canonical kind axis.
        required = {
            "name",
            "category",
            "group",
            "parent",
            "algebra",
            "status",
            "unit",
            "tags",
            "short",
            "long",
            "sign",
            "seeAlso",
            "arguments",
            "sources",
            "parse",
            "components",
            "magnitude",
            "children",
        }
        assert required.issubset(record.keys()), (
            f"missing keys: {required - set(record.keys())}"
        )
        # The retired synthetic kind axes must NOT be present.
        assert "display_kind" not in record
        assert "kind" not in record

        # Type checks.
        assert isinstance(record["tags"], list)
        assert isinstance(record["seeAlso"], list)
        assert isinstance(record["arguments"], list)
        assert isinstance(record["sources"], list)
        assert isinstance(record["parse"], list)
        assert isinstance(record["components"], list)
        assert isinstance(record["children"], list)
        assert all(
            isinstance(seg, dict) and {"role", "text"}.issubset(seg.keys())
            for seg in record["parse"]
        )

    def test_unit_present_for_physical_quantity(self, site_dataset: dict) -> None:
        record = _find_record(site_dataset, "plasma_inductance")
        assert record["unit"] == "H"

    def test_every_fixture_entry_has_nonempty_detail_payload(
        self, site_dataset: dict
    ) -> None:
        required_content = {"name", "short", "long", "unit", "parse"}
        for record in site_dataset["NAMES"]:
            payload = {key: record.get(key) for key in required_content}
            assert all(payload.values()), f"empty detail payload for {record['name']}"

    def test_generic_bindings_survive_resolution_failure_in_built_dataset(
        self,
        catalog_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        catalog_file = catalog_dir / "core_plasma_physics.yml"
        entries = yaml.safe_load(catalog_file.read_text(encoding="utf-8"))
        entries[0]["sources"] = [
            {
                "kind": "imas-dd",
                "ref": "summary/global_quantities/energy_thermal/value",
                "version": "4.0.0",
            }
        ]
        entries[1]["sources"] = [
            {
                "kind": "imas-dd",
                "ref": "summary/global_quantities/not_a_dd_leaf",
                "version": "4.0.0",
            }
        ]
        catalog_file.write_text(yaml.safe_dump(entries), encoding="utf-8")

        def resolve(ref: str, version: str) -> dict:
            assert version == "4.0.0"
            if ref.endswith("not_a_dd_leaf"):
                raise LookupError("no such Data Dictionary leaf")
            return {
                "leaf_definition": "Value",
                "parent_path": "summary/global_quantities/energy_thermal",
                "parent_definition": "Thermal plasma energy content.",
                "data_type": "FLT",
                "unit": "J",
                "coordinates": ["time"],
                "resolution_source": "imas-python",
                "resolution_status": "resolved",
            }

        monkeypatch.setattr(dataset_module, "_resolve_dd_source", resolve)
        dataset = build_site_dataset(catalog_dir)
        resolved = _find_record(dataset, "electron_temperature")
        unresolved = _find_record(dataset, "ion_temperature")

        assert resolved["short"] == "Electron temperature."
        assert resolved["long"] == _DOC
        assert resolved["unit"] == "eV"
        assert resolved["sources"][0]["parent_definition"] == (
            "Thermal plasma energy content."
        )
        assert resolved["sources"][0]["resolution_status"] == "resolved"
        assert unresolved["short"] == "Ion temperature."
        assert unresolved["long"] == _DOC
        assert unresolved["unit"] == "eV"
        assert unresolved["sources"][0]["resolution_status"] == "unresolved"
        assert "LookupError" in unresolved["sources"][0]["resolution_error"]


class TestGrammarMetadata:
    """Structural cues (axis, locus) still surface on the record even
    though ``display_kind`` no longer does."""

    def test_poloidal_magnetic_field_keeps_axis(self, site_dataset: dict) -> None:
        record = _find_record(site_dataset, "poloidal_magnetic_field")
        assert record["axis"] == "poloidal"
        assert record["algebra"] == "scalar"

    def test_safety_factor_at_magnetic_axis_keeps_locus(
        self, site_dataset: dict
    ) -> None:
        record = _find_record(site_dataset, "safety_factor_at_magnetic_axis")
        assert record["locus"] == "magnetic_axis"


class TestAlgebraAxis:
    """Algebraic kind (scalar/vector/tensor/complex/metadata) is the sole
    kind axis on each record. The retired synthetic ``display_kind`` is
    gone from the JSON output."""

    def test_magnetic_field_is_vector(self, site_dataset: dict) -> None:
        record = _find_record(site_dataset, "magnetic_field")
        assert record["algebra"] == "vector"

    def test_scalar_default(self, site_dataset: dict) -> None:
        record = _find_record(site_dataset, "electron_temperature")
        assert record["algebra"] == "scalar"

    @pytest.mark.parametrize(
        "name",
        [
            "toroidal_magnetic_field_at_magnetic_axis",
            "vertical_coordinate_of_geometric_axis",
        ],
    )
    def test_axis_bearing_scalar_keeps_declared_kind(
        self, site_dataset: dict, name: str
    ) -> None:
        record = _find_record(site_dataset, name)
        assert record["axis"] is not None
        assert record["algebra"] == "scalar"

    def test_presented_kind_matches_declared_kind(self, site_dataset: dict) -> None:
        declared_by_name = {
            entry["name"]: entry["kind"]
            for entry in [*SYNTHETIC_EQUILIBRIUM, *SYNTHETIC_CORE]
        }
        assert len(site_dataset["NAMES"]) == len(declared_by_name)
        for record in site_dataset["NAMES"]:
            assert record["algebra"] == declared_by_name[record["name"]]

    def test_every_record_has_algebra(self, site_dataset: dict) -> None:
        valid = {"scalar", "vector", "tensor", "complex", "metadata"}
        for record in site_dataset["NAMES"]:
            assert record["algebra"] in valid, (
                f"{record['name']}: bad algebra {record['algebra']!r}"
            )

    def test_magnitude_is_scalar(self, site_dataset: dict) -> None:
        # magnitude_of_<vector> is a true scalar (rotation-invariant norm).
        import pytest

        rec = next(
            (
                r
                for r in site_dataset["NAMES"]
                if r["name"] == "magnetic_field_magnitude"
            ),
            None,
        )
        if rec is None:
            pytest.skip("magnetic_field_magnitude not in catalog")
        assert rec["algebra"] == "scalar"


class TestVectorReverseLinks:
    """Vectors carry ``components`` and (when present) ``magnitude``."""

    def test_magnetic_field_components(self, site_dataset: dict) -> None:
        record = _find_record(site_dataset, "magnetic_field")
        components = record["components"]
        assert len(components) >= 2, (
            f"magnetic_field should have ≥2 components, got {components}"
        )
        # Each component carries name + axis; axes are unique.
        axes = {c["axis"] for c in components}
        assert len(axes) == len(components)
        for c in components:
            assert isinstance(c["name"], str) and c["name"]
            assert isinstance(c["axis"], str) and c["axis"]

    def test_scalar_has_empty_components(self, site_dataset: dict) -> None:
        record = _find_record(site_dataset, "electron_temperature")
        assert record["components"] == []
        assert record["magnitude"] is None

    def test_magnitude_only_when_catalog_has_it(self, site_dataset: dict) -> None:
        """``magnitude`` is set only when ``magnitude_of_<name>`` exists.

        Source-driven invariant: no speculative magnitude creation.
        Current catalog has no magnitudes of vectors, so all vectors'
        ``magnitude`` should be None.
        """
        record = _find_record(site_dataset, "magnetic_field")
        # Either it exists in the catalog (string), or None — never auto-faked.
        magnitude = record["magnitude"]
        assert magnitude is None or isinstance(magnitude, str)
        if isinstance(magnitude, str):
            # If set, the referenced name must exist in the dataset.
            assert any(n["name"] == magnitude for n in site_dataset["NAMES"])

    def test_children_index(self, site_dataset: dict) -> None:
        record = _find_record(site_dataset, "magnetic_field")
        children = record["children"]
        # magnetic_field has projection + qualifier children.
        assert any(c["operator_kind"] == "projection" for c in children), (
            f"expected projection children, got {children}"
        )


class TestSubject:
    """``subject`` is the first qualifier matching the closed Subject enum."""

    def test_electron_subject(self, site_dataset: dict) -> None:
        assert (
            _find_record(site_dataset, "electron_temperature")["subject"] == "electron"
        )

    def test_ion_subject(self, site_dataset: dict) -> None:
        assert _find_record(site_dataset, "ion_temperature")["subject"] == "ion"

    def test_unqualified_has_no_subject(self, site_dataset: dict) -> None:
        # ``magnetic_field`` is a bare base — no subject qualifier.
        assert _find_record(site_dataset, "magnetic_field")["subject"] is None


class TestParseSegments:
    """Parse segments concatenate back to the input name (modulo separators)."""

    @pytest.mark.parametrize(
        "name",
        [
            "total_plasma_current",
            "poloidal_magnetic_field",
            "safety_factor_at_magnetic_axis",
            "magnetic_field_magnitude",
            "flux_surface_averaged_magnetic_field",
            "minimum_safety_factor",
        ],
    )
    def test_parse_segments_cover_name_tokens(self, name: str) -> None:
        # Parser-only: derive segments directly from the name (no dataset /
        # catalog dependency) so every parametrised name is always exercised.
        segments = _derive_grammar_facets(name).parse_segments
        # Every token segment carries a non-empty text. Joining with
        # underscores (after stripping any leading `_` artefacts) should
        # cover all underscore-separated tokens in the original name.
        original_tokens = set(name.split("_"))
        emitted_tokens: set[str] = set()
        for seg in segments:
            text = seg["text"]
            assert text, f"empty text segment in parse: {seg!r}"
            for piece in text.split("_"):
                if piece:
                    emitted_tokens.add(piece)
        # Tokens emitted should at minimum include all original tokens.
        missing = original_tokens - emitted_tokens
        assert not missing, f"parse missing tokens {missing} for {name}"

    def test_parse_failure_is_graceful(self) -> None:
        # A deliberately invalid base token (looks like a name but won't
        # resolve in the closed vocabulary) must round-trip to a single
        # ``unparseable`` segment instead of crashing the build.
        from imas_standard_names.catalog.dataset import _derive_grammar_facets

        facets = _derive_grammar_facets("definitely_not_a_known_thing_xyz")
        assert facets.parsed is False
        assert len(facets.parse_segments) == 1
        assert facets.parse_segments[0]["role"] == "unparseable"

    def test_unparseable_does_not_crash_dataset(self, tmp_path: Path) -> None:
        # Construct a tiny standalone YAML with an unparseable name and
        # run the builder on it — the result should still produce a
        # record with an ``unparseable`` parse segment.
        domain_yaml = tmp_path / "test_domain.yml"
        domain_yaml.write_text(
            yaml.safe_dump(
                [
                    {
                        "name": "definitely_not_a_known_thing_xyz",
                        "kind": "scalar",
                        "status": "active",
                        "description": "Test entry that the parser cannot decompose.",
                        "documentation": "This entry exists only to exercise the unparseable fallback.",
                        "unit": "1",
                        "physics_domain": "general",
                    }
                ]
            ),
            encoding="utf-8",
        )

        dataset = build_site_dataset(tmp_path)
        assert len(dataset["NAMES"]) == 1
        record = dataset["NAMES"][0]
        assert record["parse"][0]["role"] == "unparseable"


class TestSeeAlsoNormalisation:
    """``links: [name:foo]`` flattens to ``seeAlso: ['foo']``."""

    def test_name_prefix_stripped(self, site_dataset: dict) -> None:
        # Find a record with at least one internal link.
        for record in site_dataset["NAMES"]:
            if record["seeAlso"]:
                for ref in record["seeAlso"]:
                    assert not ref.startswith("name:"), (
                        f"seeAlso entry {ref!r} should not retain 'name:' prefix"
                    )
                return
        pytest.skip("no records with seeAlso links in current catalog")


class TestSourcesNormalisation:
    """Sources expose generic bindings without operational ledger fields."""

    def test_sources_expose_path_without_operational_status(
        self, site_dataset: dict
    ) -> None:
        for record in site_dataset["NAMES"]:
            if record["sources"]:
                for src in record["sources"]:
                    assert "kind" in src
                    assert "ref" in src
                    assert "status" not in src
                    assert "id" not in src
                    assert src["ref"]
                    assert "semantic_facet" not in src
                return
        pytest.skip("no records with sources in current catalog")


# ---------------------------------------------------------------------------
# Roundtrip — write + load
# ---------------------------------------------------------------------------


class TestWriteSiteDataset:
    """``write_site_dataset`` produces valid JSON on disk."""

    def test_writes_to_disk(self, catalog_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "dataset.json"
        count = write_site_dataset(catalog_dir, out)

        assert out.exists()
        assert count > 0
        # Round-trip the JSON.
        data = json.loads(out.read_text(encoding="utf-8"))
        assert len(data["NAMES"]) == count


# ---------------------------------------------------------------------------
# Parent resolution — peel one layer (operator | projection | qualifier | locus)
# ---------------------------------------------------------------------------


class TestLocalIrPeel:
    """Standalone IR peel: one layer at a time, recursion is implicit.

    A direct ``ir.base.token`` shortcut collapses every layer at once;
    these tests pin the intended per-layer behaviour.
    """

    def test_leaf_returns_none(self):
        assert _local_ir_peel("temperature") is None
        assert _local_ir_peel("elongation") is None

    def test_qualifier_with_locus_peels_qualifier(self):
        # The parent must retain the boundary locus rather than skipping
        # directly to `elongation`.
        assert (
            _local_ir_peel("upper_elongation_of_plasma_boundary")
            == "elongation_of_plasma_boundary"
        )
        assert (
            _local_ir_peel("lower_elongation_of_plasma_boundary")
            == "elongation_of_plasma_boundary"
        )

    def test_locus_only_peels_locus(self):
        assert _local_ir_peel("elongation_of_plasma_boundary") == "elongation"
        assert _local_ir_peel("area_of_plasma_boundary") == "area"
        assert _local_ir_peel("safety_factor_at_magnetic_axis") == "safety_factor"

    def test_qualifier_only_peels_qualifier(self):
        assert _local_ir_peel("electron_temperature") == "temperature"
        assert _local_ir_peel("ion_pressure") == "pressure"

    def test_two_qualifiers_peels_outermost_only(self):
        # `volume_averaged_ion_temperature` — two qualifiers, peel one
        # at a time. The inner SN runs its own derivation on the next
        # hop.
        assert _local_ir_peel("volume_averaged_ion_temperature") == "ion_temperature"

    def test_unparseable_returns_none(self):
        assert _local_ir_peel("garbage_name_!@#") is None


class TestArgumentsParent:
    """When the YAML carries an ``arguments`` block (canonical, graph-
    derived), prefer it over local heuristics."""

    def test_first_argument_wins(self):
        entry = {"arguments": [{"name": "elongation_of_plasma_boundary"}]}
        assert (
            _arguments_parent("upper_elongation_of_plasma_boundary", entry)
            == "elongation_of_plasma_boundary"
        )

    def test_self_loop_skipped(self):
        entry = {"arguments": [{"name": "upper_elongation_of_plasma_boundary"}]}
        assert _arguments_parent("upper_elongation_of_plasma_boundary", entry) is None

    def test_missing_arguments_returns_none(self):
        assert _arguments_parent("x", {}) is None
        assert _arguments_parent("x", {"arguments": None}) is None
        assert _arguments_parent("x", {"arguments": []}) is None

    def test_falls_through_to_second_arg_when_first_is_self(self):
        entry = {
            "arguments": [
                {"name": "upper_elongation_of_plasma_boundary"},
                {"name": "elongation_of_plasma_boundary"},
            ]
        }
        assert (
            _arguments_parent("upper_elongation_of_plasma_boundary", entry)
            == "elongation_of_plasma_boundary"
        )


class TestParentToken:
    """End-to-end ``_parent_token`` resolution: arguments > local peel > facets."""

    def test_arguments_preferred_over_local_peel(self):
        # If the YAML carries arguments, use them even when local peel
        # would say something different.
        entry = {"arguments": [{"name": "some_specific_parent"}]}
        facets = _derive_grammar_facets("upper_elongation_of_plasma_boundary")
        assert (
            _parent_token("upper_elongation_of_plasma_boundary", facets, entry)
            == "some_specific_parent"
        )

    def test_local_peel_used_when_arguments_missing(self):
        facets = _derive_grammar_facets("upper_elongation_of_plasma_boundary")
        assert (
            _parent_token("upper_elongation_of_plasma_boundary", facets, {})
            == "elongation_of_plasma_boundary"
        )

    def test_leaf_returns_none(self):
        facets = _derive_grammar_facets("elongation")
        assert _parent_token("elongation", facets, {}) is None

    def test_legacy_callsite_without_entry(self):
        # Backwards-compat: callers that don't pass `entry` still get
        # the corrected one-layer peel.
        facets = _derive_grammar_facets("electron_temperature")
        assert _parent_token("electron_temperature", facets) == "temperature"


# ---------------------------------------------------------------------------
# Status normalisation + include-draft filtering
# ---------------------------------------------------------------------------


class TestNormaliseStatus:
    """Legacy status values map to the canonical set; unknowns drop."""

    def test_canonical_values_pass_through(self) -> None:
        for value in ("draft", "active", "deprecated", "superseded"):
            assert _normalise_status(value) == value

    def test_legacy_drafted_maps_to_draft(self) -> None:
        assert _normalise_status("drafted") == "draft"

    def test_legacy_accepted_maps_to_active(self) -> None:
        assert _normalise_status("accepted") == "active"

    def test_legacy_published_maps_to_active(self) -> None:
        assert _normalise_status("published") == "active"

    def test_unknown_value_returns_none(self) -> None:
        assert _normalise_status("not_a_real_status") is None
        assert _normalise_status("nonsense") is None

    def test_missing_value_defaults_to_draft(self) -> None:
        # Catalog entries written before status was required should fall
        # to "draft" rather than be silently dropped.
        assert _normalise_status(None) == "draft"
        assert _normalise_status("") == "draft"


class TestCanonicalKinds:
    """Output JSON exposes only the five schema kinds — no display_kind."""

    def test_no_display_kind_in_records(self, site_dataset: dict) -> None:
        for record in site_dataset["NAMES"]:
            assert "display_kind" not in record
            assert "kind" not in record  # alias also dropped

    def test_algebra_is_one_of_five_canonical(self, site_dataset: dict) -> None:
        valid = {"scalar", "vector", "tensor", "complex", "metadata"}
        for record in site_dataset["NAMES"]:
            assert record["algebra"] in valid


class TestStatusFiltering:
    """All canonical statuses are always emitted; unknown statuses are dropped."""

    def _write_yaml(self, path: Path, entries: list[dict]) -> None:
        path.write_text(yaml.safe_dump(entries), encoding="utf-8")

    def test_default_emit_includes_all_known_statuses(self, tmp_path: Path) -> None:
        """build_site_dataset emits draft, deprecated, superseded, and active."""
        self._write_yaml(
            tmp_path / "test_domain.yml",
            [
                {
                    "name": "alpha",
                    "physics_domain": "general",
                    "kind": "scalar",
                    "unit": "1",
                    "status": "active",
                    "description": "A",
                    "documentation": "doc",
                },
                {
                    "name": "beta",
                    "physics_domain": "general",
                    "kind": "scalar",
                    "unit": "1",
                    "status": "draft",
                    "description": "B",
                    "documentation": "doc",
                },
                {
                    "name": "gamma",
                    "physics_domain": "general",
                    "kind": "scalar",
                    "unit": "1",
                    "status": "deprecated",
                    "description": "C",
                    "documentation": "doc",
                },
                {
                    "name": "delta",
                    "physics_domain": "general",
                    "kind": "scalar",
                    "unit": "1",
                    "status": "superseded",
                    "description": "D",
                    "documentation": "doc",
                },
            ],
        )
        ds = build_site_dataset(tmp_path)
        names = {r["name"] for r in ds["NAMES"]}
        assert names == {"alpha", "beta", "gamma", "delta"}

    def test_no_include_draft_parameter(self) -> None:
        """build_site_dataset signature must not contain include_draft."""
        import inspect

        sig = inspect.signature(build_site_dataset)
        assert "include_draft" not in sig.parameters, (
            "include_draft has been removed — the function always emits all statuses"
        )

    def test_superseded_by_round_trips(self, tmp_path: Path) -> None:
        """superseded_by from YAML is present on the emitted record."""
        self._write_yaml(
            tmp_path / "test_domain.yml",
            [
                {
                    "name": "old_name",
                    "physics_domain": "general",
                    "kind": "scalar",
                    "unit": "1",
                    "status": "superseded",
                    "superseded_by": "new_name",
                    "description": "Old",
                    "documentation": "doc",
                },
                {
                    "name": "new_name",
                    "physics_domain": "general",
                    "kind": "scalar",
                    "unit": "1",
                    "status": "active",
                    "description": "New",
                    "documentation": "doc",
                },
            ],
        )
        ds = build_site_dataset(tmp_path)
        records = {r["name"]: r for r in ds["NAMES"]}
        assert records["old_name"]["superseded_by"] == "new_name"
        assert records["new_name"]["superseded_by"] is None

    def test_legacy_status_values_mapped(self, tmp_path: Path) -> None:
        self._write_yaml(
            tmp_path / "test_domain.yml",
            [
                {
                    "name": "old_drafted",
                    "physics_domain": "general",
                    "kind": "scalar",
                    "unit": "1",
                    "status": "drafted",
                    "description": "D",
                    "documentation": "doc",
                },
                {
                    "name": "old_accepted",
                    "physics_domain": "general",
                    "kind": "scalar",
                    "unit": "1",
                    "status": "accepted",
                    "description": "E",
                    "documentation": "doc",
                },
                {
                    "name": "old_published",
                    "physics_domain": "general",
                    "kind": "scalar",
                    "unit": "1",
                    "status": "published",
                    "description": "F",
                    "documentation": "doc",
                },
            ],
        )
        # All three legacy values normalise and are emitted.
        ds = build_site_dataset(tmp_path)
        names = {r["name"]: r["status"] for r in ds["NAMES"]}
        assert names == {
            "old_drafted": "draft",
            "old_accepted": "active",
            "old_published": "active",
        }

    def test_unknown_status_dropped(self, tmp_path: Path) -> None:
        self._write_yaml(
            tmp_path / "test_domain.yml",
            [
                {
                    "name": "weird",
                    "physics_domain": "general",
                    "kind": "scalar",
                    "unit": "1",
                    "status": "not_a_real_status",
                    "description": "W",
                    "documentation": "doc",
                },
                {
                    "name": "ok",
                    "physics_domain": "general",
                    "kind": "scalar",
                    "unit": "1",
                    "status": "active",
                    "description": "O",
                    "documentation": "doc",
                },
            ],
        )
        ds = build_site_dataset(tmp_path)
        names = {r["name"] for r in ds["NAMES"]}
        assert names == {"ok"}, "unknown status must be silently dropped"


class TestSortKeys:
    """``sort_tier`` and ``sort_axis_index`` drive canonical grouping order."""

    def test_every_record_has_sort_keys(self, site_dataset: dict) -> None:
        for record in site_dataset["NAMES"]:
            assert isinstance(record.get("sort_tier"), int)
            assert isinstance(record.get("sort_axis_index"), int)
            assert 0 <= record["sort_tier"] <= 7
            assert (
                0 <= record["sort_axis_index"] <= 5 or record["sort_axis_index"] == 99
            )

    def test_magnetic_field_family_order(self, site_dataset: dict) -> None:
        """Snapshot the magnetic_field family ordering.

        Expected grammar-derived family order:
            magnetic_field           (tier 0, vector base)
            radial_magnetic_field    (tier 1, axis index 0)
            toroidal_magnetic_field  (tier 1, axis index 1)
            vertical_magnetic_field  (tier 1, axis index 2)
            poloidal_magnetic_field  (tier 1, axis index 3)
            magnetic_field_magnitude (tier 2)
            flux_surface_averaged_magnetic_field (tier 3)

        Names not present in the catalog snapshot are skipped — the
        relative order of those that ARE present must match the above.
        """
        expected = [
            "magnetic_field",
            "radial_magnetic_field",
            "toroidal_magnetic_field",
            "vertical_magnetic_field",
            "poloidal_magnetic_field",
            "magnetic_field_magnitude",
            "flux_surface_averaged_magnetic_field",
        ]
        # Build a map name -> (tier, axis_idx) for fast assertion.
        records = {r["name"]: r for r in site_dataset["NAMES"] if r["name"] in expected}
        if len(records) < 2:
            pytest.skip(
                f"need at least 2 magnetic_field family entries in catalog, "
                f"got {len(records)}"
            )

        # Sort the expected names that DO exist by (tier, axis_idx, length, name)
        # and confirm the order matches the expected list (filtered to
        # only-present names).
        present_expected = [n for n in expected if n in records]

        def sort_key(name: str):
            r = records[name]
            return (r["sort_tier"], r["sort_axis_index"], len(name), name)

        sorted_present = sorted(present_expected, key=sort_key)
        assert sorted_present == present_expected, (
            f"family ordering wrong:\n"
            f"  expected (present subset): {present_expected}\n"
            f"  got after sort:            {sorted_present}"
        )

    def test_vector_base_is_tier_zero(self, site_dataset: dict) -> None:
        record = _find_record(site_dataset, "magnetic_field")
        assert record["sort_tier"] == 0
        assert record["sort_axis_index"] == 99  # no axis projection

    def test_component_carries_axis_index(self, site_dataset: dict) -> None:
        record = _find_record(site_dataset, "poloidal_magnetic_field")
        assert record["sort_tier"] == 1
        assert record["sort_axis_index"] == 3  # poloidal

    def test_magnitude_is_tier_two(self, site_dataset: dict) -> None:
        records = [
            r for r in site_dataset["NAMES"] if r["name"] == "magnetic_field_magnitude"
        ]
        if not records:
            pytest.skip("magnetic_field_magnitude not present in current catalog")
        assert records[0]["sort_tier"] == 2


# ---------------------------------------------------------------------------
# _sort_tier operator-token classification (unit tests, no catalog needed)
# ---------------------------------------------------------------------------


def _make_facets(
    *,
    parsed: bool = True,
    operator_tokens: tuple[str, ...] = (),
    qualifier_tokens: tuple[str, ...] = (),
    has_projection: bool = False,
    has_locus: bool = False,
    has_mechanism: bool = False,
    base_token: str | None = "temperature",
    axis: str | None = None,
    locus_token: str | None = None,
) -> _GrammarFacets:
    """Build a synthetic _GrammarFacets for deterministic _sort_tier tests."""
    return _GrammarFacets(
        parsed=parsed,
        parse_segments=[],
        base_token=base_token,
        axis=axis,
        locus_token=locus_token,
        has_projection=has_projection,
        has_locus=has_locus,
        qualifier_tokens=qualifier_tokens,
        operator_tokens=operator_tokens,
        has_mechanism=has_mechanism,
    )


class TestSortTierOperatorTokens:
    """``_sort_tier`` uses ``facets.operator_tokens`` as the primary signal
    for tiers 2 and 4, with name-substring tests as the fallback only.
    """

    def test_magnitude_operator_token_gives_tier_2(self) -> None:
        facets = _make_facets(operator_tokens=("magnitude",))
        assert _sort_tier("some_field_magnitude", "scalar", "some_field", facets) == 2

    def test_norm_operator_token_gives_tier_2(self) -> None:
        facets = _make_facets(operator_tokens=("norm",))
        assert _sort_tier("some_field_norm", "scalar", "some_field", facets) == 2

    def test_gradient_operator_token_gives_tier_4(self) -> None:
        facets = _make_facets(operator_tokens=("gradient",))
        assert _sort_tier("temperature_gradient", "scalar", "temperature", facets) == 4

    def test_shear_operator_token_gives_tier_4(self) -> None:
        facets = _make_facets(operator_tokens=("shear",))
        assert _sort_tier("velocity_shear", "scalar", "velocity", facets) == 4

    def test_divergence_operator_token_gives_tier_4(self) -> None:
        facets = _make_facets(operator_tokens=("divergence",))
        assert _sort_tier("heat_flux_divergence", "scalar", "heat_flux", facets) == 4

    def test_curl_operator_token_gives_tier_4(self) -> None:
        facets = _make_facets(operator_tokens=("curl",))
        assert (
            _sort_tier("magnetic_field_curl", "scalar", "magnetic_field", facets) == 4
        )

    def test_density_operator_token_gives_tier_4(self) -> None:
        # A synthetic entry whose operator_tokens contains "density" sorts tier 4.
        facets = _make_facets(operator_tokens=("density",))
        assert _sort_tier("neutron_density", "scalar", "neutron", facets) == 4

    def test_locus_only_gives_tier_5(self) -> None:
        # Base + locus (no operator token, no tier-4 substring in name)
        # → tier 5 (point evaluation), not tier 4.
        # Note: ``neutron_density_at_x_point`` contains ``_density``
        # which triggers the regex fallback → tier 4.  The parser's
        # operator_tokens is the correct disambiguation signal; use a
        # name that parses as a pure locus to exercise tier 5.
        facets = _make_facets(has_locus=True, locus_token="magnetic_axis")
        assert (
            _sort_tier(
                "safety_factor_at_magnetic_axis", "scalar", "safety_factor", facets
            )
            == 5
        )

    def test_fallback_regex_tier2_for_unparseable(self) -> None:
        # Unparseable: no operator_tokens → regex fallback still classifies tier 2.
        facets = _make_facets(parsed=False, operator_tokens=())
        assert _sort_tier("something_magnitude_x", "scalar", "something", facets) == 2

    def test_fallback_regex_tier4_for_unparseable(self) -> None:
        # Unparseable: no operator_tokens → regex fallback still classifies tier 4.
        facets = _make_facets(parsed=False, operator_tokens=())
        assert (
            _sort_tier("temperature_gradient_profile", "scalar", "temperature", facets)
            == 4
        )

    def test_unclassified_scalar_gives_tier_7(self) -> None:
        facets = _make_facets(operator_tokens=())
        assert _sort_tier("electron_temperature", "scalar", "temperature", facets) == 7


# ---------------------------------------------------------------------------
# GRAMMAR_VOCAB — the Grammar composer's data contract
# ---------------------------------------------------------------------------


class TestGrammarVocab:
    """``_build_grammar_vocab`` emits one entry list per closed-vocabulary
    segment the Grammar composer view consumes.

    These run against the vocabulary YAML files shipped inside the package,
    so they need no ISNC catalog checkout.
    """

    SECTIONS = (
        "operators",
        "components",
        "aggregations",
        "orbits",
        "populations",
        "subjects",
        "zones",
        "channels",
        "physical_bases",
        "geometry_carriers",
        "locus_registry",
        "regions",
        "processes",
        "physics_domains",
    )

    def test_all_sections_present_and_nonempty(self) -> None:
        vocab = _build_grammar_vocab()
        for section in self.SECTIONS:
            assert section in vocab, f"missing GRAMMAR_VOCAB section {section!r}"
            assert vocab[section], f"GRAMMAR_VOCAB section {section!r} is empty"

    def test_entries_are_token_objects(self) -> None:
        """Every entry, in every section, is an object carrying a ``token``."""
        vocab = _build_grammar_vocab()
        for section in self.SECTIONS:
            for entry in vocab[section]:
                assert isinstance(entry, dict)
                assert isinstance(entry.get("token"), str) and entry["token"]

    def test_operators_carry_kind(self) -> None:
        """The composer keys prefix/postfix rendering off ``kind``."""
        vocab = _build_grammar_vocab()
        kinds = {o["kind"] for o in vocab["operators"]}
        assert "unary_prefix" in kinds
        assert "unary_postfix" in kinds

    def test_locus_entries_carry_relations(self) -> None:
        """The locus connector switch reads ``relations`` (of / at / over)."""
        vocab = _build_grammar_vocab()
        by_token = {entry["token"]: entry for entry in vocab["locus_registry"]}
        magnetic_axis = by_token.get("magnetic_axis")
        assert magnetic_axis is not None
        assert magnetic_axis["type"] == "position"
        # A position locus admits both ``at`` and ``of``.
        assert set(magnetic_axis["relations"]) == {"at", "of"}

    def test_physical_and_geometric_bases_are_disjoint(self) -> None:
        """``baseKindOf`` in the composer relies on the two base sets being
        distinct, so a token classifies as exactly one base kind."""
        vocab = _build_grammar_vocab()
        physical = {b["token"] for b in vocab["physical_bases"]}
        geometric = {c["token"] for c in vocab["geometry_carriers"]}
        assert "magnetic_field" in physical
        assert "centroid" in geometric
        assert physical.isdisjoint(geometric)

    def test_zones_preserve_canonical_intra_order(self) -> None:
        """Zone tokens are an ORDERED multi-token segment; the SPA renders
        them in canonical intra-order (vertical → radial → region → face), so
        emission must match the loader's ordered tuple, not be re-sorted."""
        from imas_standard_names.grammar import vocab_loaders

        vocab = _build_grammar_vocab()
        tokens = [z["token"] for z in vocab["zones"]]
        assert tokens == list(vocab_loaders.load_zones())
        # Spot-check the documented vertical-precedes-radial rule.
        assert tokens.index("upper") < tokens.index("inner")

    def test_channels_preserve_locked_file_order(self) -> None:
        """Channel is a single-token segment; tokens keep their locked file
        order (heat, particle, energy, momentum)."""
        from imas_standard_names.grammar import vocab_loaders

        vocab = _build_grammar_vocab()
        tokens = [c["token"] for c in vocab["channels"]]
        assert tokens == list(vocab_loaders.load_channels())

    def test_physics_domains_match_category_field(self) -> None:
        """The domain segment matches a name's ``category`` against a
        physics-domain token, so the tokens must be the domain slugs."""
        vocab = _build_grammar_vocab()
        tokens = {d["token"] for d in vocab["physics_domains"]}
        assert "equilibrium" in tokens


def _write_catalog(tmp_path: Path, manifest: dict) -> Path:
    """Write the synthetic entries plus a custom manifest and return the
    ``standard_names`` directory build_site_dataset consumes."""
    names_dir = tmp_path / "standard_names"
    names_dir.mkdir()
    (names_dir / "equilibrium.yml").write_text(
        yaml.safe_dump(SYNTHETIC_EQUILIBRIUM), encoding="utf-8"
    )
    (names_dir / "core_plasma_physics.yml").write_text(
        yaml.safe_dump(SYNTHETIC_CORE), encoding="utf-8"
    )
    (tmp_path / "catalog.yml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return names_dir


class TestReviewBatch:
    """A ``review`` export carries a fixed id-set through the manifest into
    the SPA dataset; normal builds are unaffected."""

    def test_manifest_model_accepts_review_scope_and_batch(self) -> None:
        """export_scope='review' + review_batch round-trip through the model."""
        from imas_standard_names.models import StandardNameCatalogManifest

        data = dict(SYNTHETIC_MANIFEST)
        data["export_scope"] = "review"
        data["review_batch"] = ["ion_temperature", "magnetic_field"]
        manifest = StandardNameCatalogManifest.model_validate(data)
        assert manifest.export_scope == "review"
        assert manifest.review_batch == ["ion_temperature", "magnetic_field"]

    def test_build_emits_sorted_review_batch(self, tmp_path: Path) -> None:
        """A review manifest makes build_site_dataset emit a sorted id-set."""
        manifest = dict(SYNTHETIC_MANIFEST)
        manifest["export_scope"] = "review"
        # Deliberately unsorted to prove the emitted list is sorted.
        manifest["review_batch"] = ["magnetic_field", "ion_temperature"]
        catalog_dir = _write_catalog(tmp_path, manifest)

        dataset = build_site_dataset(catalog_dir)

        assert dataset["review_batch"] == ["ion_temperature", "magnetic_field"]

    def test_normal_build_has_no_review_batch_key(self, site_dataset: dict) -> None:
        """A normal (full/domain) manifest never emits the key."""
        assert "review_batch" not in site_dataset

    def test_empty_review_batch_is_omitted(self, tmp_path: Path) -> None:
        """An empty batch list is treated as absent — no key emitted."""
        manifest = dict(SYNTHETIC_MANIFEST)
        manifest["export_scope"] = "review"
        manifest["review_batch"] = []
        catalog_dir = _write_catalog(tmp_path, manifest)

        dataset = build_site_dataset(catalog_dir)

        assert "review_batch" not in dataset


def _commit_catalog(repo: Path, message: str) -> str:
    """Commit the synthetic catalog and return its immutable identity."""
    subprocess.run(
        ["git", "add", "catalog.yml", "standard_names/equilibrium.yml"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", message], cwd=repo, check=True, capture_output=True
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_review_preview_emits_only_added_and_edited_entries(tmp_path: Path) -> None:
    """A Git-scoped preview emits the semantic head delta and exact counts."""
    repo = tmp_path / "catalog-repo"
    names_dir = repo / "standard_names"
    names_dir.mkdir(parents=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "tests@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)

    manifest = dict(SYNTHETIC_MANIFEST)
    manifest["review_batch"] = ["magnetic_field", "safety_factor"]
    (repo / "catalog.yml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    base_entries = [
        dict(SYNTHETIC_EQUILIBRIUM[0]),
        dict(SYNTHETIC_EQUILIBRIUM[4]),
        dict(SYNTHETIC_EQUILIBRIUM[6]),
    ]
    (names_dir / "equilibrium.yml").write_text(
        yaml.safe_dump(base_entries), encoding="utf-8"
    )
    base_ref = _commit_catalog(repo, "test: establish catalog baseline")

    edited = dict(SYNTHETIC_EQUILIBRIUM[4])
    edited["description"] = "Edited safety factor description."
    head_entries = [
        dict(SYNTHETIC_EQUILIBRIUM[0]),
        edited,
        dict(SYNTHETIC_EQUILIBRIUM[2]),
    ]
    (names_dir / "equilibrium.yml").write_text(
        yaml.safe_dump(head_entries, sort_keys=False), encoding="utf-8"
    )
    head_ref = _commit_catalog(repo, "test: update review entries")

    dataset = build_site_dataset(
        names_dir,
        review_base_ref=base_ref,
        review_head_ref=head_ref,
    )

    visible = {entry["name"] for entry in dataset["NAMES"]}
    assert visible == {"poloidal_magnetic_field", "safety_factor"}
    assert "magnetic_field" not in visible
    assert dataset["review_batch"] == [
        "poloidal_magnetic_field",
        "safety_factor",
    ]
    assert dataset["review_batch_provenance"] == [
        "magnetic_field",
        "safety_factor",
    ]
    assert dataset["preview_scope"] == {
        "base_ref": base_ref,
        "head_ref": head_ref,
        "added": 1,
        "edited": 1,
        "deleted": 1,
        "visible": 2,
    }
