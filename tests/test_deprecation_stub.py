"""Deprecation stubs are first-class catalog entries.

A ``status: deprecated`` stub names its live successor via ``superseded_by``.
These tests cover the model governance rule, the structural stub check, and a
SQLite catalog round-trip that preserves the deprecation metadata.
"""

from __future__ import annotations

import pytest

from imas_standard_names.database.readwrite import CatalogReadWrite
from imas_standard_names.models import (
    StandardNameScalarEntry,
    load_standard_name_entry,
)
from imas_standard_names.validation.structural import run_structural_checks


def _stub(**overrides) -> dict:
    base = {
        "name": "ion_temperature_core",
        "kind": "scalar",
        "physics_domain": "core_plasma_physics",
        "status": "deprecated",
        "unit": "eV",
        "superseded_by": "core_ion_temperature",
        "description": "Deprecated: renamed to core_ion_temperature.",
        "documentation": "Use core_ion_temperature instead.",
        "links": ["name:core_ion_temperature"],
    }
    base.update(overrides)
    return base


def _successor() -> dict:
    return {
        "name": "core_ion_temperature",
        "kind": "scalar",
        "physics_domain": "core_plasma_physics",
        "status": "active",
        "unit": "eV",
        "description": "Core ion temperature.",
        "documentation": "The core ion temperature measured by CXRS.",
    }


class TestStubModelGovernance:
    def test_valid_stub_validates(self):
        entry = StandardNameScalarEntry.model_validate(_stub())
        assert entry.status == "deprecated"
        assert entry.superseded_by == "core_ion_temperature"

    def test_deprecated_without_successor_rejected(self):
        with pytest.raises(ValueError, match="superseded_by"):
            StandardNameScalarEntry.model_validate(_stub(superseded_by=None))


class TestStructuralStubCheck:
    def test_valid_stub_set_has_no_issues(self):
        entries = {
            "ion_temperature_core": StandardNameScalarEntry.model_validate(_stub()),
            "core_ion_temperature": StandardNameScalarEntry.model_validate(
                _successor()
            ),
        }
        assert run_structural_checks(entries) == []

    def test_deprecated_without_successor_flagged(self):
        # Bypass model validation to construct the malformed entry the
        # structural check is meant to catch.
        bad = load_standard_name_entry(_stub(superseded_by=None))
        issues = run_structural_checks({"ion_temperature_core": bad})
        assert any("superseded_by" in i for i in issues)


class TestCatalogRoundTrip:
    def test_sqlite_preserves_deprecation_metadata(self):
        catalog = CatalogReadWrite()
        catalog.load_models(
            [
                StandardNameScalarEntry.model_validate(_successor()),
                StandardNameScalarEntry.model_validate(_stub()),
            ]
        )
        fetched = catalog.get("ion_temperature_core")
        assert fetched is not None
        assert fetched.status == "deprecated"
        assert fetched.superseded_by == "core_ion_temperature"
