"""Tests for the name-only standard-name entry factory.

These entries support LLM-driven generation pipelines that compose
``name`` + ``unit`` first and fill documentation in a later pass. The
validation must enforce grammar, governance, and unit dimensional
analysis while allowing description/documentation to be absent.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from imas_standard_names.models import (
    StandardNameBase,
    StandardNameMetadataNameOnly,
    StandardNameNameOnly,
    StandardNameScalarEntry,
    StandardNameScalarNameOnly,
    StandardNameVectorNameOnly,
    create_standard_name_entry,
)


def test_name_only_scalar_minimal_dict():
    """Name-only factory accepts minimal scalar dict (no description/documentation)."""
    entry = create_standard_name_entry(
        {
            "name": "plasma_current",
            "kind": "scalar",
            "unit": "A",
        },
        name_only=True,
    )
    assert isinstance(entry, StandardNameScalarNameOnly)
    assert entry.name == "plasma_current"
    assert entry.unit == "A"


def test_name_only_vector_minimal_dict():
    entry = create_standard_name_entry(
        {
            "name": "magnetic_field",
            "kind": "vector",
            "unit": "T",
        },
        name_only=True,
    )
    assert isinstance(entry, StandardNameVectorNameOnly)


def test_name_only_metadata_minimal_dict():
    """Metadata has no unit field."""
    entry = create_standard_name_entry(
        {
            "name": "plasma_boundary",
            "kind": "metadata",
        },
        name_only=True,
    )
    assert isinstance(entry, StandardNameMetadataNameOnly)


def test_name_only_rejects_description_field():
    """Extra=forbid: passing description in name-only mode must fail."""
    with pytest.raises(ValidationError):
        create_standard_name_entry(
            {
                "name": "plasma_current",
                "kind": "scalar",
                "unit": "A",
                "description": "accidentally included",
            },
            name_only=True,
        )


def test_name_only_rejects_documentation_field():
    with pytest.raises(ValidationError):
        create_standard_name_entry(
            {
                "name": "plasma_current",
                "kind": "scalar",
                "unit": "A",
                "documentation": "accidentally included",
            },
            name_only=True,
        )


def test_name_only_enforces_grammar_vocabulary():
    """Grammar vocabulary validation runs in name-only mode.

    The ``component_of`` template is vocabulary-gated; an unknown component
    token must raise.
    """
    with pytest.raises(ValidationError, match="Grammar vocabulary"):
        create_standard_name_entry(
            {
                "name": "nonexistent_component_of_magnetic_field",
                "kind": "scalar",
                "unit": "T",
            },
            name_only=True,
        )


def test_name_only_enforces_unit_canonicalization():
    """Unit canonicalization (lexicographic reorder) still runs."""
    entry = create_standard_name_entry(
        {
            "name": "acceleration",
            "kind": "scalar",
            "unit": "s^-2.m",  # non-canonical, should reorder to m.s^-2
        },
        name_only=True,
    )
    assert entry.unit == "m.s^-2"


def test_name_only_enforces_unit_pint_validation():
    """Invalid unit tokens are rejected."""
    with pytest.raises(ValidationError):
        create_standard_name_entry(
            {
                "name": "bogus_quantity",
                "kind": "scalar",
                "unit": "notaunit",
            },
            name_only=True,
        )


def test_name_only_enforces_deprecated_rule():
    """Base governance: deprecated entries require superseded_by."""
    with pytest.raises(ValidationError, match="superseded_by"):
        create_standard_name_entry(
            {
                "name": "plasma_current",
                "kind": "scalar",
                "unit": "A",
                "status": "deprecated",
            },
            name_only=True,
        )


def test_full_factory_still_requires_description():
    """Default factory (name_only=False) still demands full documentation."""
    with pytest.raises(ValidationError):
        create_standard_name_entry(
            {
                "name": "plasma_current",
                "kind": "scalar",
                "unit": "A",
            },
            name_only=False,
        )


def test_full_factory_roundtrip_unchanged():
    """Existing full-entry creation path remains untouched."""
    entry = create_standard_name_entry(
        {
            "name": "plasma_current",
            "kind": "scalar",
            "physics_domain": "core_plasma_physics",
            "unit": "A",
            "description": "Total toroidal plasma current.",
            "documentation": "Positive when flowing counter-clockwise viewed from above.",
        },
    )
    assert isinstance(entry, StandardNameScalarEntry)
    assert entry.description == "Total toroidal plasma current."


def test_name_only_class_hierarchy():
    """Name-only subclasses inherit from StandardNameBase, NOT StandardNameEntryBase."""
    assert issubclass(StandardNameScalarNameOnly, StandardNameBase)
    assert issubclass(StandardNameVectorNameOnly, StandardNameBase)
    assert issubclass(StandardNameMetadataNameOnly, StandardNameBase)

    # They are NOT entry-base subclasses (that would reintroduce the doc requirement)
    from imas_standard_names.models import StandardNameEntryBase

    assert not issubclass(StandardNameScalarNameOnly, StandardNameEntryBase)
    assert not issubclass(StandardNameVectorNameOnly, StandardNameEntryBase)
    assert not issubclass(StandardNameMetadataNameOnly, StandardNameEntryBase)


def test_name_only_union_discrimination():
    """Union discriminator by ``kind`` field."""
    scalar = create_standard_name_entry(
        {
            "name": "plasma_current",
            "kind": "scalar",
            "unit": "A",
        },
        name_only=True,
    )
    vector = create_standard_name_entry(
        {
            "name": "magnetic_field",
            "kind": "vector",
            "unit": "T",
        },
        name_only=True,
    )
    metadata = create_standard_name_entry(
        {
            "name": "plasma_boundary",
            "kind": "metadata",
        },
        name_only=True,
    )
    assert scalar.kind == "scalar"
    assert vector.kind == "vector"
    assert metadata.kind == "metadata"
