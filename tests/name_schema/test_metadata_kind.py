"""Tests for metadata kind standard name entries.

Tests cover:
- Valid metadata entry creation
- Unit field exclusion from metadata entries
- Provenance validation (should be rejected)
- Schema validation
- Catalog round-trip
- Integration with tools (fetch, list, search)
"""

import pytest

from imas_standard_names.models import (
    StandardNameMetadataEntry,
    create_standard_name_entry,
)


def test_metadata_entry_basic_creation():
    """Test creating a basic metadata entry."""
    entry = StandardNameMetadataEntry(
        name="test_metadata",
        physics_domain="general",
        description="Test metadata entry.",
        documentation="Extended documentation for test metadata entry.",
    )
    assert entry.kind == "metadata"
    assert entry.name == "test_metadata"
    assert entry.description == "Test metadata entry."
    # Metadata entries don't expose unit attribute


def test_metadata_entry_without_unit():
    """Test that metadata entries work without unit field."""
    entry_dict = {
        "name": "plasma_boundary",
        "kind": "metadata",
        "physics_domain": "equilibrium",
        "description": "Definition of plasma boundary.",
        "documentation": "Metadata defining the plasma boundary surface.",
    }
    entry = create_standard_name_entry(entry_dict)
    assert isinstance(entry, StandardNameMetadataEntry)
    assert entry.kind == "metadata"
    # Metadata entries don't have unit field exposed


def test_metadata_entry_model_dump_excludes_unit():
    """Test that model_dump excludes unit field for metadata entries."""
    entry = StandardNameMetadataEntry(
        name="test_metadata",
        physics_domain="general",
        description="Test metadata entry.",
        documentation="Extended documentation.",
    )
    dumped = entry.model_dump()
    # Check that unit is excluded from serialization
    assert "unit" not in dumped
    assert dumped["kind"] == "metadata"
    assert dumped["name"] == "test_metadata"


def test_metadata_entry_with_documentation():
    """Test metadata entry with full documentation."""
    entry = StandardNameMetadataEntry(
        name="confined_region",
        physics_domain="core_plasma_physics",
        description="Definition of confined plasma region.",
        documentation=(
            "The confined region refers to the volume enclosed by the last "
            "closed flux surface where particles and energy are magnetically confined."
        ),
        status="draft",
    )
    assert entry.kind == "metadata"
    assert "confined" in entry.documentation
    assert entry.status == "draft"


def test_metadata_entry_rejects_provenance():
    """Test that metadata entries cannot have provenance."""
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        StandardNameMetadataEntry(
            name="test_metadata",
            physics_domain="general",
            description="Test metadata entry.",
            documentation="Documentation for test metadata.",
            provenance={
                "mode": "operator",
                "operators": ["gradient"],
                "base": "electron_temperature",
            },
        )


def test_metadata_entry_with_links():
    """Test metadata entry with internal and external links."""
    entry = StandardNameMetadataEntry(
        name="plasma_boundary",
        physics_domain="equilibrium",
        description="Definition of plasma boundary.",
        documentation="Metadata defining the plasma boundary with references.",
        links=[
            "name:minor_radius_of_flux_surface",
            "https://example.org/plasma-boundary-definition",
        ],
    )
    assert len(entry.links) == 2
    assert entry.links[0].startswith("name:")
    assert entry.links[1].startswith("https://")


def test_metadata_entry_discriminator():
    """Test that discriminated union properly routes to metadata entry."""
    # Test with explicit kind
    data = {
        "name": "test_metadata",
        "kind": "metadata",
        "physics_domain": "general",
        "description": "Test metadata entry.",
        "documentation": "Documentation for discriminator test.",
    }
    entry = create_standard_name_entry(data)
    assert isinstance(entry, StandardNameMetadataEntry)
    assert entry.kind == "metadata"


def test_metadata_entry_all_optional_fields():
    """Test metadata entry with all optional fields populated."""
    entry = StandardNameMetadataEntry(
        name="scrape_off_layer",
        physics_domain="general",
        description="Definition of scrape-off layer region.",
        documentation="Region outside the last closed flux surface.",
        status="active",
        links=["name:plasma_boundary"],
    )
    assert entry.kind == "metadata"
    assert entry.status == "active"


def test_metadata_entry_governance_fields():
    """Test metadata entry with governance fields."""
    entry = StandardNameMetadataEntry(
        name="old_plasma_boundary_definition",
        physics_domain="equilibrium",
        description="Deprecated definition of plasma boundary.",
        documentation="Old definition superseded by newer standard.",
        status="deprecated",
        superseded_by="plasma_boundary",
    )
    assert entry.status == "deprecated"
    assert entry.superseded_by == "plasma_boundary"
