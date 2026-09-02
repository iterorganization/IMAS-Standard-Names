import pytest

from imas_standard_names.models import create_standard_name_entry


def test_links_trailing_whitespace_preserved_if_duplicate():
    # Current implementation does not deduplicate; we document that.
    sn = create_standard_name_entry(
        {
            "kind": "scalar",
            "physics_domain": "core_plasma_physics",
            "name": "core_density",
            "description": "Core density",
            "documentation": "Density in the plasma core.",
            "unit": "m^-3",
            "status": "draft",
            "links": ["https://example.com/a", "https://example.com/a"],
        }
    )
    assert sn.links == ["https://example.com/a", "https://example.com/a"]


def test_links_external_urls():
    """Test external URL links are accepted."""
    sn = create_standard_name_entry(
        {
            "kind": "scalar",
            "physics_domain": "general",
            "name": "test_quantity",
            "description": "Test quantity",
            "documentation": "Test quantity with external links.",
            "unit": "m",
            "status": "draft",
            "links": ["https://example.org/spec", "http://doi.org/10.1234/example"],
        }
    )
    assert sn.links == ["https://example.org/spec", "http://doi.org/10.1234/example"]


def test_links_internal_references():
    """Test internal standard name references with name: prefix."""
    sn = create_standard_name_entry(
        {
            "kind": "scalar",
            "physics_domain": "general",
            "name": "test_quantity",
            "description": "Test quantity",
            "documentation": "Test quantity with internal references.",
            "unit": "m",
            "status": "draft",
            "links": ["name:plasma_boundary", "name:minor_radius_of_flux_surface"],
        }
    )
    assert sn.links == ["name:plasma_boundary", "name:minor_radius_of_flux_surface"]


def test_links_mixed_types():
    """Test mixing external URLs and internal references."""
    sn = create_standard_name_entry(
        {
            "kind": "scalar",
            "physics_domain": "general",
            "name": "test_quantity",
            "description": "Test quantity",
            "documentation": "Test quantity with mixed link types.",
            "unit": "m",
            "status": "draft",
            "links": [
                "https://example.org/spec",
                "name:plasma_boundary",
                "http://doi.org/example",
            ],
        }
    )
    assert sn.links == [
        "https://example.org/spec",
        "name:plasma_boundary",
        "http://doi.org/example",
    ]


def test_links_invalid_internal_reference():
    """Test that invalid internal references are rejected."""
    with pytest.raises(ValueError, match="not a valid standard name token"):
        create_standard_name_entry(
            {
                "kind": "scalar",
                "physics_domain": "general",
                "name": "test_quantity",
                "description": "Test",
                "unit": "m",
                "status": "draft",
                "links": ["name:Invalid-Name"],  # hyphens not allowed
            }
        )


def test_links_empty_internal_reference():
    """Test that empty name after name: prefix is rejected."""
    with pytest.raises(ValueError, match="name cannot be empty"):
        create_standard_name_entry(
            {
                "kind": "scalar",
                "physics_domain": "general",
                "name": "test_quantity",
                "description": "Test",
                "unit": "m",
                "status": "draft",
                "links": ["name:"],  # empty name
            }
        )


def test_links_invalid_format():
    """Test that links without proper format are rejected."""
    with pytest.raises(ValueError, match="must be either an external URL"):
        create_standard_name_entry(
            {
                "kind": "scalar",
                "physics_domain": "general",
                "name": "test_quantity",
                "description": "Test",
                "unit": "m",
                "status": "draft",
                "links": ["just_a_string"],  # neither URL nor name: reference
            }
        )
