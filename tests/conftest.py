"""Shared pytest fixtures for IMAS Standard Names tests."""

import importlib.resources as ir
from pathlib import Path

import pytest
import yaml

from imas_standard_names.models import create_standard_name_entry
from imas_standard_names.repository import StandardNameCatalog


def _write_entry_yaml(root: Path, entry):
    """Write a standard name entry as a YAML file to disk.

    Uses the per-domain list format: entries are appended
    to ``root/<domain>.yml`` as a YAML sequence.
    """
    domain = getattr(entry, "physics_domain", "general") or "general"
    domain_file = root / f"{domain}.yml"
    data = {
        k: v
        for k, v in entry.model_dump(mode="json").items()
        if v not in (None, [], "")
    }
    data["name"] = entry.name
    data["physics_domain"] = str(domain)

    # Load existing entries if the domain file already exists
    existing: list[dict] = []
    if domain_file.exists():
        with open(domain_file, encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
        if isinstance(loaded, list):
            existing = loaded

    existing.append(data)
    with open(domain_file, "w", encoding="utf-8") as fh:
        yaml.safe_dump(existing, fh, sort_keys=False, allow_unicode=True, width=80)


@pytest.fixture
def write_yaml():
    """Fixture providing a helper to write standard name entries as YAML files."""
    return _write_entry_yaml


# Helper functions for creating valid test entries
def make_valid_entry(kind="scalar", **overrides):
    """Create a valid standard name entry with all required fields.

    Args:
        kind: Entry kind (scalar, vector, metadata)
        **overrides: Override default values for any field

    Returns:
        StandardNameEntry with all required fields populated
    """
    defaults = {
        "kind": kind,
        "name": "test_quantity",
        "physics_domain": "general",
        "description": "Test quantity for unit tests.",
        "documentation": "Detailed documentation for test quantity.",
        "unit": "m^-3" if kind != "metadata" else "",
        "status": "draft",
    }

    # Merge with overrides
    data = {**defaults, **overrides}

    # Remove unit for metadata if not explicitly set
    if kind == "metadata" and "unit" not in overrides:
        data.pop("unit", None)

    return create_standard_name_entry(data)


@pytest.fixture
def make_entry():
    """Fixture providing the make_valid_entry helper function."""
    return make_valid_entry


# Configure pytest-anyio to only use asyncio backend (trio not installed)
# This prevents pytest from parameterizing tests with trio backend
@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture
def temp_catalog_dir(tmp_path):
    """Create a temporary directory for catalog YAML files."""
    catalog_dir = tmp_path / "catalog"
    catalog_dir.mkdir()
    return catalog_dir


@pytest.fixture
def temp_catalog(temp_catalog_dir):
    """Create a StandardNameCatalog using temporary storage.

    This catalog starts empty and is isolated from the production catalog.
    All changes are scoped to the test and cleaned up automatically.
    """
    catalog = StandardNameCatalog(root=temp_catalog_dir)
    return catalog


@pytest.fixture
def sample_scalar_entry():
    """Sample scalar catalog entry for testing."""
    return {
        "name": "test_scalar",
        "kind": "scalar",
        "physics_domain": "general",
        "description": "Test scalar entry.",
        "unit": "m",
        "status": "draft",
    }


@pytest.fixture
def sample_vector_entry():
    """Sample vector catalog entry for testing."""
    return {
        "name": "test_vector",
        "kind": "vector",
        "physics_domain": "general",
        "description": "Test vector entry.",
        "unit": "T",
        "status": "draft",
    }


@pytest.fixture
def sample_entries_with_provenance():
    """Sample entries with various provenance patterns."""
    return [
        {
            "name": "base_quantity",
            "kind": "scalar",
            "physics_domain": "general",
            "description": "Base quantity for provenance tests.",
            "unit": "m",
        },
        {
            "name": "derived_quantity",
            "kind": "scalar",
            "physics_domain": "general",
            "description": "Derived quantity using operator.",
            "unit": "m.s^-1",
            "provenance": {
                "mode": "operator",
                "operators": ["time_derivative"],
                "base": "base_quantity",
                "operator_id": "time_derivative",
            },
        },
    ]


@pytest.fixture(scope="session")
def examples_catalog():
    """Load examples catalog using importlib.resources."""
    files_obj = ir.files("imas_standard_names") / "resources" / "standard_name_examples"
    with ir.as_file(files_obj) as examples_path:
        return StandardNameCatalog(root=examples_path, permissive=True)


@pytest.fixture
def example_scalars(examples_catalog):
    """Get scalar examples auto-discovered from catalog."""
    return examples_catalog.list(kind="scalar")[:3]


@pytest.fixture
def example_vectors(examples_catalog):
    """Get vector examples auto-discovered from catalog."""
    return examples_catalog.list(kind="vector")[:3]


@pytest.fixture
def example_metadata(examples_catalog):
    """Get metadata examples auto-discovered from catalog."""
    return examples_catalog.list(kind="metadata")[:2]


@pytest.fixture
def copy_examples(examples_catalog):
    """Return function to copy examples to temp directory."""

    def _copy(target_dir: Path, count: int = 5, kind: str | None = None):
        examples = examples_catalog.list(kind=kind)[:count]
        for entry in examples:
            _write_entry_yaml(target_dir, entry)
        return examples

    return _copy


@pytest.fixture
def sample_catalog(tmp_path, examples_catalog):
    """Create a StandardNameCatalog pre-populated with sample entries.

    This fixture provides a catalog with several standard entries for tests
    that need existing data (e.g., testing search, list operations).
    Uses real examples from the examples catalog for comprehensive testing.
    Isolated from production catalog and cleaned up automatically.
    """
    catalog_dir = tmp_path / "sample_catalog"
    catalog_dir.mkdir()

    # Copy examples from the examples catalog
    examples = examples_catalog.list()[:10]  # Get first 10 examples for variety
    for entry in examples:
        _write_entry_yaml(catalog_dir, entry)

    # Return initialized catalog
    return StandardNameCatalog(root=catalog_dir)
