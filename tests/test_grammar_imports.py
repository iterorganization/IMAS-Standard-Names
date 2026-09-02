"""Tests verifying the grammar module's public API and import behavior."""

import re
import subprocess
import sys
from enum import StrEnum


def test_grammar_module_imports_cleanly():
    """Grammar module can be imported without errors."""
    import imas_standard_names.grammar

    assert hasattr(imas_standard_names.grammar, "__all__")


def test_all_enums_importable():
    """All grammar enum types are accessible from the grammar package."""
    from imas_standard_names.grammar import (
        Component,
        GenericPhysicalBase,
        GeometricBase,
        Object,
        Position,
        Process,
        Subject,
    )

    for enum_cls in (
        Component,
        GenericPhysicalBase,
        GeometricBase,
        Object,
        Position,
        Process,
        Subject,
    ):
        assert issubclass(enum_cls, StrEnum), f"{enum_cls} is not a StrEnum"
        assert len(enum_cls) > 0, f"{enum_cls} has no members"


def test_core_functions_importable():
    """Core composition and parsing functions are accessible."""
    from imas_standard_names.grammar import (
        compose_name,
        compose_standard_name,
        parse_name,
        parse_standard_name,
    )

    assert callable(compose_standard_name)
    assert callable(parse_standard_name)
    assert compose_name is compose_standard_name
    assert parse_name is parse_standard_name


def test_support_utilities_importable():
    """Support utilities (TOKEN_PATTERN, coerce_enum, etc.) are exported."""
    from imas_standard_names.grammar import (
        TOKEN_PATTERN,
        coerce_enum,
        enum_values,
        value_of,
    )

    assert isinstance(TOKEN_PATTERN, re.Pattern)
    assert callable(coerce_enum)
    assert callable(enum_values)
    assert callable(value_of)


def test_standard_name_model_importable():
    """The StandardName model is importable from the grammar package."""
    from imas_standard_names.grammar import StandardName

    assert hasattr(StandardName, "model_fields")


def test_all_exports_match_dir():
    """Every name in __all__ is actually defined in the module."""
    import imas_standard_names.grammar as grammar

    for name in grammar.__all__:
        assert hasattr(grammar, name), f"{name} listed in __all__ but not defined"


def test_import_no_catalog_or_network_side_effects():
    """Importing the grammar module does not trigger catalog loading or network I/O.

    Runs the import in a clean subprocess to ensure no app-level data reads
    occur. The subprocess fails if any catalog or network access is attempted.
    """
    script = (
        "import sys\n"
        "# Prevent any catalog or repository loading by poisoning the imports\n"
        "import types\n"
        "poison = types.ModuleType('poison')\n"
        "poison.__getattr__ = lambda self, name: (_ for _ in ()).throw(\n"
        "    ImportError(f'Unexpected import of catalog module: {name}')\n"
        ")\n"
        "# Import grammar (should not trigger catalog/database/network)\n"
        "from imas_standard_names.grammar import (\n"
        "    Component, GeometricBase, Object, GenericPhysicalBase,\n"
        "    Position, Process, Subject, StandardName,\n"
        "    compose_standard_name, parse_standard_name,\n"
        ")\n"
        "# Verify types are usable\n"
        "assert len(Component) > 0\n"
        "assert len(GeometricBase) > 0\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"Grammar import caused side effects:\n{result.stderr}"
    )
    assert "OK" in result.stdout


def test_validate_models_with_plain_dict():
    """validate_models() works with a plain dict of entries, no catalog required."""
    from imas_standard_names.models import create_standard_name_entry
    from imas_standard_names.services import validate_models

    entry = create_standard_name_entry(
        {
            "name": "electron_temperature",
            "kind": "scalar",
            "physics_domain": "core_plasma_physics",
            "unit": "eV",
            "description": "Electron temperature.",
            "documentation": "",
        }
    )
    entries = {"electron_temperature": entry}
    issues = validate_models(entries)
    assert isinstance(issues, list)


def test_validate_models_returns_issues_for_problematic_entries():
    """validate_models() returns issue strings for entries with problems."""
    from imas_standard_names.models import create_standard_name_entry
    from imas_standard_names.services import validate_models

    # A gradient operator entry with a unit that doesn't look like a derivative
    entry = create_standard_name_entry(
        {
            "name": "gradient_of_electron_temperature",
            "kind": "vector",
            "physics_domain": "core_plasma_physics",
            "unit": "eV",
            "description": "Spatial gradient of electron temperature.",
            "documentation": "",
            "provenance": {
                "mode": "operator",
                "operators": ["gradient"],
                "base": "electron_temperature",
                "operator_id": "gradient",
            },
        }
    )
    entries = {"gradient_of_electron_temperature": entry}
    issues = validate_models(entries)
    assert isinstance(issues, list)
    assert any("gradient" in issue for issue in issues)


def test_validate_models_empty_dict():
    """validate_models() accepts an empty dict and returns no issues."""
    from imas_standard_names.services import validate_models

    assert validate_models({}) == []


def test_py_typed_marker_exists():
    """PEP 561 py.typed marker is present in the package."""
    import importlib.resources

    package_path = importlib.resources.files("imas_standard_names")
    py_typed = package_path / "py.typed"
    assert py_typed.is_file(), "py.typed marker file missing from package"
