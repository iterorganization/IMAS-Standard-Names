"""Tests for the referential-integrity semantic check.

Covers structured references (links, deprecates, superseded_by, arguments,
error_variants) and inline markdown "[label](name:target)" references in
documentation prose, resolved against the set of entries passed to
``run_semantic_checks``.
"""

from __future__ import annotations

from imas_standard_names.models import create_standard_name_entry
from imas_standard_names.validation.semantic import (
    INLINE_REFERENCE_SEVERITY,
    run_semantic_checks,
)


def _entry(name: str, **overrides):
    base = {
        "name": name,
        "kind": "scalar",
        "physics_domain": "general",
        "unit": "eV",
        "description": f"Description of {name}.",
        "documentation": f"Documentation of {name}.",
    }
    base.update(overrides)
    return create_standard_name_entry(base)


def test_links_reference_resolves():
    temperature = _entry("temperature")
    dependent = _entry("gradient_of_temperature", links=["name:temperature"])
    entries = {e.name: e for e in (temperature, dependent)}
    issues = run_semantic_checks(entries)
    assert not [i for i in issues if "gradient_of_temperature" in i]


def test_links_reference_dangling_is_error():
    entry = _entry("gradient_of_temperature", links=["name:does_not_exist"])
    issues = run_semantic_checks({entry.name: entry})
    matches = [i for i in issues if "does_not_exist" in i]
    assert len(matches) == 1
    assert "ERROR" in matches[0]


def test_links_external_url_is_not_checked():
    entry = _entry("temperature", links=["https://example.org/ref"])
    issues = run_semantic_checks({entry.name: entry})
    assert not [i for i in issues if "temperature" in i]


def test_superseded_by_resolves_to_active_successor():
    """A deprecated stub pointing forward to an active name must not be
    flagged — this is the intended shape of the (concurrently landing)
    deprecation architecture, not a defect."""
    successor = _entry("temperature")
    stub = _entry(
        "electron_temperature_legacy",
        status="deprecated",
        superseded_by="temperature",
    )
    entries = {e.name: e for e in (successor, stub)}
    issues = run_semantic_checks(entries)
    assert not [i for i in issues if "electron_temperature_legacy" in i]


def test_superseded_by_dangling_is_error():
    stub = _entry(
        "electron_temperature_legacy",
        status="deprecated",
        superseded_by="ghost_name",
    )
    issues = run_semantic_checks({stub.name: stub})
    matches = [i for i in issues if "ghost_name" in i]
    assert len(matches) == 1
    assert "ERROR" in matches[0]
    assert "superseded_by" in matches[0]


def test_arguments_dangling_reference_is_error():
    entry = _entry(
        "maximum_of_temperature",
        arguments=[
            {
                "name": "ghost_base",
                "operator": "maximum",
                "operator_kind": "unary_prefix",
            }
        ],
    )
    issues = run_semantic_checks({entry.name: entry})
    matches = [i for i in issues if "ghost_base" in i]
    assert len(matches) == 1
    assert "WARNING" in matches[0]
    assert "arguments" in matches[0]


def test_arguments_resolvable_reference_is_clean():
    base = _entry("temperature")
    entry = _entry(
        "maximum_of_temperature",
        arguments=[
            {
                "name": "temperature",
                "operator": "maximum",
                "operator_kind": "unary_prefix",
            }
        ],
    )
    entries = {e.name: e for e in (base, entry)}
    issues = run_semantic_checks(entries)
    assert not [i for i in issues if "maximum_of_temperature" in i]


def test_error_variants_dangling_reference_is_error():
    entry = _entry(
        "temperature",
        error_variants={"upper": "upper_uncertainty_of_temperature"},
    )
    issues = run_semantic_checks({entry.name: entry})
    matches = [i for i in issues if "upper_uncertainty_of_temperature" in i]
    assert len(matches) == 1
    assert "WARNING" in matches[0]
    assert "error_variants" in matches[0]


def test_error_variants_resolvable_reference_is_clean():
    upper = _entry("upper_uncertainty_of_temperature")
    entry = _entry(
        "temperature",
        error_variants={"upper": "upper_uncertainty_of_temperature"},
    )
    entries = {e.name: e for e in (upper, entry)}
    issues = run_semantic_checks(entries)
    assert not [i for i in issues if i.startswith("temperature:")]


def test_inline_documentation_reference_dangling_is_warning():
    entry = _entry(
        "current_at_divertor_target",
        documentation=(
            "Current at a divertor target, complementing the parent "
            "[current of divertor](name:ghost_current_of_divertor)."
        ),
    )
    issues = run_semantic_checks({entry.name: entry})
    matches = [i for i in issues if "ghost_current_of_divertor" in i]
    assert len(matches) == 1
    assert INLINE_REFERENCE_SEVERITY in matches[0]
    # Severity knob: inline refs must not block a normal (non-strict) run.
    assert ": ERROR" not in matches[0]


def test_inline_documentation_reference_resolves():
    parent = _entry("current_of_divertor")
    entry = _entry(
        "current_at_divertor_target",
        documentation=(
            "Current at a divertor target, complementing the parent "
            "[current of divertor](name:current_of_divertor)."
        ),
    )
    entries = {e.name: e for e in (parent, entry)}
    issues = run_semantic_checks(entries)
    assert not [i for i in issues if "current_at_divertor_target" in i]
