"""Tests for the source/flux near-name unit-collision semantic check.

A ``*_flux`` or ``*_source`` name asserts a dimensional derivative of the
bare quantity, so a pair differing only by that token must not share a
unit. The check catches the defect class where bare momentum entries
(actually momentum source densities) collided dimensionally with the
``*_momentum_flux`` family.
"""

from __future__ import annotations

from imas_standard_names.models import create_standard_name_entry
from imas_standard_names.validation.semantic import run_semantic_checks


def _entry(name: str, unit: str = "Pa", **overrides):
    base = {
        "name": name,
        "kind": "scalar",
        "physics_domain": "general",
        "unit": unit,
        "description": f"Description of {name}.",
        "documentation": f"Documentation of {name}.",
    }
    base.update(overrides)
    return create_standard_name_entry(base)


def _collision_issues(issues):
    return [i for i in issues if "differ only" in i]


def test_flux_pair_sharing_unit_is_flagged():
    bare = _entry("electron_momentum", unit="kg.m^-2.s^-1")
    flux = _entry("electron_momentum_flux", unit="kg.m^-2.s^-1")
    entries = {e.name: e for e in (bare, flux)}
    matches = _collision_issues(run_semantic_checks(entries))
    assert len(matches) == 1
    assert "electron_momentum_flux" in matches[0]
    assert "WARNING" in matches[0]


def test_source_pair_sharing_unit_is_flagged():
    bare = _entry("electron_energy", unit="J.m^-3")
    source = _entry("electron_energy_source", unit="J.m^-3")
    entries = {e.name: e for e in (bare, source)}
    matches = _collision_issues(run_semantic_checks(entries))
    assert len(matches) == 1
    assert "electron_energy_source" in matches[0]


def test_flux_token_mid_name_is_matched():
    bare = _entry("ion_momentum_at_wall", unit="Pa")
    flux = _entry("ion_momentum_flux_at_wall", unit="Pa")
    entries = {e.name: e for e in (bare, flux)}
    matches = _collision_issues(run_semantic_checks(entries))
    assert len(matches) == 1
    assert "ion_momentum_flux_at_wall" in matches[0]


def test_pair_with_distinct_units_is_clean():
    bare = _entry("electron_momentum", unit="kg.m.s^-1")
    flux = _entry("electron_momentum_flux", unit="Pa")
    entries = {e.name: e for e in (bare, flux)}
    assert not _collision_issues(run_semantic_checks(entries))


def test_flux_without_bare_sibling_is_clean():
    flux = _entry("electron_momentum_flux", unit="Pa")
    assert not _collision_issues(run_semantic_checks({flux.name: flux}))


def test_unitless_entries_are_ignored():
    entries = {}
    for name in ("gas_valve_state", "gas_valve_state_flux"):
        entry = create_standard_name_entry(
            {
                "name": name,
                "kind": "metadata",
                "physics_domain": "general",
                "description": f"Description of {name}.",
                "documentation": f"Documentation of {name}.",
            }
        )
        entries[entry.name] = entry
    assert not _collision_issues(run_semantic_checks(entries))
