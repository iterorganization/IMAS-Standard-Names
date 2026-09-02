"""Tests for the single pint-defined canonical unit authority.

``canonical_unit`` is the one place unit ordering and symbol spelling are
decided: both the SN-side stored unit and any DD-side comparison string flow
through it so that ordering-only and spelling-only differences collapse to
string equality. These tests pin its output form (so a pint version bump or a
symbol-spelling change fails CI rather than silently drifting the catalog's
canonical form) and prove the reconciliation property.
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import pytest

from imas_standard_names import canonical_unit
from imas_standard_names.models import (
    StandardNameEntryBase,
    create_standard_name_entry,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Canonical dot-exponent grammar: ASCII short symbols (each starting with a
# letter so a stray float-exponent digit-token like the "0" in "second^-1.0"
# cannot masquerade as a unit), lexicographically sorted, dot-separated, with
# signed integer exponents. "1" is the dimensionless sentinel.
_TOKEN = r"[A-Za-z][A-Za-z0-9]*(?:\^-?[1-9]\d*)?"
_DOTEXP_GRAMMAR = re.compile(rf"^(?:1|{_TOKEN}(?:\.{_TOKEN})*)$")


def _catalog_units() -> set[str]:
    """Every distinct authored ``unit`` string in the shipped catalog."""
    units: set[str] = set()

    def walk(obj):
        if isinstance(obj, dict):
            u = obj.get("unit")
            if isinstance(u, str):
                units.add(u)
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)

    import yaml

    pattern = str(_REPO_ROOT / "imas_standard_names" / "resources" / "**" / "*.yml")
    for path in glob.glob(pattern, recursive=True):
        try:
            walk(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
        except yaml.YAMLError:
            continue
    return units


CATALOG_UNITS = sorted(_catalog_units())


class TestCanonicalUnitForm:
    def test_dimensionless_sentinel(self):
        assert canonical_unit("1") == "1"

    @pytest.mark.parametrize(
        "authored,expected",
        [
            ("s^-2.m", "m.s^-2"),
            ("m.s^-1", "m.s^-1"),
            ("keV.m^-1", "keV.m^-1"),
            ("T.m^-2.A", "A.T.m^-2"),
            ("s.kg.m^2", "kg.m^2.s"),
        ],
    )
    def test_ordering_is_lexicographic_short_symbol(self, authored, expected):
        assert canonical_unit(authored) == expected

    @pytest.mark.parametrize(
        "authored,expected",
        [
            ("m.ohm", "m.ohm"),  # pint short glyph Ω -> ASCII ohm
            ("us", "us"),  # micro prefix µ -> ASCII u
            ("uohm.m", "m.uohm"),
            ("degC", "degC"),  # degree glyph ° -> ASCII deg
        ],
    )
    def test_symbols_are_ascii(self, authored, expected):
        result = canonical_unit(authored)
        assert result == expected
        assert result.isascii(), f"{result!r} is not ASCII"

    def test_float_exponent_artifact_is_gone(self):
        # The retired "U" formatter rendered a reordered s^-1.m as
        # 'second^-1.0'; the canonical form must carry an integer exponent.
        assert canonical_unit("s^-1.m") == "m.s^-1"
        assert ".0" not in canonical_unit("s^-1.m")

    def test_invalid_unit_raises(self):
        with pytest.raises(ValueError):
            canonical_unit("notaunit")

    @pytest.mark.parametrize("authored", CATALOG_UNITS)
    def test_idempotent_over_catalog(self, authored):
        once = canonical_unit(authored)
        assert canonical_unit(once) == once


class TestCanonicalUnitStability:
    """Guard against pint drift: canonical output must always match the
    dot-exponent grammar over the full catalog unit set."""

    @pytest.mark.parametrize("authored", CATALOG_UNITS)
    def test_output_matches_dotexp_grammar(self, authored):
        result = canonical_unit(authored)
        assert result.isascii(), f"{result!r} carries a non-ASCII glyph"
        assert " " not in result and "/" not in result and "*" not in result
        assert ".0" not in result, f"{result!r} has a float-exponent artifact"
        assert _DOTEXP_GRAMMAR.match(result), (
            f"{result!r} violates the canonical dot-exponent grammar"
        )
        # tokens are lexicographically sorted
        if result != "1":
            symbols = [tok.split("^")[0] for tok in result.split(".")]
            assert symbols == sorted(symbols), f"{result!r} tokens are not sorted"

    def test_catalog_is_non_trivial(self):
        # Fail loudly if the catalog scan silently found nothing to guard.
        assert len(CATALOG_UNITS) >= 5


class TestReconciliation:
    """Ordering/spelling variants of one physical unit normalize equal — the
    property the downstream SN<->DD mismatch axis relies on."""

    @pytest.mark.parametrize(
        "left,right",
        [
            ("s^-1.m", "m.s^-1"),
            ("A.T.m^-2", "m^-2.A.T"),
            ("kg.m^2.s", "s.m^2.kg"),
            ("m.ohm", "ohm.m"),
        ],
    )
    def test_reordered_variants_are_equal(self, left, right):
        assert canonical_unit(left) == canonical_unit(right)

    def test_validator_uses_canonical_unit(self):
        # The scalar unit validator must produce exactly canonical_unit()'s
        # form — one ordering authority, no second implementation.
        entry = create_standard_name_entry(
            {
                "kind": "scalar",
                "physics_domain": "core_plasma_physics",
                "name": "acceleration_probe",
                "description": "probe",
                "documentation": "Canonical-order probe entry.",
                "unit": "s^-2.m",
                "status": "draft",
            }
        )
        assert entry.unit == canonical_unit("s^-2.m") == "m.s^-2"

    def test_canonicalize_helper_delegates_to_canonical_unit(self):
        for authored in ("s^-2.m", "T.m^-2.A", "m.ohm"):
            assert StandardNameEntryBase._canonicalize_unit_order(
                authored
            ) == canonical_unit(authored)
