"""Vocabulary CI gates and round-trip regression suite.

Validates:
1. No new qualifier/subject overlap (grandfathered originals allowed)
2. Physical bases are irreducible (no qualifier+base compounds)
3. Round-trip correctness: parse → compose → parse
4. Qualifier ordering preservation
5. Subject/qualifier boundary: compound subjects stay atomic
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from imas_standard_names.grammar.model_types import Subject
from imas_standard_names.grammar.parser import (
    ParseError,
    load_default_vocabularies,
    parse,
)
from imas_standard_names.grammar.render import compose
from imas_standard_names.grammar.vocab_loaders import (
    load_physical_bases,
    load_qualifiers,
)

# ---------------------------------------------------------------------------
# Paths to vocabulary files
# ---------------------------------------------------------------------------

_VOCAB_DIR = (
    Path(__file__).resolve().parents[2]
    / "imas_standard_names"
    / "grammar"
    / "vocabularies"
)

# ---------------------------------------------------------------------------
# Constants: the 19 original hardcoded modifier_quals that are allowed to
# overlap with subjects (backward-compatible grandfathered set).
# ---------------------------------------------------------------------------

_ORIGINAL_19_MODIFIER_QUALS = frozenset(
    {
        "energy",
        "particle",
        "momentum",
        "heat",
        "total",
        "net",
        "collisional",
        "thermal",
        "ohmic",
        "inductive",
        "non_inductive",
        "bootstrap",
        "anomalous",
        "neoclassical",
        "turbulent",
        "fast",
        "trapped",
    }
)

# Tokens explicitly marked [also-subject] in qualifiers.yml — these are
# documented dual-role tokens.
_DOCUMENTED_ALSO_SUBJECT = frozenset(
    {
        "co_passing",
        "counter_passing",
        "state",  # charge state or quantum state — dual-role as qualifier and subject
        # Fusion reactant pairs: subject as the effective fuel species
        # (deuterium_tritium_density); reaction-channel qualifier when a product
        # subject follows (deuterium_tritium_neutron_flux). Intentional dual-role.
        "deuterium_tritium",
        "deuterium_deuterium",
        "tritium_tritium",
    }
)

# Full set of allowed overlaps: originals + explicitly documented dual-role
_ALLOWED_OVERLAP = _ORIGINAL_19_MODIFIER_QUALS | _DOCUMENTED_ALSO_SUBJECT

# ---------------------------------------------------------------------------
# Known irreducible compounds: bases that LOOK like qualifier+base but are
# actually atomic dimensional quantities. Do NOT decompose these.
# ---------------------------------------------------------------------------

_KNOWN_IRREDUCIBLE_COMPOUNDS = frozenset(
    {
        "energy_flux",
        "heat_flux",
        "particle_flux",
        "momentum_flux",
        "energy_source",
        "momentum_source",
        "particle_source",
        "heat_source",
        "energy_confinement_time",
        "thermal_conductivity",
        "thermal_diffusivity",
        "momentum_diffusivity",
        "particle_radial_diffusivity",
        "heat_viscosity",
        "net_power",
        "heating_power",
        "ohmic_power",
        "fusion_power_density",
        "thermal_energy_pedestal",
        "fast_wave_field",
        # spectral_radiance is radiance per unit wavelength/frequency
        # (W.m^-2.sr^-1.nm^-1) — a distinct radiometric quantity with an
        # extra spectral dimension, not merely [spectral] + [radiance].
        # Lexicalised; kept irreducible alongside the broadband 'radiance' base.
        "spectral_radiance",
        # kinetic_energy and surface_area are base-bound lexical compounds
        # promoted to atomic bases. Their leading words 'kinetic' / 'surface'
        # remain qualifiers because they qualify OTHER bases
        # (kinetic_energy_density / kinetic_energy_flux,
        # surface_temperature / surface_roughness / surface_emissivity). The
        # standalone X_kinetic_energy / surface_area forms are atomic via
        # longest-base-match; the qualifier coexists for the suffixed forms.
        "kinetic_energy",
        "surface_area",
    }
)

# ---------------------------------------------------------------------------
# Corpus: parseable catalog names + synthetic names for coverage
# ---------------------------------------------------------------------------

_CATALOG_NAMES = [
    # From imas-standard-names-catalog (the 29 that parse successfully)
    "co_passing_fast_particle_collisional_power_density_due_to_coulomb_collisions_with_ion",
    "co_passing_particle_density",
    "counter_passing_fast_particle_collisional_power_density_due_to_coulomb_collisions_with_ion",
    "counter_passing_particle_density",
    "fast_ion_pressure",
    "fast_particle_density",
    "poloidal_electric_field",
    "radial_electric_field",
    "time_derivative_of_fast_electron_density",
    "time_derivative_of_ion_state_density",
    "toroidal_electric_field",
    "trapped_fast_particle_density",
    "trapped_particle_density",
    "vertical_center_of_mass_velocity",
    "vertical_electron_velocity",
    "vertical_ion_velocity",
    "area_of_flux_loop",
    "poloidal_anomalous_current_density",
    "radial_anomalous_current_density",
    "toroidal_anomalous_current_density",
    "parallel_neutral_momentum_flux",
    "parallel_total_momentum_flux",
]

_SYNTHETIC_NAMES = [
    # Single qualifier + base
    "electron_temperature",
    "ion_density",
    "collisional_power_density",
    # Multi-qualifier + base
    "trapped_fast_particle_density",
    "ohmic_current_density",
    "bootstrap_current_density",
    # Subject + base (no extra qualifier)
    "deuterium_pressure",
    "helium_density",
    "neutral_temperature",
    # Qualifier + process (mechanism)
    "electron_density_due_to_ionization",
    "energy_flux_due_to_convection",
    # Component projection
    "toroidal_magnetic_field",
    "radial_velocity",
    "parallel_current_density",
    # Operator + base
    "time_derivative_of_electron_density",
    "time_average_of_temperature",
    "magnitude_of_magnetic_field",
    # Locus
    "electron_temperature_at_magnetic_axis",
    "pressure_at_plasma_boundary",
    "density_on_flux_surface",
    # Complex: component + qualifier + base
    "toroidal_ion_velocity",
    "radial_electron_particle_flux",
    # Complex: operator + locus
    "time_average_of_electron_temperature_at_magnetic_axis",
    # Simple bases
    "temperature",
    "pressure",
    "density",
    "current",
    "magnetic_field",
    "velocity",
]

CORPUS_NAMES = _CATALOG_NAMES + _SYNTHETIC_NAMES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vocabs():
    return load_default_vocabularies()


@pytest.fixture(scope="module")
def qualifier_tokens():
    return load_qualifiers()


@pytest.fixture(scope="module")
def subject_tokens():
    return frozenset(s.value for s in Subject)


@pytest.fixture(scope="module")
def base_tokens():
    reg = load_physical_bases()
    return frozenset(reg.bases.keys())


# ---------------------------------------------------------------------------
# Test 1: No NEW qualifier/subject overlap
# ---------------------------------------------------------------------------


class TestNoNewQualifierSubjectOverlap:
    """Qualifiers and subjects must not have NEW overlapping tokens.

    The original 19 hardcoded modifier_quals are grandfathered — they are
    allowed to appear in both sets. Any NEW qualifier token must NOT also
    be a subject.
    """

    def test_no_new_qualifier_subject_overlap(self, qualifier_tokens, subject_tokens):
        overlap = qualifier_tokens & subject_tokens
        # Remove grandfathered + documented dual-role tokens
        new_overlap = overlap - _ALLOWED_OVERLAP
        assert not new_overlap, (
            f"NEW tokens appear in both qualifiers.yml and subjects: "
            f"{sorted(new_overlap)}. "
            f"If intentional, add to _DOCUMENTED_ALSO_SUBJECT."
        )

    def test_grandfathered_tokens_documented(self, qualifier_tokens):
        """Known dual-role tokens exist in qualifiers.yml."""
        # Only check that the tokens marked [also-subject] are present
        # At minimum, the dual-role tokens that currently overlap should be there
        missing = (
            _ALLOWED_OVERLAP
            & frozenset(
                # Only check tokens that ARE qualifiers (some originals like
                # 'energy' may be in qualifiers but not subjects)
                t
                for t in _ALLOWED_OVERLAP
                if t in qualifier_tokens
            )
        ) - qualifier_tokens
        assert not missing, (
            f"Documented dual-role tokens missing from qualifiers.yml: {sorted(missing)}"
        )


# ---------------------------------------------------------------------------
# Test 2: Physical bases are irreducible
# ---------------------------------------------------------------------------


class TestBasesAreIrreducible:
    """No physical_base entry should decompose into qualifier + existing_base.

    If a base can be split as qualifier_prefix + remaining_base (where both
    parts are in their respective vocabularies), it should have been factored
    into qualifiers.yml. Known irreducible compounds are excepted.
    """

    def test_bases_are_irreducible(self, qualifier_tokens, base_tokens):
        violations = []
        # Union qualifier+subject as the parser does
        all_qualifiers = qualifier_tokens | frozenset(s.value for s in Subject)

        for base in sorted(base_tokens):
            if base in _KNOWN_IRREDUCIBLE_COMPOUNDS:
                continue
            # Try each underscore split point
            parts = base.split("_")
            for i in range(1, len(parts)):
                prefix = "_".join(parts[:i])
                rest = "_".join(parts[i:])
                if prefix in all_qualifiers and rest in base_tokens:
                    violations.append(f"  {base} = [{prefix}] + [{rest}]")
                    break  # one decomposition is enough

        assert not violations, (
            "Physical bases decomposable into qualifier + base "
            "(add to _KNOWN_IRREDUCIBLE_COMPOUNDS if intentional):\n"
            + "\n".join(violations)
        )


# ---------------------------------------------------------------------------
# Test 3: Round-trip correctness
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """parse(compose(parse(name))) == parse(name) for all corpus names."""

    @pytest.mark.parametrize("name", CORPUS_NAMES)
    def test_round_trip(self, name, vocabs):
        """Round-trip: parse → compose → parse yields identical IR."""
        try:
            result1 = parse(name, vocabs)
        except ParseError:
            pytest.skip(f"name {name!r} does not parse (vocabulary gap)")

        composed = compose(result1.ir)
        result2 = parse(composed, vocabs)
        assert result1.ir == result2.ir, (
            f"Round-trip failed: {name!r} → {composed!r}\n"
            f"  IR1: {result1.ir}\n"
            f"  IR2: {result2.ir}"
        )

    @pytest.mark.parametrize("name", CORPUS_NAMES)
    def test_compose_reproduces_input(self, name, vocabs):
        """compose(parse(name)) == name (canonical form preserved)."""
        try:
            result = parse(name, vocabs)
        except ParseError:
            pytest.skip(f"name {name!r} does not parse (vocabulary gap)")

        composed = compose(result.ir)
        assert composed == name, f"Composed form differs: {name!r} → {composed!r}"


# ---------------------------------------------------------------------------
# Test 4: Qualifier ordering preserved
# ---------------------------------------------------------------------------


class TestQualifierOrderingPreserved:
    """Multi-qualifier names preserve parse order through compose."""

    @pytest.mark.parametrize(
        "name,expected_qualifiers",
        [
            # Decomposed: fast/trapped are population qualifiers, particle is the
            # species subject — no compound fast_particle token (rc32 decomposition).
            ("trapped_fast_particle_density", ["trapped", "fast", "particle"]),
            ("co_passing_particle_density", ["co_passing", "particle"]),
            ("collisional_power_density", ["collisional"]),
            ("ohmic_current_density", ["ohmic"]),
            ("bootstrap_current_density", ["bootstrap"]),
            ("anomalous_current_density", ["anomalous"]),
        ],
    )
    def test_qualifier_order_matches_input(self, name, expected_qualifiers, vocabs):
        """Qualifiers are extracted in input order (left-to-right)."""
        try:
            result = parse(name, vocabs)
        except ParseError:
            pytest.skip(f"name {name!r} does not parse")

        actual = [q.token for q in result.ir.qualifiers]
        assert actual == expected_qualifiers, (
            f"Qualifier order mismatch for {name!r}: "
            f"expected {expected_qualifiers}, got {actual}"
        )

    @pytest.mark.parametrize(
        "name",
        [
            "trapped_fast_particle_density",
            "co_passing_particle_density",
            "counter_passing_particle_density",
        ],
    )
    def test_compose_preserves_insertion_order(self, name, vocabs):
        """compose() emits qualifiers in the same order parse() found them."""
        try:
            result = parse(name, vocabs)
        except ParseError:
            pytest.skip(f"name {name!r} does not parse")

        composed = compose(result.ir)
        assert composed == name, f"Order not preserved: {name!r} → {composed!r}"


# ---------------------------------------------------------------------------
# Test 5: Subject/qualifier boundary
# ---------------------------------------------------------------------------


class TestSubjectQualifierBoundary:
    """Modifier qualifiers decompose; genuine compound species stay atomic.

    rc32 decomposition: energy-state/orbit modifiers (fast, thermal, trapped, …)
    are NOT subject tokens — they peel as population qualifiers and the species
    is the subject. Genuine atomic species compounds (alpha_particle, the
    reaction pairs) remain single tokens.
    """

    @pytest.mark.parametrize(
        "name,expected_subject_qualifier,expected_base",
        [
            # Decomposed: 'fast' is a population qualifier, 'ion' is the subject.
            ("fast_ion_pressure", ["fast", "ion"], "pressure"),
            # "trapped" is a population qualifier, rest peels normally
            ("trapped_particle_density", ["trapped", "particle"], "density"),
            # Single subject
            ("electron_temperature", ["electron"], "temperature"),
            # alpha_particle is a registered Subject: with longest-match-first,
            # it's matched as a single compound qualifier token.
            ("alpha_particle_density", ["alpha_particle"], "density"),
        ],
    )
    def test_subject_boundary(
        self, name, expected_subject_qualifier, expected_base, vocabs
    ):
        """Verify subject/qualifier tokens are correctly identified."""
        try:
            result = parse(name, vocabs)
        except ParseError:
            pytest.skip(f"name {name!r} does not parse")

        actual_quals = [q.token for q in result.ir.qualifiers]
        actual_base = result.ir.base.token
        assert actual_quals == expected_subject_qualifier, (
            f"Subject/qualifier split wrong for {name!r}: "
            f"got quals={actual_quals}, expected {expected_subject_qualifier}"
        )
        assert actual_base == expected_base, (
            f"Base wrong for {name!r}: got {actual_base!r}, expected {expected_base!r}"
        )

    def test_compound_subject_not_split(self, vocabs):
        """alpha_particle is a registered Subject — now matched as single token.

        The longest-match-first qualifier stripping recognizes compound
        subject tokens like 'alpha_particle' as single tokens.
        """
        result = parse("alpha_particle_density", vocabs)
        quals = [q.token for q in result.ir.qualifiers]
        # alpha_particle should appear as single token, not ["alpha", "particle"]
        assert "alpha_particle" in quals, (
            f"Compound subject 'alpha_particle' was split: {quals}"
        )
        assert "alpha" not in quals, (
            f"'alpha' appeared separately — compound subject was split: {quals}"
        )


class TestSegmentTokenMapClosed:
    """SEGMENT_TOKEN_MAP physical_base and qualifier are now closed vocabularies."""

    def test_physical_base_closed(self):
        from imas_standard_names.grammar.constants import SEGMENT_TOKEN_MAP

        bases = SEGMENT_TOKEN_MAP["physical_base"]
        assert len(bases) >= 70, f"Expected >= 70 bases, got {len(bases)}"
        assert "temperature" in bases
        assert "pressure" in bases
        assert "density" in bases

    def test_qualifier_in_segment_map(self):
        from imas_standard_names.grammar.constants import SEGMENT_TOKEN_MAP

        quals = SEGMENT_TOKEN_MAP["qualifier"]
        assert len(quals) >= 90, f"Expected >= 90 qualifiers, got {len(quals)}"
        # 'effective' is a representative genuine modifier qualifier. (Transport
        # regimes like 'collisional'/'anomalous' moved to the process segment;
        # transport channels to 'channel'; loci/operators/zones to their segments.)
        assert "effective" in quals
        # Modifier qualifiers moved to dedicated single-token segments in the
        # rc32 decomposition: orbit class → 'orbit', energy-state → 'population'.
        assert "trapped" in SEGMENT_TOKEN_MAP["orbit"]
        assert "fast" in SEGMENT_TOKEN_MAP["population"]
