"""Canonical DD quantity identities and the vocabulary that preserves them."""

from dataclasses import dataclass
from importlib.resources import files

import pytest
import yaml

from imas_standard_names import compose, parse
from imas_standard_names.grammar.constants import SEGMENT_TOKEN_MAP
from imas_standard_names.grammar.model import parse_standard_name
from imas_standard_names.grammar.vocab_loaders import load_scoping_qualifiers


@dataclass(frozen=True)
class VocabularyCase:
    """A token, its physical definition and the source quantity it names."""

    token: str
    segment: str
    filename: str
    definition: str
    name: str
    source_path: str


CASES = [
    VocabularyCase(
        "pellet_species",
        "subject",
        "subjects.yml",
        "A chemically or isotopically identified constituent population in the material composition of an intact injected pellet, resolved by the pellet species coordinate; distinct from a shattered-pellet constituent.",
        "pellet_species_number_density_of_pellet",
        "pellets/time_slice/pellet/species/density",
    ),
    VocabularyCase(
        "breakdown_candidate",
        "position",
        "locus_registry.yml",
        "The magnetic-field-line set selected as plasma-breakdown candidates because each selected line encounters the Townsend breakdown condition or is a closed magnetic field line. The grid representation contains original tracing-grid points containing at least one selected line; the position representation contains samples along the selected lines.",
        "vertical_grid_of_breakdown_candidate",
        "plasma_initiation/b_field_lines/townsend_or_closed_grid_positions/z",
    ),
    VocabularyCase(
        "breakdown_candidate",
        "position",
        "locus_registry.yml",
        "The magnetic-field-line set selected as plasma-breakdown candidates because each selected line encounters the Townsend breakdown condition or is a closed magnetic field line. The grid representation contains original tracing-grid points containing at least one selected line; the position representation contains samples along the selected lines.",
        "vertical_position_of_breakdown_candidate",
        "plasma_initiation/b_field_lines/townsend_or_closed_positions/z",
    ),
    VocabularyCase(
        "current_profile_alignment",
        "physical_base",
        "physical_bases.yml",
        "A global, current-valued figure of merit measuring spatial agreement among the current-density profiles supplying the plasma current, evaluated with the source-profile comparison and weighting prescribed by the DD-cited Wade et al. convention; it is not an angle, a current-density vector component, or an assumed dimensionless factor.",
        "current_profile_alignment",
        "summary/global_quantities/current_alignment/value",
    ),
    VocabularyCase(
        "plateau_factor",
        "physical_base",
        "physical_bases.yml",
        "The dimensionless magnetic-configuration factor for plateau-regime neoclassical transport, sensitive to toroidal and helical magnetic-field harmonics and designated by the DD as the plateau factor of Stroth et al., equation (25); it characterizes magnetic geometry, not flattening of a density or temperature profile.",
        "plateau_factor",
        "summary/local/r_eff_norm_2_3/plateau_factor/value",
    ),
    VocabularyCase(
        "sourced",
        "qualifier",
        "qualifiers.yml",
        "Selects the additive contribution supplied by a source term to the named physical quantity, rather than the total quantity. It does not assert a particular mechanism or external origin, and it preserves the dimensions of the named quantity rather than introducing a temporal derivative.",
        "sourced_current_density",
        "edge_sources/source/ggd/current/values",
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_source_quantity_round_trips(case: VocabularyCase) -> None:
    result = parse(case.name, strict=True)
    assert compose(result.ir) == case.name, case.source_path
    assert parse(compose(result.ir), strict=True).ir == result.ir
    assert case.token in SEGMENT_TOKEN_MAP[case.segment]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_token_keeps_its_definition(case: VocabularyCase) -> None:
    text = (
        files("imas_standard_names.grammar") / "vocabularies" / case.filename
    ).read_text()
    data = yaml.safe_load(text)
    if isinstance(data, list):
        assert case.token in data
        assert f"- {case.token}  # {case.definition}" in text
    else:
        registry = data["loci" if case.segment == "position" else "bases"]
        assert registry[case.token]["definition"] == case.definition
        if case.segment == "position":
            assert registry[case.token]["type"] == "position"
            assert registry[case.token]["allowed_relations"] == ["at", "of", "along"]
        else:
            assert registry[case.token]["kind"] == "scalar"
            assert registry[case.token].get("inherently_dimensional", False) == (
                case.token == "current_profile_alignment"
            )


def test_intact_and_shattered_pellet_species_are_distinct_subjects() -> None:
    intact = parse_standard_name("pellet_species_number_density_of_pellet")
    shattered = parse_standard_name("shattered_pellet_species_number_density")
    assert intact.subject == "pellet_species"
    assert shattered.subject == "shattered_pellet_species"
    assert intact.physical_base == shattered.physical_base == "number_density"
    assert {"pellet_species", "shattered_pellet_species"} <= set(
        SEGMENT_TOKEN_MAP["subject"]
    )


def test_sourced_scopes_over_the_species_quantity() -> None:
    name = "sourced_electron_current_density"
    result = parse(name, strict=True)
    assert compose(result.ir) == name
    flat = parse_standard_name(name)
    assert flat.subject == "electron"
    assert flat.qualifier == ("sourced",)
    assert flat.physical_base == "current_density"
    assert "sourced" in load_scoping_qualifiers()
    assert (
        next(q.category for q in result.ir.qualifiers if q.token == "sourced")
        == "source"
    )
    assert "source" not in SEGMENT_TOKEN_MAP["qualifier"]
