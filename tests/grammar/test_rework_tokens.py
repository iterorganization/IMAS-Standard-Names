"""Distinct pellet populations and whole-quantity source contributions."""

from imas_standard_names import compose, parse
from imas_standard_names.grammar.constants import SEGMENT_TOKEN_MAP
from imas_standard_names.grammar.ir import StandardNameIR
from imas_standard_names.grammar.model import parse_standard_name
from imas_standard_names.grammar.vocab_loaders import load_scoping_qualifiers


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
    assert compose(result.ir) == "sourced_electron_current_density"
    expected_ir = StandardNameIR.model_validate(
        {
            "base": {"token": "current_density", "kind": "quantity"},
            "qualifiers": [
                {"token": "sourced", "category": "source"},
                {"token": "electron"},
            ],
        }
    )
    assert result.ir == expected_ir
    assert parse(compose(result.ir), strict=True).ir == expected_ir
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
