"""Grammar-owned guidance for non-canonical source terms."""

import json

import pytest

from imas_standard_names.grammar import loader as alias_loader
from imas_standard_names.grammar.context import get_grammar_context
from imas_standard_names.grammar.loader import load_advisory_aliases
from imas_standard_names.grammar.parser import load_default_vocabularies

ACCEPTED_ALIASES = {
    ("position", "rectangle_centre"): "rectangle_center",
    ("position", "rectangular_cross_section_centre"): "rectangle_center",
    ("position", "annulus_centre"): "annulus_center",
    ("physical_base", "strain_tensor"): "strain",
    ("transformation", "tendency"): "time_derivative",
}

CONTEXTUAL_REWRITES = {
    "rectangle",
    "oblique",
    "oblique_geometry",
    "arcs_of_circle",
    "arc_of_circle",
    "annular_centreline",
    "annulus",
}


def test_advisory_aliases_are_segment_scoped_and_vocab_validated() -> None:
    aliases = load_advisory_aliases()

    assert {
        (segment, source): definition["canonical"]
        for segment, entries in aliases.items()
        for source, definition in entries.items()
    } == ACCEPTED_ALIASES
    assert all(
        definition["reason"]
        for entries in aliases.values()
        for definition in entries.values()
    )


def test_contextual_rewrites_are_not_admitted_as_aliases() -> None:
    aliases = load_advisory_aliases()
    sources = {source for entries in aliases.values() for source in entries}

    assert sources.isdisjoint(CONTEXTUAL_REWRITES)


def test_advisory_aliases_do_not_change_parser_acceptance() -> None:
    vocabularies = load_default_vocabularies()

    assert not (set(CONTEXTUAL_REWRITES) & set(vocabularies.loci))
    assert not (set(CONTEXTUAL_REWRITES) & vocabularies.bases)
    assert not (
        {source for _, source in ACCEPTED_ALIASES} & set(vocabularies.carrier_aliases)
    )
    assert not (
        {source for _, source in ACCEPTED_ALIASES} & set(vocabularies.base_aliases)
    )


def test_advisory_alias_mapping_is_immutable_and_serializable() -> None:
    aliases = load_advisory_aliases()

    json.dumps(aliases)
    with pytest.raises(TypeError):
        aliases["position"]["rectangle_centre"] = {
            "canonical": "rectangle_center",
            "reason": "mutation",
        }


def test_context_cache_preserves_alias_immutability() -> None:
    aliases = get_grammar_context()["grammar"]["advisory_aliases"]

    json.dumps(aliases)
    with pytest.raises(TypeError):
        aliases["position"]["rectangle_centre"]["canonical"] = "rectangle"


@pytest.mark.parametrize(
    "payload",
    [
        """
segments:
  unknown_segment:
    source_term:
      canonical: rectangle_center
      reason: This segment is deliberately not governed.
""",
        """
segments:
  position:
    source_term:
      canonical: missing_target
      reason: This target is deliberately not registered.
""",
        """
segments:
  position:
    rectangle_center:
      canonical: rectangle_center
      reason: This source deliberately collides with the vocabulary.
""",
        """
segments:
  position:
    shared_source:
      canonical: rectangle_center
      reason: This source is deliberately repeated across segments.
  physical_base:
    shared_source:
      canonical: strain
      reason: This source is deliberately repeated across segments.
""",
        """
segments:
  position:
    repeated_source:
      canonical: rectangle_center
      reason: This source is deliberately repeated in one segment.
    repeated_source:
      canonical: annulus_center
      reason: This source is deliberately repeated in one segment.
""",
        """
segments:
  position:
    invalid-source:
      canonical: rectangle_center
      reason: This source deliberately violates token syntax.
""",
    ],
)
def test_advisory_alias_loader_rejects_invalid_contract(
    payload: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    vocabulary_dir = tmp_path / "vocabularies"
    vocabulary_dir.mkdir()
    (vocabulary_dir / "advisory_aliases.yml").write_text(payload, encoding="utf-8")
    monkeypatch.setattr(alias_loader.resources, "files", lambda package: tmp_path)
    load_advisory_aliases.cache_clear()

    with pytest.raises(ValueError):
        load_advisory_aliases()

    load_advisory_aliases.cache_clear()
