"""Photon-number radiance stays distinct from energy radiometric quantities."""

from imas_standard_names import compose, get_grammar_context, parse
from imas_standard_names.grammar import (
    compose_standard_name,
    parse_standard_name,
)
from imas_standard_names.models import create_standard_name_entry
from imas_standard_names.validation import run_semantic_checks


def test_photon_radiance_at_spectral_line_strict_round_trip() -> None:
    name = "photon_radiance_at_spectral_line"

    parsed = parse(name, strict=True)

    assert parsed.ir.base.token == "photon_radiance"
    assert compose(parsed.ir) == name


def test_photon_radiance_at_spectral_line_flat_round_trip() -> None:
    name = "photon_radiance_at_spectral_line"

    parsed = parse_standard_name(name)

    assert parsed.physical_base == "photon_radiance"
    assert compose_standard_name(parsed) == name


def test_photon_radiance_is_a_distinct_dimensional_base() -> None:
    parsed_bases = {
        name: parse(name, strict=True).ir.base.token
        for name in ("photon_radiance", "radiance", "brightness")
    }
    assert parsed_bases == {
        "photon_radiance": "photon_radiance",
        "radiance": "radiance",
        "brightness": "brightness",
    }

    entry = create_standard_name_entry(
        {
            "name": "photon_radiance",
            "kind": "scalar",
            "physics_domain": "radiation_measurement_diagnostics",
            "status": "draft",
            "unit": "1",
            "description": "Photon-number radiance for dimensional validation.",
            "documentation": (
                "Photon-number radiance used to verify that the quantity "
                "requires physical units."
            ),
        }
    )
    issues = run_semantic_checks({entry.name: entry})
    assert any(
        "dimensionless unit '1'" in issue and "'photon_radiance'" in issue
        for issue in issues
    )


def test_public_grammar_context_exposes_photon_radiance() -> None:
    sections = {
        section["segment"]: section
        for section in get_grammar_context()["vocabulary_sections"]
    }

    assert "photon_radiance" in sections["physical_base"]["tokens"]
