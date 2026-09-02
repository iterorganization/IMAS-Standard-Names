"""Unit-vector carriers need an owning-object locus (semantic check)."""

from imas_standard_names.models import create_standard_name_entry
from imas_standard_names.validation.semantic import run_semantic_checks


def _entry(name: str) -> dict:
    return {
        "name": name,
        "kind": "scalar",
        "physics_domain": "general",
        "status": "draft",
        "unit": "1",
        "description": "Unit vector component for locus-check testing.",
        "documentation": (
            "Unit vector component used to exercise the orientation locus "
            "semantic check."
        ),
    }


def _issues_for(name: str) -> list[str]:
    entry = create_standard_name_entry(_entry(name))
    return [i for i in run_semantic_checks({name: entry}) if name in i]


def test_locusless_direction_unit_vector_errors():
    issues = _issues_for("x_direction_unit_vector")
    assert any("ERROR" in i and "direction_unit_vector" in i for i in issues)


def test_locus_qualified_direction_unit_vector_clean():
    issues = _issues_for("x_direction_unit_vector_of_camera")
    assert not any("unit vector" in i for i in issues)


def test_locusless_normal_still_errors():
    issues = _issues_for("surface_normal")
    assert any("ERROR" in i for i in issues)


def test_position_qualifier_satisfies_geometric_requirement():
    issues = _issues_for("normalized_toroidal_flux_coordinate_at_magnetic_axis")
    assert not any("ERROR" in i for i in issues)


def test_intrinsic_coordinate_bases_exempt_when_bare():
    for name in (
        "normalized_toroidal_flux_coordinate",
        "toroidal_flux_coordinate",
        "poloidal_angle",
    ):
        issues = _issues_for(name)
        assert not any("ERROR" in i for i in issues), issues


def test_locus_qualified_geometric_base_clean():
    issues = _issues_for("radial_position_of_flux_loop")
    assert not any("ERROR" in i for i in issues)
