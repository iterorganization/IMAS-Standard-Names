"""Grammar contracts for plane-qualified sections and local arc radii."""

import pytest

from imas_standard_names import compose, get_grammar_context, parse
from imas_standard_names.grammar import (
    GeometryRepresentation,
    SectionPlane,
    compose_standard_name,
    parse_standard_name,
)
from imas_standard_names.grammar.parser import ParseError


def test_poloidal_cross_section_uses_plane_segment_not_projection() -> None:
    name = "poloidal_plane_cross_sectional_area_of_flux_surface"

    ir = parse(name, strict=True).ir
    model = parse_standard_name(name)

    assert ir.projection is None
    assert [(qualifier.token, qualifier.category) for qualifier in ir.qualifiers] == [
        ("poloidal", "section_plane"),
        ("cross_sectional", "geometry"),
    ]
    assert model.section_plane is SectionPlane.POLOIDAL
    assert model.component is None
    assert model.physical_base == "cross_sectional_area"
    assert compose(ir) == name
    assert compose_standard_name(model) == name


def test_section_plane_has_one_canonical_position() -> None:
    assert (
        compose_standard_name(
            {
                "section_plane": "poloidal",
                "geometric_base": "cross_section",
                "object": "coil_conductor",
            }
        )
        == "poloidal_plane_cross_section_of_coil_conductor"
    )
    geometric_model = parse_standard_name(
        "poloidal_plane_cross_section_of_coil_conductor"
    )
    assert geometric_model.section_plane is SectionPlane.POLOIDAL
    assert geometric_model.geometric_base.value == "cross_section"

    assert (
        compose_standard_name(
            {
                "section_plane": "poloidal",
                "physical_base": "cross_sectional_area",
                "object": "conductor_cross_section",
            }
        )
        == "poloidal_plane_cross_sectional_area_of_conductor_cross_section"
    )

    projection_name = "poloidal_cross_section"
    projection_ir = parse(projection_name, strict=True).ir
    assert projection_ir.projection is not None
    assert projection_ir.projection.axis == "poloidal"
    assert projection_ir.qualifiers == []
    assert compose(projection_ir) == projection_name

    with pytest.raises(ParseError, match="cross-sectional identities require"):
        parse("cross_sectional_area_of_conductor_cross_section", strict=True)
    with pytest.raises(ParseError, match="cross-sectional identities require"):
        parse("cross_section_of_coil_conductor", strict=True)
    with pytest.raises(ValueError, match="only valid on a cross-sectional"):
        compose_standard_name(
            {
                "section_plane": "poloidal",
                "physical_base": "area",
                "object": "conductor_cross_section",
            }
        )


def test_local_circle_radius_is_owner_qualified_representation() -> None:
    name = "local_circle_radius_of_passive_loop_element"

    ir = parse(name, strict=True).ir
    model = parse_standard_name(name)

    assert ir.projection is None
    assert [qualifier.token for qualifier in ir.qualifiers] == ["local_circle"]
    assert model.geometry_representation is GeometryRepresentation.LOCAL_CIRCLE
    assert model.physical_base == "radius"
    assert model.object.value == "passive_loop_element"
    assert compose(ir) == name
    assert compose_standard_name(model) == name


def test_local_circle_radius_does_not_collapse_to_global_or_outline_identity() -> None:
    identities = {
        "local_circle_radius_of_passive_loop_element",
        "radial_coordinate_of_passive_loop_element",
        "radial_outline_of_passive_loop_element",
    }

    assert {compose(parse(name, strict=True).ir) for name in identities} == identities
    assert (
        parse_standard_name(
            "radial_coordinate_of_passive_loop_element"
        ).geometric_base.value
        == "radial_coordinate"
    )
    assert (
        parse_standard_name(
            "radial_outline_of_passive_loop_element"
        ).geometric_base.value
        == "outline"
    )


@pytest.mark.parametrize(
    "name",
    [
        "local_circle_radius",
        "radial_local_circle_radius_of_passive_loop_element",
        "local_circle_area_of_passive_loop_element",
    ],
)
def test_local_circle_representation_rejects_missing_or_conflicting_semantics(
    name: str,
) -> None:
    with pytest.raises(ParseError):
        parse(name, strict=True)


@pytest.mark.parametrize(
    "name",
    [
        "start_local_circle_radius_of_passive_loop_element",
        "end_local_circle_radius_of_passive_loop_element",
        "first_local_circle_radius_of_passive_loop_element",
        "second_poloidal_plane_cross_sectional_area_of_conductor_cross_section",
        "third_poloidal_plane_cross_sectional_area_of_conductor_cross_section",
    ],
)
def test_ordered_sample_labels_are_not_identity_tokens(name: str) -> None:
    with pytest.raises(ParseError):
        parse(name, strict=True)


def test_public_context_exposes_plane_and_representation_vocabularies() -> None:
    context = get_grammar_context()
    vocabularies = context["grammar"]["vocabularies"]
    templates = context["grammar"]["canonical_templates"]

    assert vocabularies["section_planes"] == ["poloidal"]
    assert vocabularies["geometry_representations"] == [
        "arc_of_circle",
        "local_circle",
        "oblique",
        "thick_line",
    ]
    assert templates["section_plane"] == "<plane>_plane_<cross_sectional_quantity>"
    assert (
        templates["geometry_representation"] == "<representation>_<quantity>_of_<owner>"
    )
    assert "section_plane" in context["segment_order"]
    assert "geometry_representation" in context["segment_order"]


@pytest.mark.parametrize(
    "name",
    [
        "arc_of_circle_radius_of_antenna_strap",
        "radial_arc_of_circle_vertex_of_antenna_strap",
        "vertical_oblique_position_of_iron_core_segment",
        "second_inclination_oblique_angle_of_iron_core_segment",
        "radial_thick_line_vertex_of_plasma_filament",
    ],
)
def test_declared_representation_bases_keep_their_canonical_identity(name: str) -> None:
    ir = parse(name, strict=True).ir
    model = parse_standard_name(name)
    assert compose(ir) == name
    assert compose_standard_name(model) == name


@pytest.mark.parametrize(
    "name",
    [
        "arc_of_circle_area_of_antenna_strap",
        "oblique_radius_of_iron_core_segment",
        "thick_line_length_of_plasma_filament",
        "arc_of_circle_vertex",
        "oblique_position",
        "thick_line_vertex",
        "radial_arc_of_circle_radius_of_antenna_strap",
        "arc_of_circle_oblique_position_of_iron_core_segment",
        "arc_of_circle_local_circle_radius_of_antenna_strap",
    ],
)
def test_representation_rejects_undeclared_bases_and_incompatible_context(
    name: str,
) -> None:
    with pytest.raises(ParseError):
        parse(name, strict=True)
