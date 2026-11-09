"""DD-backed vocabulary registrations and canonical quantity identities."""

from dataclasses import dataclass
from importlib.resources import files

import pytest
import yaml

from imas_standard_names import compose, parse
from imas_standard_names.grammar.constants import SEGMENT_TOKEN_MAP


@dataclass(frozen=True)
class VocabularyCase:
    """A governed token, its definition and a quantity from its DD source."""

    token: str
    segment: str
    filename: str
    definition: str
    name: str
    source_path: str


CASES = [
    VocabularyCase(
        "horizontal",
        "component",
        "components.yml",
        "Selects the horizontal transverse direction of a beam cross-section, perpendicular to its vertical transverse direction and beam propagation axis.",
        "horizontal_width",
        "nbi/unit/beamlets_group/width_horizontal",
    ),
    VocabularyCase(
        "pipe",
        "object",
        "locus_registry.yml",
        "A bounded tubular conduit transporting a fluid between inlet and outlet.",
        "length_of_pipe",
        "gas_injection/pipe/length",
    ),
    VocabularyCase(
        "image_surface",
        "position",
        "locus_registry.yml",
        "The optical surface on which an optical system forms a focused image.",
        "radius_of_image_surface",
        "spectrometer_uv/channel/grating/image_field/curvature_radius",
    ),
    VocabularyCase(
        "ion_source",
        "object",
        "locus_registry.yml",
        "The emitting source that creates and extracts ions to form an injected particle beam.",
        "area_of_ion_source",
        "nbi/unit/source/surface",
    ),
    VocabularyCase(
        "optical_fibre",
        "object",
        "locus_registry.yml",
        "An individual dielectric optical waveguide fibre that transports light, distinct from a bundle of fibres.",
        "radius_of_optical_fibre",
        "camera_visible/channel/fibre_bundle/fibre_radius",
    ),
    VocabularyCase(
        "power_convertor",
        "object",
        "locus_registry.yml",
        "Electrical equipment that converts and regulates power delivered to a load.",
        "voltage_of_power_convertor",
        "pf_active/supply/voltage",
    ),
    VocabularyCase(
        "slit",
        "object",
        "locus_registry.yml",
        "A narrow optical aperture that limits the admitted light in one transverse direction.",
        "width_of_slit",
        "spectrometer_visible/channel/grating_spectrometer/slit_width",
    ),
    VocabularyCase(
        "arc_of_circle",
        "geometry_representation",
        "geometry_representations.yml",
        "A contour represented by circular arc segments, each specified by an endpoint and a curvature radius.",
        "radial_arc_of_circle_vertex_of_antenna_strap",
        "ic_antennas/antenna/module/strap/geometry/arcs_of_circle/r",
    ),
    VocabularyCase(
        "image",
        "geometric_base",
        "geometry_carriers.yml",
        "The spatial image of an object formed by an optical system.",
        "vertical_image_of_optical_fibre",
        "spectrometer_visible/channel/fibre_image/outline/z",
    ),
    VocabularyCase(
        "thermal_shield",
        "object",
        "locus_registry.yml",
        "A shielding structure that intercepts thermal radiation and reduces heat transfer to colder equipment.",
        "radial_outline_of_thermal_shield",
        "cryostat/description_2d/thermal_shield/unit/annular/outline_outer/r",
    ),
    VocabularyCase(
        "oblique",
        "geometry_representation",
        "geometry_representations.yml",
        "A planar cross-section represented by inclined edges and reference coordinates rather than an axis-aligned rectangle.",
        "vertical_oblique_position_of_iron_core_segment",
        "iron_core/segment/geometry/oblique/z",
    ),
    VocabularyCase(
        "thick_line",
        "geometry_representation",
        "geometry_representations.yml",
        "A cross-section represented by a central line segment together with its finite transverse thickness.",
        "radial_thick_line_vertex_of_plasma_filament",
        "pf_plasma/element/geometry/thick_line/first_point/r",
    ),
    VocabularyCase(
        "ballooning_parameter",
        "physical_base",
        "physical_bases.yml",
        "The normalized pressure-gradient parameter that measures ballooning-mode drive in the adopted equilibrium convention.",
        "ratio_of_critical_ballooning_parameter_to_ballooning_parameter",
        "summary/pedestal_fits/mtanh/stability/bootstrap_current_hager/alpha_ratio/value",
    ),
    VocabularyCase(
        "delta_prime",
        "physical_base",
        "physical_bases.yml",
        "The tearing-stability contribution with inverse-length units entering the modified Rutherford equation for magnetic-island width evolution.",
        "delta_prime_of_neoclassical_tearing_mode",
        "ntms/time_slice/mode/deltaw/value",
    ),
    VocabularyCase(
        "helical",
        "qualifier",
        "qualifiers.yml",
        "Selects a variation following a helical path around a toroidal confinement system.",
        "effective_helical_ripple_magnetic_field_amplitude",
        "summary/local/r_eff_norm_2_3/effective_helical_ripple/value",
    ),
    VocabularyCase(
        "ripple",
        "qualifier",
        "qualifiers.yml",
        "Selects the modulation of magnetic-field magnitude, with the particular ripple measure defined by its physical convention.",
        "effective_helical_ripple_magnetic_field_amplitude",
        "summary/local/r_eff_norm_2_3/effective_helical_ripple/value",
    ),
    VocabularyCase(
        "sine",
        "qualifier",
        "qualifiers.yml",
        "Selects the sine basis functions in a Fourier expansion of a periodic quantity.",
        "sine_coefficient_radial_derivative_of_flux_surface",
        "gyrokinetics_local/flux_surface/ds_dr_minor_norm",
    ),
    VocabularyCase(
        "stress",
        "physical_base",
        "physical_bases.yml",
        "A rank-two tensor whose contraction with a surface normal gives force per unit area transmitted across that surface.",
        "stress",
        "mhd_linear/time_slice/toroidal_mode/plasma/ntv/imaginary",
    ),
    VocabularyCase(
        "magnetic_field_line",
        "position",
        "locus_registry.yml",
        "A curve whose tangent is everywhere parallel to the magnetic-field vector.",
        "vertical_position_of_magnetic_field_line",
        "plasma_initiation/b_field_lines/starting_positions/z",
    ),
    VocabularyCase(
        "grid",
        "geometric_base",
        "geometry_carriers.yml",
        "A set of spatial sampling points on which a physical field or distribution is represented, with coordinates and connectivity specified independently.",
        "vertical_grid",
        "b_field_non_axisymmetric/time_slice/field_map/grid/z",
    ),
    VocabularyCase(
        "pre_sawtooth_crash",
        "position",
        "locus_registry.yml",
        "The evaluation state immediately before a sawtooth crash.",
        "psi_star_at_pre_sawtooth_crash",
        "sawteeth/profiles_1d/psi_star_pre_crash",
    ),
    VocabularyCase(
        "chemical",
        "qualifier",
        "qualifiers.yml",
        "Selects material removal mediated by chemical reaction and volatile-product formation.",
        "effective_chemical_coefficient_due_to_sputtering",
        "wall/global_quantities/neutral/incident_species/sputtering_chemical",
    ),
    VocabularyCase(
        "tritium_decay",
        "process",
        "processes.yml",
        "Radioactive beta decay of tritium whose emitted electrons supply a runaway-electron seed population.",
        "runaway_electron_source_rate_due_to_tritium_decay",
        "runaway_electrons/ggd_fluid/ddensity_dt_tritium/values",
    ),
    VocabularyCase(
        "bounce",
        "qualifier",
        "qualifiers.yml",
        "Selects the periodic bounce or transit motion along a particle orbit used as the angular variable of an orbit Fourier expansion.",
        "bounce_mode_number",
        "distributions/distribution/markers/orbit_integrals/bounce_harmonics",
    ),
    VocabularyCase(
        "compressive",
        "qualifier",
        "qualifiers.yml",
        "Selects a structural force component defined as positive when it compresses the loaded component.",
        "radial_compressive_force_of_poloidal_field_coil",
        "pf_active/coil/force_radial_crushing",
    ),
    VocabularyCase(
        "cosine",
        "qualifier",
        "qualifiers.yml",
        "Selects the cosine basis functions in a Fourier expansion of a periodic quantity.",
        "cosine_coefficient_radial_derivative_of_flux_surface",
        "gyrokinetics_local/flux_surface/dc_dr_minor_norm",
    ),
    VocabularyCase(
        "fragmentation",
        "process",
        "processes.yml",
        "Breakup of a solid pellet into smaller fragments, including the gas released during that breakup.",
        "gas_flow_of_pellet_injector_due_to_fragmentation",
        "spi/injector/fragmentation_gas/flow_rate",
    ),
    VocabularyCase(
        "threshold_crossing",
        "position",
        "locus_registry.yml",
        "The event at which a specified observable first crosses a prescribed threshold in a specified direction.",
        "time_at_threshold_crossing",
        "summary/disruption/time_half_ip/value",
    ),
    VocabularyCase(
        "lithium_6",
        "subject",
        "subjects.yml",
        "The lithium isotope with mass number six, whose abundance is measured relative to total lithium in the blanket.",
        "lithium_6_fraction_of_breeder_blanket",
        "breeding_blanket/li6_fraction",
    ),
    VocabularyCase(
        "second_inclination",
        "qualifier",
        "qualifiers.yml",
        "Selects the second edge-orientation angle in a two-angle oblique cross-section parameterization.",
        "second_inclination_oblique_angle_of_iron_core_segment",
        "iron_core/segment/geometry/oblique/beta",
    ),
    VocabularyCase(
        "rotational",
        "qualifier",
        "qualifiers.yml",
        "Selects the pressure contribution R0² ρ ω²/2 associated with toroidal bulk plasma rotation.",
        "rotational_pressure",
        "equilibrium/time_slice/constraints/pressure_rotational/measured",
    ),
    VocabularyCase(
        "side",
        "zone",
        "zones.yml",
        "Selects a lateral face of a component, excluding its upper and lower end faces.",
        "side_thickness_of_breeder_blanket_module",
        "breeding_blanket/module/side_wall_thickness",
    ),
    VocabularyCase(
        "sublimation",
        "process",
        "processes.yml",
        "The direct phase transition of a solid into a gas, whose energy here is given per atom.",
        "energy_of_pellet_due_to_sublimation",
        "pellets/time_slice/pellet/species/sublimation_energy",
    ),
    VocabularyCase(
        "tangency",
        "qualifier",
        "qualifiers.yml",
        "Selects the closest-approach geometry where a beam centreline is tangent to a toroidal circle.",
        "tangency_radius",
        "nbi/unit/beamlets_group/beamlets/tangency_radii",
    ),
]


@pytest.fixture(scope="module")
def vocabulary_sources() -> dict[str, str]:
    root = files("imas_standard_names.grammar") / "vocabularies"
    return {case.filename: (root / case.filename).read_text() for case in CASES}


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.token)
def test_source_quantity_round_trips(case: VocabularyCase) -> None:
    result = parse(case.name, strict=True)
    assert compose(result.ir) == case.name, case.source_path
    assert parse(compose(result.ir), strict=True).ir == result.ir
    assert case.token in SEGMENT_TOKEN_MAP[case.segment]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.token)
def test_token_keeps_its_source_definition(
    case: VocabularyCase, vocabulary_sources: dict[str, str]
) -> None:
    text = vocabulary_sources[case.filename]
    data = yaml.safe_load(text)
    if isinstance(data, list):
        assert case.token in data
        assert f"- {case.token}  # {case.definition}" in text
    elif "loci" in data:
        entry = data["loci"][case.token]
        assert entry["definition"] == case.definition
        expected_type = "entity" if case.segment == "object" else "position"
        assert entry["type"] == expected_type
        assert entry["allowed_relations"] == (
            ["of"]
            if expected_type == "entity"
            else ["at", "of", "along"]
            if case.token == "magnetic_field_line"
            else ["at", "of"]
        )
    elif "bases" in data or "carriers" in data:
        root = "bases" if "bases" in data else "carriers"
        assert data[root][case.token]["definition"] == case.definition
    else:
        assert data[case.token][case.token] == case.definition


@pytest.mark.parametrize(
    ("segment", "token"),
    [
        ("qualifier", "fit_offset"),
        ("qualifier", "s0"),
        ("qualifier", "s1"),
        ("qualifier", "s2"),
        ("qualifier", "s3"),
        ("region", "outside_last_closed_flux_surface"),
        ("object", "poloidal_field_circuit"),
        ("object", "poloidal_field_power_supply"),
        ("component", "toroidal_placeholder"),
        ("physical_base", "coordinate_placeholder"),
    ],
)
def test_rejected_spelling_is_not_registered(segment: str, token: str) -> None:
    assert token not in SEGMENT_TOKEN_MAP[segment]


@pytest.mark.parametrize(
    ("token", "segments"),
    [
        ("spun", {"qualifier"}),
        ("toroidal", {"component", "coordinate"}),
        ("coordinate", {"geometric_base"}),
    ],
)
def test_reused_token_keeps_exact_registrations(token: str, segments: set[str]) -> None:
    assert {
        segment for segment, tokens in SEGMENT_TOKEN_MAP.items() if token in tokens
    } == segments
