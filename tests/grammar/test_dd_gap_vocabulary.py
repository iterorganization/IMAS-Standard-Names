"""DD-backed vocabulary registrations and canonical quantity identities."""

from dataclasses import dataclass
from importlib.resources import files

import pytest
import yaml

from imas_standard_names import compose, parse
from imas_standard_names.grammar.constants import SEGMENT_TOKEN_MAP
from imas_standard_names.grammar.ir import StandardNameIR


@dataclass(frozen=True)
class VocabularyCase:
    """A governed token, its definition and a quantity from its DD source."""

    token: str
    segment: str
    filename: str
    definition: str
    name: str
    source_path: str
    expected_spelling: str
    expected_ir: dict[str, object]
    allowed_relations: tuple[str, ...] | None = None
    kind: str | None = None
    inherently_dimensional: bool | None = None


CASES = [
    VocabularyCase(
        "horizontal",
        "component",
        "components.yml",
        "Selects the horizontal transverse direction of a beam cross-section, perpendicular to its vertical transverse direction and beam propagation axis.",
        "horizontal_width",
        "nbi/unit/beamlets_group/width_horizontal",
        expected_spelling="horizontal_width",
        expected_ir={
            "base": {"token": "width", "kind": "quantity"},
            "projection": {"axis": "horizontal", "shape": "component"},
        },
    ),
    VocabularyCase(
        "pipe",
        "object",
        "locus_registry.yml",
        "A bounded tubular conduit transporting a fluid between inlet and outlet.",
        "length_of_pipe",
        "gas_injection/pipe/length",
        expected_spelling="length_of_pipe",
        expected_ir={
            "base": {"token": "length", "kind": "quantity"},
            "locus": {"relation": "of", "token": "pipe", "type": "entity"},
        },
    ),
    VocabularyCase(
        "image_surface",
        "position",
        "locus_registry.yml",
        "The optical surface on which an optical system forms a focused image.",
        "radius_of_image_surface",
        "spectrometer_uv/channel/grating/image_field/curvature_radius",
        expected_spelling="radius_of_image_surface",
        expected_ir={
            "base": {"token": "radius", "kind": "quantity"},
            "locus": {"relation": "of", "token": "image_surface", "type": "position"},
        },
    ),
    VocabularyCase(
        "ion_source",
        "object",
        "locus_registry.yml",
        "The emitting source that creates and extracts ions to form an injected particle beam.",
        "area_of_ion_source",
        "nbi/unit/source/surface",
        expected_spelling="area_of_ion_source",
        expected_ir={
            "base": {"token": "area", "kind": "quantity"},
            "locus": {"relation": "of", "token": "ion_source", "type": "entity"},
        },
    ),
    VocabularyCase(
        "optical_fibre",
        "object",
        "locus_registry.yml",
        "An individual dielectric optical waveguide fibre that transports light, distinct from a bundle of fibres.",
        "radius_of_optical_fibre",
        "camera_visible/channel/fibre_bundle/fibre_radius",
        expected_spelling="radius_of_optical_fibre",
        expected_ir={
            "base": {"token": "radius", "kind": "quantity"},
            "locus": {"relation": "of", "token": "optical_fibre", "type": "entity"},
        },
    ),
    VocabularyCase(
        "power_convertor",
        "object",
        "locus_registry.yml",
        "Electrical equipment that converts and regulates power delivered to a load.",
        "voltage_of_power_convertor",
        "pf_active/supply/voltage",
        expected_spelling="voltage_of_power_convertor",
        expected_ir={
            "base": {"token": "voltage", "kind": "quantity"},
            "locus": {"relation": "of", "token": "power_convertor", "type": "entity"},
        },
    ),
    VocabularyCase(
        "slit",
        "object",
        "locus_registry.yml",
        "A narrow optical aperture that limits the admitted light in one transverse direction.",
        "width_of_slit",
        "spectrometer_visible/channel/grating_spectrometer/slit_width",
        expected_spelling="width_of_slit",
        expected_ir={
            "base": {"token": "width", "kind": "quantity"},
            "locus": {"relation": "of", "token": "slit", "type": "entity"},
        },
    ),
    VocabularyCase(
        "arc_of_circle",
        "geometry_representation",
        "geometry_representations.yml",
        "A contour represented by circular arc segments, each specified by an endpoint and a curvature radius.",
        "radial_arc_of_circle_vertex_of_antenna_strap",
        "ic_antennas/antenna/module/strap/geometry/arcs_of_circle/r",
        expected_spelling="radial_arc_of_circle_vertex_of_antenna_strap",
        expected_ir={
            "base": {"token": "vertex", "kind": "geometry"},
            "qualifiers": [{"token": "arc_of_circle"}],
            "locus": {"relation": "of", "token": "antenna_strap", "type": "entity"},
            "projection": {"axis": "radial", "shape": "coordinate"},
        },
    ),
    VocabularyCase(
        "image",
        "geometric_base",
        "geometry_carriers.yml",
        "The spatial image of an object formed by an optical system.",
        "vertical_image_of_optical_fibre",
        "spectrometer_visible/channel/fibre_image/outline/z",
        expected_spelling="vertical_image_of_optical_fibre",
        expected_ir={
            "base": {"token": "image", "kind": "geometry"},
            "locus": {"relation": "of", "token": "optical_fibre", "type": "entity"},
            "projection": {"axis": "vertical", "shape": "coordinate"},
        },
    ),
    VocabularyCase(
        "thermal_shield",
        "object",
        "locus_registry.yml",
        "A shielding structure that intercepts thermal radiation and reduces heat transfer to colder equipment.",
        "radial_outline_of_thermal_shield",
        "cryostat/description_2d/thermal_shield/unit/annular/outline_outer/r",
        expected_spelling="radial_outline_of_thermal_shield",
        expected_ir={
            "base": {"token": "outline", "kind": "geometry"},
            "locus": {"relation": "of", "token": "thermal_shield", "type": "entity"},
            "projection": {"axis": "radial", "shape": "coordinate"},
        },
    ),
    VocabularyCase(
        "oblique",
        "geometry_representation",
        "geometry_representations.yml",
        "A planar cross-section represented by inclined edges and reference coordinates rather than an axis-aligned rectangle.",
        "vertical_oblique_position_of_iron_core_segment",
        "iron_core/segment/geometry/oblique/z",
        expected_spelling="vertical_oblique_position_of_iron_core_segment",
        expected_ir={
            "base": {"token": "position", "kind": "geometry"},
            "qualifiers": [{"token": "oblique"}],
            "locus": {"relation": "of", "token": "iron_core_segment", "type": "entity"},
            "projection": {"axis": "vertical", "shape": "coordinate"},
        },
    ),
    VocabularyCase(
        "thick_line",
        "geometry_representation",
        "geometry_representations.yml",
        "A cross-section represented by a central line segment together with its finite transverse thickness.",
        "radial_thick_line_vertex_of_plasma_filament",
        "pf_plasma/element/geometry/thick_line/first_point/r",
        expected_spelling="radial_thick_line_vertex_of_plasma_filament",
        expected_ir={
            "base": {"token": "vertex", "kind": "geometry"},
            "qualifiers": [{"token": "thick_line"}],
            "locus": {"relation": "of", "token": "plasma_filament", "type": "entity"},
            "projection": {"axis": "radial", "shape": "coordinate"},
        },
    ),
    VocabularyCase(
        "ballooning_parameter",
        "physical_base",
        "physical_bases.yml",
        "The normalized pressure-gradient parameter that measures ballooning-mode drive in the adopted equilibrium convention.",
        "ratio_of_critical_ballooning_parameter_to_ballooning_parameter",
        "summary/pedestal_fits/mtanh/stability/bootstrap_current_hager/alpha_ratio/value",
        expected_spelling="ratio_of_critical_ballooning_parameter_to_ballooning_parameter",
        expected_ir={
            "base": {"token": "placeholder", "kind": "quantity"},
            "operators": [
                {
                    "kind": "binary",
                    "op": "ratio",
                    "separator": "to",
                    "args": [
                        {
                            "base": {
                                "token": "ballooning_parameter",
                                "kind": "quantity",
                            },
                            "qualifiers": [{"token": "critical", "category": "state"}],
                        },
                        {"base": {"token": "ballooning_parameter", "kind": "quantity"}},
                    ],
                }
            ],
        },
    ),
    VocabularyCase(
        "delta_prime",
        "physical_base",
        "physical_bases.yml",
        "The tearing-stability contribution with inverse-length units entering the modified Rutherford equation for magnetic-island width evolution.",
        "delta_prime_of_neoclassical_tearing_mode",
        "ntms/time_slice/mode/deltaw/value",
        expected_spelling="delta_prime_of_neoclassical_tearing_mode",
        expected_ir={
            "base": {"token": "delta_prime", "kind": "quantity"},
            "locus": {
                "relation": "of",
                "token": "neoclassical_tearing_mode",
                "type": "entity",
            },
        },
    ),
    VocabularyCase(
        "helical",
        "qualifier",
        "qualifiers.yml",
        "Selects a variation following a helical path around a toroidal confinement system.",
        "effective_helical_ripple_magnetic_field_amplitude",
        "summary/local/r_eff_norm_2_3/effective_helical_ripple/value",
        expected_spelling="effective_helical_ripple_magnetic_field_amplitude",
        expected_ir={
            "base": {"token": "magnetic_field", "kind": "quantity"},
            "qualifiers": [
                {"token": "effective", "category": "state"},
                {"token": "helical", "category": "geometry"},
                {"token": "ripple", "category": "geometry"},
            ],
            "operators": [{"kind": "unary_postfix", "op": "amplitude"}],
        },
    ),
    VocabularyCase(
        "ripple",
        "qualifier",
        "qualifiers.yml",
        "Selects the modulation of magnetic-field magnitude, with the particular ripple measure defined by its physical convention.",
        "effective_helical_ripple_magnetic_field_amplitude",
        "summary/local/r_eff_norm_2_3/effective_helical_ripple/value",
        expected_spelling="effective_helical_ripple_magnetic_field_amplitude",
        expected_ir={
            "base": {"token": "magnetic_field", "kind": "quantity"},
            "qualifiers": [
                {"token": "effective", "category": "state"},
                {"token": "helical", "category": "geometry"},
                {"token": "ripple", "category": "geometry"},
            ],
            "operators": [{"kind": "unary_postfix", "op": "amplitude"}],
        },
    ),
    VocabularyCase(
        "sine",
        "qualifier",
        "qualifiers.yml",
        "Selects the sine basis functions in a Fourier expansion of a periodic quantity.",
        "sine_coefficient_radial_derivative_of_flux_surface",
        "gyrokinetics_local/flux_surface/ds_dr_minor_norm",
        expected_spelling="sine_coefficient_radial_derivative_of_flux_surface",
        expected_ir={
            "base": {"token": "coefficient", "kind": "quantity"},
            "qualifiers": [{"token": "sine", "category": "diagnostic"}],
            "locus": {"relation": "of", "token": "flux_surface", "type": "position"},
            "operators": [{"kind": "unary_prefix", "op": "radial_derivative"}],
        },
    ),
    VocabularyCase(
        "stress",
        "physical_base",
        "physical_bases.yml",
        "A rank-two tensor whose contraction with a surface normal gives force per unit area transmitted across that surface.",
        "stress",
        "mhd_linear/time_slice/toroidal_mode/plasma/ntv/imaginary",
        expected_spelling="stress",
        expected_ir={"base": {"token": "stress", "kind": "quantity"}},
    ),
    VocabularyCase(
        "magnetic_field_line",
        "position",
        "locus_registry.yml",
        "A curve whose tangent is everywhere parallel to the magnetic-field vector.",
        "vertical_position_of_magnetic_field_line",
        "plasma_initiation/b_field_lines/starting_positions/z",
        expected_spelling="vertical_position_of_magnetic_field_line",
        expected_ir={
            "base": {"token": "position", "kind": "geometry"},
            "locus": {
                "relation": "of",
                "token": "magnetic_field_line",
                "type": "position",
            },
            "projection": {"axis": "vertical", "shape": "coordinate"},
        },
    ),
    VocabularyCase(
        "grid",
        "geometric_base",
        "geometry_carriers.yml",
        "A set of spatial sampling points on which a physical field or distribution is represented, with coordinates and connectivity specified independently.",
        "vertical_grid",
        "b_field_non_axisymmetric/time_slice/field_map/grid/z",
        expected_spelling="vertical_grid",
        expected_ir={
            "base": {"token": "grid", "kind": "geometry"},
            "projection": {"axis": "vertical", "shape": "coordinate"},
        },
    ),
    VocabularyCase(
        "pre_sawtooth_crash",
        "position",
        "locus_registry.yml",
        "The evaluation state immediately before a sawtooth crash.",
        "psi_star_at_pre_sawtooth_crash",
        "sawteeth/profiles_1d/psi_star_pre_crash",
        expected_spelling="psi_star_at_pre_sawtooth_crash",
        expected_ir={
            "base": {"token": "psi_star", "kind": "quantity"},
            "locus": {
                "relation": "at",
                "token": "pre_sawtooth_crash",
                "type": "position",
            },
        },
    ),
    VocabularyCase(
        "chemical",
        "qualifier",
        "qualifiers.yml",
        "Selects material removal mediated by chemical reaction and volatile-product formation.",
        "effective_chemical_coefficient_due_to_sputtering",
        "wall/global_quantities/neutral/incident_species/sputtering_chemical",
        expected_spelling="effective_chemical_coefficient_due_to_sputtering",
        expected_ir={
            "base": {"token": "coefficient", "kind": "quantity"},
            "qualifiers": [
                {"token": "effective", "category": "state"},
                {"token": "chemical", "category": "source"},
            ],
            "mechanism": {"token": "sputtering"},
        },
    ),
    VocabularyCase(
        "tritium_decay",
        "process",
        "processes.yml",
        "Radioactive beta decay of tritium whose emitted electrons supply a runaway-electron seed population.",
        "runaway_electron_source_rate_due_to_tritium_decay",
        "runaway_electrons/ggd_fluid/ddensity_dt_tritium/values",
        expected_spelling="runaway_electron_source_rate_due_to_tritium_decay",
        expected_ir={
            "base": {"token": "source_rate", "kind": "quantity"},
            "qualifiers": [{"token": "runaway_electron"}],
            "mechanism": {"token": "tritium_decay"},
        },
    ),
    VocabularyCase(
        "bounce",
        "qualifier",
        "qualifiers.yml",
        "Selects the periodic bounce or transit motion along a particle orbit used as the angular variable of an orbit Fourier expansion.",
        "bounce_mode_number",
        "distributions/distribution/markers/orbit_integrals/bounce_harmonics",
        expected_spelling="bounce_mode_number",
        expected_ir={
            "base": {"token": "mode_number", "kind": "quantity"},
            "qualifiers": [{"token": "bounce", "category": "transport"}],
        },
    ),
    VocabularyCase(
        "compressive",
        "qualifier",
        "qualifiers.yml",
        "Selects a structural force component defined as positive when it compresses the loaded component.",
        "radial_compressive_force_of_poloidal_field_coil",
        "pf_active/coil/force_radial_crushing",
        expected_spelling="radial_compressive_force_of_poloidal_field_coil",
        expected_ir={
            "base": {"token": "force", "kind": "quantity"},
            "qualifiers": [{"token": "compressive", "category": "engineering"}],
            "locus": {
                "relation": "of",
                "token": "poloidal_field_coil",
                "type": "entity",
            },
            "projection": {"axis": "radial", "shape": "component"},
        },
    ),
    VocabularyCase(
        "cosine",
        "qualifier",
        "qualifiers.yml",
        "Selects the cosine basis functions in a Fourier expansion of a periodic quantity.",
        "cosine_coefficient_radial_derivative_of_flux_surface",
        "gyrokinetics_local/flux_surface/dc_dr_minor_norm",
        expected_spelling="cosine_coefficient_radial_derivative_of_flux_surface",
        expected_ir={
            "base": {"token": "coefficient", "kind": "quantity"},
            "qualifiers": [{"token": "cosine", "category": "diagnostic"}],
            "locus": {"relation": "of", "token": "flux_surface", "type": "position"},
            "operators": [{"kind": "unary_prefix", "op": "radial_derivative"}],
        },
    ),
    VocabularyCase(
        "fragmentation",
        "process",
        "processes.yml",
        "Breakup of a solid pellet into smaller fragments, including the gas released during that breakup.",
        "gas_flow_of_pellet_injector_due_to_fragmentation",
        "spi/injector/fragmentation_gas/flow_rate",
        expected_spelling="gas_flow_of_pellet_injector_due_to_fragmentation",
        expected_ir={
            "base": {"token": "flow", "kind": "quantity"},
            "qualifiers": [{"token": "gas", "category": "source"}],
            "locus": {"relation": "of", "token": "pellet_injector", "type": "entity"},
            "mechanism": {"token": "fragmentation"},
        },
    ),
    VocabularyCase(
        "threshold_crossing",
        "position",
        "locus_registry.yml",
        "The event at which a specified observable first crosses a prescribed threshold in a specified direction.",
        "time_at_threshold_crossing",
        "summary/disruption/time_half_ip/value",
        expected_spelling="time_at_threshold_crossing",
        expected_ir={
            "base": {"token": "time", "kind": "quantity"},
            "locus": {
                "relation": "at",
                "token": "threshold_crossing",
                "type": "position",
            },
        },
    ),
    VocabularyCase(
        "lithium_6",
        "subject",
        "subjects.yml",
        "The lithium isotope with mass number six, whose abundance is measured relative to total lithium in the blanket.",
        "lithium_6_fraction_of_breeder_blanket",
        "breeding_blanket/li6_fraction",
        expected_spelling="lithium_6_fraction_of_breeder_blanket",
        expected_ir={
            "base": {"token": "fraction", "kind": "quantity"},
            "qualifiers": [{"token": "lithium_6"}],
            "locus": {"relation": "of", "token": "breeder_blanket", "type": "entity"},
        },
    ),
    VocabularyCase(
        "second_inclination",
        "qualifier",
        "qualifiers.yml",
        "Selects the second edge-orientation angle in a two-angle oblique cross-section parameterization.",
        "second_inclination_oblique_angle_of_iron_core_segment",
        "iron_core/segment/geometry/oblique/beta",
        expected_spelling="second_inclination_oblique_angle_of_iron_core_segment",
        expected_ir={
            "base": {"token": "angle", "kind": "quantity"},
            "qualifiers": [
                {"token": "second_inclination", "category": "geometry"},
                {"token": "oblique"},
            ],
            "locus": {"relation": "of", "token": "iron_core_segment", "type": "entity"},
        },
    ),
    VocabularyCase(
        "rotational",
        "qualifier",
        "qualifiers.yml",
        "Selects the pressure contribution R0² ρ ω²/2 associated with toroidal bulk plasma rotation.",
        "rotational_pressure",
        "equilibrium/time_slice/constraints/pressure_rotational/measured",
        expected_spelling="rotational_pressure",
        expected_ir={
            "base": {"token": "pressure", "kind": "quantity"},
            "qualifiers": [{"token": "rotational", "category": "energy"}],
        },
    ),
    VocabularyCase(
        "side",
        "zone",
        "zones.yml",
        "Selects a lateral face of a component, excluding its upper and lower end faces.",
        "side_thickness_of_breeder_blanket_module",
        "breeding_blanket/module/side_wall_thickness",
        expected_spelling="side_thickness_of_breeder_blanket_module",
        expected_ir={
            "base": {"token": "thickness", "kind": "quantity"},
            "qualifiers": [{"token": "side"}],
            "locus": {
                "relation": "of",
                "token": "breeder_blanket_module",
                "type": "entity",
            },
        },
    ),
    VocabularyCase(
        "sublimation",
        "process",
        "processes.yml",
        "The direct phase transition of a solid into a gas, whose energy here is given per atom.",
        "energy_of_pellet_due_to_sublimation",
        "pellets/time_slice/pellet/species/sublimation_energy",
        expected_spelling="energy_of_pellet_due_to_sublimation",
        expected_ir={
            "base": {"token": "energy", "kind": "quantity"},
            "locus": {"relation": "of", "token": "pellet", "type": "entity"},
            "mechanism": {"token": "sublimation"},
        },
    ),
    VocabularyCase(
        "tangency",
        "qualifier",
        "qualifiers.yml",
        "Selects the closest-approach geometry where a beam centreline is tangent to a toroidal circle.",
        "tangency_radius",
        "nbi/unit/beamlets_group/beamlets/tangency_radii",
        expected_spelling="tangency_radius",
        expected_ir={
            "base": {"token": "radius", "kind": "quantity"},
            "qualifiers": [{"token": "tangency", "category": "geometry"}],
        },
    ),
    VocabularyCase(
        "pellet_species",
        "subject",
        "subjects.yml",
        "A chemically or isotopically identified constituent population in the material composition of an intact injected pellet, resolved by the pellet species coordinate; distinct from a shattered-pellet constituent.",
        "pellet_species_number_density_of_pellet",
        "pellets/time_slice/pellet/species/density",
        expected_spelling="pellet_species_number_density_of_pellet",
        expected_ir={
            "base": {"token": "number_density", "kind": "quantity"},
            "qualifiers": [{"token": "pellet_species"}],
            "locus": {"relation": "of", "token": "pellet", "type": "entity"},
        },
    ),
    VocabularyCase(
        "breakdown_candidate",
        "position",
        "locus_registry.yml",
        "The magnetic-field-line set selected as plasma-breakdown candidates because each selected line encounters the Townsend breakdown condition or is a closed magnetic field line. The grid representation contains original tracing-grid points containing at least one selected line; the position representation contains samples along the selected lines.",
        "vertical_grid_of_breakdown_candidate",
        "plasma_initiation/b_field_lines/townsend_or_closed_grid_positions/z",
        expected_spelling="vertical_grid_of_breakdown_candidate",
        expected_ir={
            "base": {"token": "grid", "kind": "geometry"},
            "locus": {
                "relation": "of",
                "token": "breakdown_candidate",
                "type": "position",
            },
            "projection": {"axis": "vertical", "shape": "coordinate"},
        },
        allowed_relations=("at", "of", "along"),
    ),
    VocabularyCase(
        "breakdown_candidate",
        "position",
        "locus_registry.yml",
        "The magnetic-field-line set selected as plasma-breakdown candidates because each selected line encounters the Townsend breakdown condition or is a closed magnetic field line. The grid representation contains original tracing-grid points containing at least one selected line; the position representation contains samples along the selected lines.",
        "vertical_position_of_breakdown_candidate",
        "plasma_initiation/b_field_lines/townsend_or_closed_positions/z",
        expected_spelling="vertical_position_of_breakdown_candidate",
        expected_ir={
            "base": {"token": "position", "kind": "geometry"},
            "locus": {
                "relation": "of",
                "token": "breakdown_candidate",
                "type": "position",
            },
            "projection": {"axis": "vertical", "shape": "coordinate"},
        },
        allowed_relations=("at", "of", "along"),
    ),
    VocabularyCase(
        "current_profile_alignment",
        "physical_base",
        "physical_bases.yml",
        "A global, current-valued figure of merit measuring spatial agreement among the current-density profiles supplying the plasma current, evaluated with the source-profile comparison and weighting prescribed by the DD-cited Wade et al. convention; it is not an angle, a current-density vector component, or an assumed dimensionless factor.",
        "current_profile_alignment",
        "summary/global_quantities/current_alignment/value",
        expected_spelling="current_profile_alignment",
        expected_ir={
            "base": {"token": "current_profile_alignment", "kind": "quantity"}
        },
        kind="scalar",
        inherently_dimensional=True,
    ),
    VocabularyCase(
        "plateau_factor",
        "physical_base",
        "physical_bases.yml",
        "The dimensionless magnetic-configuration factor for plateau-regime neoclassical transport, sensitive to toroidal and helical magnetic-field harmonics and designated by the DD as the plateau factor of Stroth et al., equation (25); it characterizes magnetic geometry, not flattening of a density or temperature profile.",
        "plateau_factor",
        "summary/local/r_eff_norm_2_3/plateau_factor/value",
        expected_spelling="plateau_factor",
        expected_ir={"base": {"token": "plateau_factor", "kind": "quantity"}},
        kind="scalar",
        inherently_dimensional=False,
    ),
    VocabularyCase(
        "sourced",
        "qualifier",
        "qualifiers.yml",
        "Selects the additive contribution supplied by a source term to the named physical quantity, rather than the total quantity. It does not assert a particular mechanism or external origin, and it preserves the dimensions of the named quantity rather than introducing a temporal derivative.",
        "sourced_current_density",
        "edge_sources/source/ggd/current/values",
        expected_spelling="sourced_current_density",
        expected_ir={
            "base": {"token": "current_density", "kind": "quantity"},
            "qualifiers": [{"token": "sourced", "category": "source"}],
        },
    ),
]


@pytest.fixture(scope="module")
def vocabulary_sources() -> dict[str, str]:
    root = files("imas_standard_names.grammar") / "vocabularies"
    return {case.filename: (root / case.filename).read_text() for case in CASES}


@pytest.mark.parametrize(
    "case", CASES, ids=lambda case: f"{case.token}:{case.expected_spelling}"
)
def test_source_quantity_round_trips(case: VocabularyCase) -> None:
    result = parse(case.name, strict=True)
    expected_ir = StandardNameIR.model_validate(case.expected_ir)
    assert result.ir == expected_ir, case.source_path
    rendered = compose(result.ir)
    assert rendered == case.expected_spelling, case.source_path
    assert parse(rendered, strict=True).ir == expected_ir
    assert case.token in SEGMENT_TOKEN_MAP[case.segment]


@pytest.mark.parametrize(
    "case", CASES, ids=lambda case: f"{case.token}:{case.expected_spelling}"
)
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
            list(case.allowed_relations)
            if case.allowed_relations is not None
            else ["of"]
            if expected_type == "entity"
            else ["at", "of", "along"]
            if case.token == "magnetic_field_line"
            else ["at", "of"]
        )
    elif "bases" in data or "carriers" in data:
        root = "bases" if "bases" in data else "carriers"
        entry = data[root][case.token]
        assert entry["definition"] == case.definition
        if case.kind is not None:
            assert entry["kind"] == case.kind
        if case.inherently_dimensional is not None:
            assert (
                entry.get("inherently_dimensional", False)
                == case.inherently_dimensional
            )
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
