"""Pydantic v2 loaders for vocabulary files.

Each loader reads the corresponding YAML under
``imas_standard_names/grammar/vocabularies/`` and validates it with
strict Pydantic models (``extra='forbid'``).

Public API::

    axes   = load_coordinate_axes()    # CoordinateAxesRegistry
    loci   = load_locus_registry()     # LocusRegistry
    ops    = load_operators()          # OperatorRegistry
    bases  = load_physical_bases()     # PhysicalBasesRegistry
    geo    = load_geometry_carriers()  # GeometryCarriersRegistry
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_VOCAB_DIR = Path(__file__).parent / "vocabularies"


def _load_yaml(filename: str) -> dict:
    """Load a YAML file from the vocabularies directory."""
    path = _VOCAB_DIR / filename
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# coordinate_axes.yml
# ---------------------------------------------------------------------------


class CoordinateAxisDef(BaseModel, extra="forbid"):
    """A single coordinate axis entry."""

    aliases: list[str] = []


class CoordinateAxesRegistry(BaseModel, extra="forbid"):
    """Top-level structure of ``coordinate_axes.yml``."""

    axes: dict[str, CoordinateAxisDef]


def load_coordinate_axes() -> CoordinateAxesRegistry:
    """Load and validate ``coordinate_axes.yml``."""
    data = _load_yaml("coordinate_axes.yml")
    # YAML may contain entries as None (bare key) — normalise to empty dict
    raw_axes = data.get("axes") or {}
    normalised = {k: (v or {}) for k, v in raw_axes.items()}
    return CoordinateAxesRegistry(axes=normalised)


# ---------------------------------------------------------------------------
# locus_registry.yml
# ---------------------------------------------------------------------------

LocusType = Literal["entity", "position", "geometry", "region"]
LocusRelation = Literal["of", "at", "over", "along"]


class LocusEntry(BaseModel, extra="forbid"):
    """A single locus entry (entity, position, geometry, or region)."""

    type: LocusType
    allowed_relations: list[LocusRelation]
    qualifiable: bool = False
    """When true, the feature composes with the registry's ``locus_qualifiers``
    (e.g. ``strike_point`` -> ``inner_strike_point``, ``upper_outer_strike_point``)
    instead of enumerating each geometric variant as its own flat token."""
    definition: str = Field(min_length=20)
    """Normative definition of the governed locus term."""
    abbreviations: list[str] = []
    """Display and search abbreviations; never accepted as parser aliases."""
    references: list[str] = []
    """Optional authoritative references supporting the definition."""
    defining_quantity: str = ""
    """Optional standard name that DEFINES this locus's position (e.g. the
    pedestal is located by ``normalized_poloidal_flux_coordinate_of_pedestal``).
    The description pipeline cross-links this quantity so a name evaluated at the
    locus points at the quantity that fixes the locus. Data-driven: the rule
    lives in the prompt, the mapping lives here. Empty when the locus has no
    single defining standard quantity."""


class LocusRegistry(BaseModel, extra="forbid"):
    """Top-level structure of ``locus_registry.yml``."""

    loci: dict[str, LocusEntry]
    locus_qualifiers: list[str] = []
    """Ordered geometric qualifiers (canonical intra-order) that compose onto a
    ``qualifiable`` locus feature. Scales to advanced divertor topologies
    (snowflake/X/super-X) without enumerating every strike-point/target
    combination."""


def load_locus_registry() -> LocusRegistry:
    """Load and validate ``locus_registry.yml``."""
    data = _load_yaml("locus_registry.yml")
    return LocusRegistry(**data)


# ---------------------------------------------------------------------------
# operators.yml
# ---------------------------------------------------------------------------

OperatorKind = Literal["unary_prefix", "unary_postfix", "binary"]
OperatorSemanticEffect = Literal["temporal_change"]


class OperatorDef(BaseModel, extra="forbid"):
    """A single operator entry."""

    kind: OperatorKind
    precedence: int
    returns: str | None = None
    arg_types: list[str] | None = None
    separator: str | None = None
    indexed: bool = False
    index_params: list[str] | None = None
    dimensionless: bool = False
    # The operator changes the DIMENSIONS of its argument (integrals multiply
    # by a length/area/volume/time, derivatives divide, inverse/square remap).
    # The base-implies-unit heuristic in validation/semantic.py is invalid for
    # names carrying such an operator (volume_integrated density is a count).
    dimension_transforming: bool = False
    # The operator reduces over a flux surface (flux_surface_averaged,
    # maximum/minimum_over_flux_surface). Applied to a base flagged
    # constant_on_flux_surface it is a no-op (the reduction of a flux
    # function is the value itself) and composition rejects the name.
    flux_surface_reduction: bool = False
    # The operator reduces to an extremum over a spatial domain (maximum,
    # minimum). On a base flagged extremum_is_transformation this reduction
    # is only legal in transformation (prefix) position — maximum_of_<base> —
    # so the same token appearing as an INFIX qualifier is rejected.
    extremum_reduction: bool = False
    semantic_effects: frozenset[OperatorSemanticEffect] = frozenset()


class OperatorRegistry(BaseModel, extra="forbid"):
    """Top-level structure of ``operators.yml``."""

    operators: dict[str, OperatorDef]


def load_operators() -> OperatorRegistry:
    """Load and validate ``operators.yml``."""
    data = _load_yaml("operators.yml")
    return OperatorRegistry(**data)


# ---------------------------------------------------------------------------
# physical_bases.yml
# ---------------------------------------------------------------------------


BaseKind = Literal["scalar", "vector", "tensor", "complex"]


class PhysicalBaseDef(BaseModel, extra="forbid"):
    """A single physical base entry.

    Attributes:
        aliases: Alternate tokens that map to this base.
        kind: Physical kind — scalar, vector, tensor, or complex.
        inherently_dimensional: If true, this base normally carries SI units
            and marking it dimensionless (unit='1') is flagged by the validator.
        constant_on_flux_surface: If true, this base is a flux function —
            constant on any flux surface — so a flux-surface reduction
            operator (flux_surface_averaged, maximum/minimum_over_flux_surface)
            applied to it is a no-op and the composition gate rejects it.
        extremum_is_transformation: If true, an extremum of this base over a
            spatial domain (peak/maximum/minimum) is a reduction transformation
            and must be spelled maximum_of_<base>_... / minimum_of_<base>_...;
            an infix extremum qualifier inside the base is rejected.
        definition: Normative term definition exposed to grammar consumers.
    """

    aliases: list[str] = []
    kind: BaseKind
    inherently_dimensional: bool = False
    constant_on_flux_surface: bool = False
    extremum_is_transformation: bool = False
    definition: str = ""


class PhysicalBasesRegistry(BaseModel, extra="forbid"):
    """Top-level structure of ``physical_bases.yml``."""

    bases: dict[str, PhysicalBaseDef] = {}


def load_physical_bases() -> PhysicalBasesRegistry:
    """Load and validate ``physical_bases.yml``."""
    data = _load_yaml("physical_bases.yml")
    # YAML may contain entries as None (bare key) — normalise to empty dict
    raw_bases = data.get("bases") or {}
    normalised = {k: (v or {}) for k, v in raw_bases.items()}
    return PhysicalBasesRegistry(bases=normalised)


# ---------------------------------------------------------------------------
# geometry_carriers.yml
# ---------------------------------------------------------------------------


class GeometryCarrierDef(BaseModel, extra="forbid"):
    """A single geometry carrier entry.

    Attributes:
        aliases: Alternate tokens that map to this carrier.
        constant_on_flux_surface: If true, this carrier is a flux label
            (rho_tor and friends) — constant on any flux surface — so a
            flux-surface reduction operator applied to it is a no-op and
            the composition gate rejects it.
        definition: Normative term definition exposed to grammar consumers.
    """

    aliases: list[str] = []
    constant_on_flux_surface: bool = False
    definition: str = ""


class GeometryCarriersRegistry(BaseModel, extra="forbid"):
    """Top-level structure of ``geometry_carriers.yml``."""

    carriers: dict[str, GeometryCarrierDef] = {}


def load_geometry_carriers() -> GeometryCarriersRegistry:
    """Load and validate ``geometry_carriers.yml``."""
    data = _load_yaml("geometry_carriers.yml")
    # YAML may contain entries as None (bare key) — normalise to empty dict
    raw_carriers = data.get("carriers") or {}
    normalised = {k: (v or {}) for k, v in raw_carriers.items()}
    return GeometryCarriersRegistry(carriers=normalised)


# ---------------------------------------------------------------------------
# qualifiers.yml
# ---------------------------------------------------------------------------


def load_qualifiers() -> frozenset[str]:
    """Load qualifier tokens from ``qualifiers.yml``.

    The file is a flat YAML list of string tokens (with optional inline
    comments). Returns the set of modifier qualifiers; the parser unions
    these with Subject enum tokens to form the full qualifier vocabulary.
    """
    path = _VOCAB_DIR / "qualifiers.yml"
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not data:
        return frozenset()
    if isinstance(data, list):
        return frozenset(str(token) for token in data)
    # Support dict format with top-level "qualifiers:" key
    if isinstance(data, dict) and "qualifiers" in data:
        return frozenset(str(token) for token in (data["qualifiers"] or []))
    return frozenset()


def load_qualifier_categories() -> dict[str, str]:
    """Load the token → category map from ``qualifier_categories.yml``.

    The file groups every ``qualifiers.yml`` token under one of a small set of
    normalized presentation categories (transport, source, geometry, region,
    state, energy, diagnostic, polarization, temporal, normalized, species,
    engineering). Returns a flat ``{token: category}`` mapping; missing or
    malformed file yields an empty dict so callers degrade gracefully.
    """
    try:
        data = _load_yaml("qualifier_categories.yml")
    except Exception:
        return {}
    sections = data.get("qualifier_categories") if isinstance(data, dict) else None
    if not isinstance(sections, dict):
        return {}
    out: dict[str, str] = {}
    for category, tokens in sections.items():
        for token in tokens or []:
            out[str(token)] = str(category)
    return out


# ---------------------------------------------------------------------------
# populations.yml
# ---------------------------------------------------------------------------


def load_populations() -> frozenset[str]:
    """Load population modifier tokens from ``populations.yml``.

    Population tokens (energy-state fast/thermal/cold/hot/suprathermal,
    molecularity, bulk core) compose orthogonally with a species subject. The
    parser unions these into its qualifier vocabulary so they peel; the
    StandardName model retains them in the scalar ``population`` segment.
    Flat YAML list of string tokens (with optional inline comments).
    """
    return _load_flat_token_list("populations.yml", "populations")


def load_states() -> frozenset[str]:
    """Load state-resolution tokens from ``states.yml``.

    State tokens (charge_state on ions, internal_state on neutrals) form a
    single-token closed segment that refines the subject to a specific
    charge/internal state. The parser unions these into its qualifier
    vocabulary so they peel; the StandardName model retains them in the scalar
    ``state`` segment. Flat YAML list of string tokens.
    """
    return _load_flat_token_list("states.yml", "states")


def load_orbits() -> frozenset[str]:
    """Load orbit/transit-class tokens from ``orbits.yml``.

    Orbit tokens (trapped, co_passing, counter_passing, co_current,
    counter_current) form a single-token closed segment orthogonal to
    ``population`` and ``subject``. Unioned into the parser qualifier
    vocabulary so they peel; retained on the model in the ``orbit`` segment.
    """
    return _load_flat_token_list("orbits.yml", "orbits")


def load_aggregations() -> frozenset[str]:
    """Load aggregation tokens from ``aggregations.yml``.

    Aggregation tokens (total, net) mark a population/species/contribution
    reduction. Single-token closed segment, orthogonal to ``population`` (it
    legitimately stacks with it) and rendered outermost. Unioned into the parser
    qualifier vocabulary so they peel; retained on the model in the
    ``aggregation`` segment.
    """
    return _load_flat_token_list("aggregations.yml", "aggregations")


def load_section_planes() -> frozenset[str]:
    """Load plane tokens used to qualify cross-sectional identities."""
    return _load_flat_token_list("section_planes.yml", "section_planes")


def load_geometry_representations() -> frozenset[str]:
    """Load object-local geometry representation tokens."""
    return _load_flat_token_list(
        "geometry_representations.yml", "geometry_representations"
    )


def load_zones() -> tuple[str, ...]:
    """Load zone tokens from ``zones.yml`` in canonical intra-order.

    Zone tokens (plasma-region / geometric sub-selectors: core, edge,
    pedestal, separatrix, divertor, scrape_off_layer; vertical upper/lower;
    radial inner/outer; PFC face front_surface/back_surface/wetted) compose as
    an ordered PREFIX segment. Unlike the other modifier segments a name may
    carry MULTIPLE zone tokens, which MUST appear in the order returned here
    (the file order is the canonical intra-order). The parser unions these into
    its qualifier vocabulary so they peel; the StandardName model retains them
    (multiple) in the ``zone`` segment.

    Returns an ORDERED tuple (not a frozenset) so the canonical intra-order is
    preserved for both rendering and validation.
    """
    path = _VOCAB_DIR / "zones.yml"
    if not path.exists():
        return ()
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not data:
        return ()
    if isinstance(data, list):
        return tuple(str(token) for token in data)
    if isinstance(data, dict) and "zones" in data:
        return tuple(str(token) for token in (data["zones"] or []))
    return ()


def load_channels() -> tuple[str, ...]:
    """Load transport-channel tokens from ``channels.yml`` in file order.

    Channel tokens (heat, particle, energy, momentum) name WHAT is transported.
    They compose as the INNERMOST prefix segment (immediately before the base,
    after any residual qualifier). The channel segment is SINGLE-token: a name
    carries at most one transport channel. The parser unions these into its
    qualifier acceptance set so they peel; the StandardName model retains the
    single token in the ``channel`` segment.

    Returns an ORDERED tuple (file order) so the generated ``Channel`` enum
    preserves the locked vocabulary order (heat, particle, energy, momentum).
    """
    path = _VOCAB_DIR / "channels.yml"
    if not path.exists():
        return ()
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not data:
        return ()
    if isinstance(data, list):
        return tuple(str(token) for token in data)
    if isinstance(data, dict) and "channels" in data:
        return tuple(str(token) for token in (data["channels"] or []))
    return ()


def load_channel_qualifiers() -> tuple[str, ...]:
    """Load channel-qualifier tokens from ``channel_qualifiers.yml`` (file order).

    Channel-qualifier tokens (kinetic, plasma) bind to the transport channel:
    they refine WHICH channel quantity is meant and render immediately OUTER of
    the channel (before it) and INNER of the zone. The segment is SINGLE-token:
    a name carries at most one channel-qualifier. The parser unions these into
    its qualifier acceptance set so they peel; the StandardName model retains
    the single token in the ``channel_qualifier`` segment.

    Returns an ORDERED tuple (file order) so the generated ``ChannelQualifier``
    enum preserves the locked vocabulary order (kinetic, plasma).
    """
    path = _VOCAB_DIR / "channel_qualifiers.yml"
    if not path.exists():
        return ()
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not data:
        return ()
    if isinstance(data, list):
        return tuple(str(token) for token in data)
    if isinstance(data, dict) and "channel_qualifiers" in data:
        return tuple(str(token) for token in (data["channel_qualifiers"] or []))
    return ()


def _load_flat_token_list(filename: str, dict_key: str) -> frozenset[str]:
    """Load a flat YAML token list (or ``{<dict_key>: [...]}``)."""
    path = _VOCAB_DIR / filename
    if not path.exists():
        return frozenset()
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not data:
        return frozenset()
    if isinstance(data, list):
        return frozenset(str(token) for token in data)
    if isinstance(data, dict) and dict_key in data:
        return frozenset(str(token) for token in (data[dict_key] or []))
    return frozenset()


# ---------------------------------------------------------------------------
# normalizing_qualifiers.yml
# ---------------------------------------------------------------------------


def load_scoping_qualifiers() -> frozenset[str]:
    """Load the phrase-scoping subset of the qualifier vocabulary.

    These are qualifier tokens (must also appear in ``qualifiers.yml``)
    that modify the whole compound noun phrase rather than forming a
    lexical kind with the base — they route to the model's ``qualifier``
    segment and render outermost among the refined qualifiers (before
    zone, orbit, population, subject, and the channel pair). Every other
    qualifier token is kind-forming and stays glued to the base.
    """
    path = _VOCAB_DIR / "scoping_qualifiers.yml"
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not data:
        return frozenset()
    if isinstance(data, dict) and "scoping_qualifiers" in data:
        return frozenset(str(t) for t in (data["scoping_qualifiers"] or []))
    if isinstance(data, list):
        return frozenset(str(token) for token in data)
    return frozenset()


def load_normalizing_qualifiers() -> frozenset[str]:
    """Load qualifier tokens that imply dimensionless output.

    These are a subset of qualifier tokens (must also appear in
    ``qualifiers.yml``) that indicate the quantity has been normalized
    to dimensionless form (e.g. gyrokinetic normalization).
    """
    path = _VOCAB_DIR / "normalizing_qualifiers.yml"
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not data:
        return frozenset()
    if isinstance(data, list):
        return frozenset(str(token) for token in data)
    return frozenset()


# ---------------------------------------------------------------------------
# Cross-registry validation
# ---------------------------------------------------------------------------


class _AllRegistries(BaseModel):
    """Container used only for cross-registry duplicate-key checks."""

    axes: CoordinateAxesRegistry
    loci: LocusRegistry
    operators: OperatorRegistry
    bases: PhysicalBasesRegistry
    carriers: GeometryCarriersRegistry

    @model_validator(mode="after")
    def _no_duplicate_names_across_registries(self) -> _AllRegistries:
        """Assert no token appears in more than one vocabulary registry."""
        pools: list[tuple[str, set[str]]] = [
            ("coordinate_axes", set(self.axes.axes)),
            ("locus_registry", set(self.loci.loci)),
            ("operators", set(self.operators.operators)),
            ("physical_bases", set(self.bases.bases)),
            ("geometry_carriers", set(self.carriers.carriers)),
        ]
        seen: dict[str, str] = {}  # token -> first registry
        duplicates: list[str] = []
        for registry_name, tokens in pools:
            for token in tokens:
                if token in seen:
                    duplicates.append(
                        f"'{token}' in both '{seen[token]}' and '{registry_name}'"
                    )
                else:
                    seen[token] = registry_name
        if duplicates:
            raise ValueError(
                "Duplicate tokens found across vocabulary registries:\n  "
                + "\n  ".join(duplicates)
            )
        return self


def validate_no_cross_registry_duplicates() -> None:
    """Load all five registries and assert no duplicate token names.

    Raises ``ValueError`` if any token appears in more than one registry.
    """
    _AllRegistries(
        axes=load_coordinate_axes(),
        loci=load_locus_registry(),
        operators=load_operators(),
        bases=load_physical_bases(),
        carriers=load_geometry_carriers(),
    )
