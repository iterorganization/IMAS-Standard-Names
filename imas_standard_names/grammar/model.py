"""Static StandardName model and friendly wrappers.

This module holds the hand-written Pydantic model and thin compose/parse
wrappers that bridge the parser/renderer (IR-based) to the flat
StandardName model used by the rest of the codebase.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from functools import cache
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from imas_standard_names.grammar.constants import (
    BINARY_OPERATOR_CONNECTORS,
    EXCLUSIVE_SEGMENT_PAIRS,
    GENERIC_PHYSICAL_BASES,
)
from imas_standard_names.grammar.ir import (
    BARE_PREFIX_OPERATORS,
    LOCUS_VALUE_PATTERN,
    AxisProjection,
    BaseKind,
    LocusRef,
    LocusRelation,
    LocusType,
    OperatorApplication,
    OperatorKind,
    Process as IRProcess,
    ProjectionShape,
    Qualifier,
    QuantityOrCarrier,
    StandardNameIR,
)
from imas_standard_names.grammar.model_types import (
    Aggregation,
    BinaryOperator,
    Channel,
    ChannelQualifier,
    Component,
    Coordinate,
    Decomposition,
    GeometricBase,
    GeometryRepresentation,
    Object,
    Orbit,
    Population,
    Position,
    Process,
    Qualifier as QualifierToken,
    Region,
    SectionPlane,
    State,
    Subject,
    Transformation,
    Zone,
)
from imas_standard_names.grammar.parser import (
    ParseError,
    _NonCanonicalParseError,
    parse as _parse_ir,
)
from imas_standard_names.grammar.render import compose as _compose_ir
from imas_standard_names.grammar.support import (
    TOKEN_PATTERN,
    value_of as _value_of,
)
from imas_standard_names.grammar.vocab_loaders import (
    load_qualifier_categories as _load_qualifier_categories,
    load_scoping_qualifiers as _load_scoping_qualifiers,
)

# BaseToken: pattern for physical_base segment (closed vocabulary since rc56)
TOKEN_PATTERN_STR = r"^[a-z][a-z0-9_]*$"
BaseToken = Annotated[
    str,
    Field(
        description=(
            "Base segment token (root of a standard name); snake_case token "
            "matching ^[a-z][a-z0-9_]*$. Examples: 'temperature', 'density', "
            "'magnetic_field', 'particle_flux'."
        ),
        pattern=TOKEN_PATTERN_STR,
        examples=["temperature", "density", "magnetic_field", "particle_flux"],
    ),
]

# ---------------------------------------------------------------------------
# Subject / Object value sets for qualifier classification
# ---------------------------------------------------------------------------
_SUBJECT_VALUES: frozenset[str] = frozenset(s.value for s in Subject)
_OBJECT_VALUES: frozenset[str] = frozenset(o.value for o in Object)
_POSITION_VALUES: frozenset[str] = frozenset(p.value for p in Position)
_REGION_VALUES: frozenset[str] = frozenset(r.value for r in Region)
_GEOMETRY_VALUES: frozenset[str] = frozenset(g.value for g in GeometricBase)

# Aggregation, population, and orbit: orthogonal single-token modifier segments
# split out of the old compound subject tokens (rc32 decomposition) plus the
# aggregation segment. Each contributes at most one token (no intra-segment
# stacking — a second same-segment token is a hard error, see
# ``_ir_to_model_dict``). Aggregation (total/net) is a distinct dimension that
# legitimately STACKS with population (energy-state); orbit is the transit class.
# They render as prefixes outermost-to-inner:
# ``<aggregation>_<orbit>_<population>_<subject>_<base>``, e.g.
# ``total_trapped_fast_ion_energy`` → aggregation=total, orbit=trapped,
# population=fast, subject=ion.
_AGGREGATION_VALUES: frozenset[str] = frozenset(a.value for a in Aggregation)
_POPULATION_VALUES: frozenset[str] = frozenset(p.value for p in Population)
_STATE_VALUES: frozenset[str] = frozenset(s.value for s in State)
_ORBIT_VALUES: frozenset[str] = frozenset(o.value for o in Orbit)
_SECTION_PLANE_VALUES: frozenset[str] = frozenset(p.value for p in SectionPlane)
_GEOMETRY_REPRESENTATION_VALUES: frozenset[str] = frozenset(
    representation.value for representation in GeometryRepresentation
)

# Zone: ordered plasma-region / geometric sub-selector PREFIX segment. Unlike
# the single-token aggregation/orbit/population segments, a name may carry
# MULTIPLE zone tokens (lower_outer), which MUST appear in the FIXED canonical
# intra-order declared in zones.yml (mirrored by the Zone enum member order:
# vertical upper/lower, radial inner/outer, region core..scrape_off_layer, face
# front_surface/back_surface/wetted). ``_ZONE_ORDER`` gives each token its rank
# so compose() can canonicalize and the validator can reject out-of-order input.
_ZONE_VALUES: frozenset[str] = frozenset(z.value for z in Zone)
_ZONE_ORDER: dict[str, int] = {z.value: i for i, z in enumerate(Zone)}

# Channel: transport-channel PREFIX segment (heat, particle, energy, momentum —
# WHAT is transported). A structural role, not a generic qualifier. SINGLE-token
# (at most one channel per name, like subject/aggregation) and INNERMOST: it
# renders immediately before the base, AFTER any residual qualifier(s). Note
# energy/momentum are ALSO physical_bases — the parser matches the longest base
# first, so a standalone energy/momentum never reaches the qualifier stage and
# only the *_flux/*_diffusivity/* compounds strip the channel token.
_CHANNEL_VALUES: frozenset[str] = frozenset(c.value for c in Channel)

# Qualifier: refined PHRASE-SCOPING qualifier segment. Only the scoping
# subset of qualifiers.yml routes here (scoping_qualifiers.yml: implicit,
# effective, incident, fluctuating, ...): a scoping qualifier modifies the
# WHOLE compound noun phrase — English adjective order (Forsyth's royal
# order; Scontras et al.: less intrinsic composes further from the noun) —
# so it renders OUTERMOST among the refined qualifiers, before zone, orbit,
# population, subject, and the channel pair
# (implicit_electron_energy_source_rate, incident_neutron_fluence,
# effective_ion_momentum_convection_velocity). Every OTHER qualifier token
# is kind-forming (atomic_mass, prefill_count, saturated_current,
# deposited_power with the species as recipient-possessor) and stays glued
# to the base, INNER of the species block. Multi-token; stacked scoping
# qualifiers are canonically ordered by qualifier-category rank (see
# ``_QUALIFIER_CATEGORY_RANK`` below), mirroring the zone-order mechanism.
_QUALIFIER_VALUES: frozenset[str] = frozenset(q.value for q in QualifierToken)
_SCOPING_QUALIFIER_VALUES: frozenset[str] = _load_scoping_qualifiers()

# Canonical intra-order for a MULTI-token refined qualifier segment: rank each
# scoping qualifier by its qualifier_categories.yml category (file order),
# mirroring ``_ZONE_ORDER``. compose() sorts the segment by this rank so a
# non-canonical authored order canonicalizes and is rejected by
# parse_standard_name via the round-trip. The sort is stable, so tokens sharing
# a category keep their authored order.
_QUALIFIER_CATEGORY_OF: dict[str, str] = _load_qualifier_categories()
_QUALIFIER_CATEGORY_RANK: dict[str, int] = {
    category: i
    for i, category in enumerate(dict.fromkeys(_QUALIFIER_CATEGORY_OF.values()))
}


def _qualifier_order(token: str) -> int:
    """Category rank of a qualifier token (unknown tokens sort last)."""
    category = _QUALIFIER_CATEGORY_OF.get(token)
    return _QUALIFIER_CATEGORY_RANK.get(category, len(_QUALIFIER_CATEGORY_RANK))


# Reaction-channel qualifier tokens (reactant pairs). DUAL-ROLE: also subjects.
# A pair is routed to the qualifier segment (rendered before the subject) ONLY
# when another subject token follows it — e.g. deuterium_tritium_neutron_flux
# (qualifier deuterium_tritium + subject neutron + base flux) — so the neutron
# flux of a fusion reaction is expressible. As the sole species token the pair
# stays the subject (deuterium_tritium_density).
_REACTION_CHANNEL_VALUES: frozenset[str] = frozenset(
    token
    for token, category in _load_qualifier_categories().items()
    if category == "reaction_channel"
)

# ChannelQualifier: qualifier that binds to the transport CHANNEL (kinetic,
# plasma). It refines WHICH channel quantity is meant and renders immediately
# OUTER of the channel (before it) and INNER of the zone. SINGLE-token (at most
# one channel-qualifier per name). Distinct from the BASE-binding qualifiers,
# which render INNER of the channel. Note kinetic also forms the atomic base
# kinetic_energy — the parser matches the longest base first, so a standalone
# electron_kinetic_energy parses as base=kinetic_energy and only the
# *_energy_flux / *_momentum_flux compounds strip the channel-qualifier token.
_CHANNEL_QUALIFIER_VALUES: frozenset[str] = frozenset(c.value for c in ChannelQualifier)

# Map from LocusType to model field name
_LOCUS_TYPE_TO_FIELD: dict[LocusType, str] = {
    LocusType.ENTITY: "object",
    LocusType.GEOMETRY: "geometry",
    LocusType.POSITION: "position",
    LocusType.REGION: "region",
}

# Map from model field name to (LocusRelation, LocusType)
_FIELD_TO_LOCUS: dict[str, tuple[LocusRelation, LocusType]] = {
    "object": (LocusRelation.OF, LocusType.ENTITY),
    "geometry": (LocusRelation.OF, LocusType.POSITION),
    "position": (LocusRelation.AT, LocusType.POSITION),
    "region": (LocusRelation.OVER, LocusType.REGION),
    "path": (LocusRelation.ALONG, LocusType.POSITION),
}

# Map between model BinaryOperator tokens (e.g. "ratio_of") and IR bare tokens
_BINARY_MODEL_TO_IR: dict[str, str] = {
    "product_of": "product",
    "ratio_of": "ratio",
    "difference_of": "difference",
}
_BINARY_IR_TO_MODEL: dict[str, str] = {v: k for k, v in _BINARY_MODEL_TO_IR.items()}

# Map from IR binary operator separator to connector (for lookups)
_BINARY_SEPARATOR_MAP: dict[str, str] = {}
for _model_tok, _connector in BINARY_OPERATOR_CONNECTORS.items():
    _ir_tok = _BINARY_MODEL_TO_IR.get(_model_tok)
    if _ir_tok:
        _BINARY_SEPARATOR_MAP[_ir_tok] = _connector

# ---------------------------------------------------------------------------
# Transformation form classification
# ---------------------------------------------------------------------------
# Transformation tokens that render as bare prefixes (<token>_<base>) rather
# than _of_ form (<token>_of_<base>).  The parser treats these as
# qualifiers (not operators) since they lack the _of_ joining marker.
# Determined by corpus analysis of canonical standard names.
_TRANSFORMATION_VALUES: frozenset[str] = frozenset(t.value for t in Transformation)
_DECOMPOSITION_VALUES: frozenset[str] = frozenset(d.value for d in Decomposition)

# Canonical intra-order for locus geometric qualifiers — mirror of
# ``locus_registry.yml`` ``locus_qualifiers``. Kept as a module constant so the
# model layer needs no runtime YAML load; the parser enforces the closed
# vocabulary (only registered qualifiers strip), this enforces canonical order
# and the requires-a-feature rule for directly-constructed models.
_LOCUS_QUALIFIER_ORDER: tuple[str, ...] = (
    "primary",
    "secondary",
    "upper",
    "lower",
    "inner",
    "outer",
)
# The bare-spelling prefix set is declared in the IR layer, which the renderer
# also consults — one list, so the model's transformation slot and the rendered
# spelling cannot drift apart.
_BARE_PREFIX_TRANSFORMATIONS: frozenset[str] = BARE_PREFIX_OPERATORS


# ---------------------------------------------------------------------------
# Indexed-operator token validation
# ---------------------------------------------------------------------------
# An indexed unary operator (``derivative_with_respect_to[coord]``,
# ``bessel_0[order]``, ``fourier_coefficient[m,n]``) binds its index into the
# operator token: the parser fuses ``<op>_<index>`` so the canonical renderer
# reproduces it verbatim (``derivative_with_respect_to_radial_coordinate_of_
# pressure``). The flat StandardName model therefore has to accept the FUSED
# token in its ``transformation`` / ``decomposition`` slot, not only the bare
# registered enum members. The closed StrEnums emitted by the codegen carry
# only the bare tokens, so the slots are validated by token RULE here instead
# of by enum membership — accepting either a bare registered operator OR a
# ``<indexed_op>_<index>`` form whose base operator declares ``index_params``.
#
# The coordinate index of ``derivative_with_respect_to`` is drawn from the same
# coordinate / flux-coordinate universe the parser uses (carriers + axes), so
# the model and parser agree on exactly which fused tokens round-trip. Loading
# is lazy + cached to avoid importing the vocabulary loaders at module import.


class _IndexedOperatorRules:
    """Lazily-loaded vocabulary view for indexed-operator token validation."""

    def __init__(self) -> None:
        self._loaded = False
        self._indexed_coord_prefix_ops: frozenset[str] = frozenset()
        self._coordinate_universe: frozenset[str] = frozenset()

    def _load(self) -> None:
        if self._loaded:
            return
        # Deferred import: the parser pulls in the vocabulary loaders, which we
        # do not want to trigger at module import time (model.py is imported
        # widely, including during codegen-adjacent paths).
        from imas_standard_names.grammar.parser import (  # noqa: PLC0415
            _coordinate_universe,
            load_default_vocabularies,
        )

        vocabs = load_default_vocabularies()
        self._coordinate_universe = frozenset(_coordinate_universe(vocabs))
        # Indexed unary operators whose single index parameter is ``coord`` bind
        # a coordinate token in the fused form ``<op>_<coord>`` (prefix derivative
        # family). Other indexed operators (bessel_*[order], fourier_coefficient
        # [m,n]) bind their index purely positionally and the parser only ever
        # emits their BARE token, so they are already covered by enum membership.
        self._indexed_coord_prefix_ops = frozenset(
            op
            for op, meta in vocabs.operators.items()
            if meta.get("indexed") and list(meta.get("index_params") or []) == ["coord"]
        )
        self._loaded = True

    def coord_index_operators(self) -> frozenset[str]:
        self._load()
        return self._indexed_coord_prefix_ops

    def coordinate_universe(self) -> frozenset[str]:
        self._load()
        return self._coordinate_universe


_INDEXED_OPERATOR_RULES = _IndexedOperatorRules()


def _split_fused_indexed_operator(token: str) -> tuple[str, str] | None:
    """Split ``<op>_<index>`` for a registered coordinate-indexed operator.

    Returns ``(base_op, index)`` when ``token`` is a coordinate-indexed
    operator (``derivative_with_respect_to``) followed by a registered
    coordinate token, else ``None``. The longest base-operator match wins so
    overlapping prefixes never shadow the real operator.
    """
    coord_ops = _INDEXED_OPERATOR_RULES.coord_index_operators()
    coords = _INDEXED_OPERATOR_RULES.coordinate_universe()
    best: tuple[str, str] | None = None
    for op in coord_ops:
        marker = f"{op}_"
        if not token.startswith(marker):
            continue
        index = token[len(marker) :]
        if index not in coords:
            continue
        if best is None or len(op) > len(best[0]):
            best = (op, index)
    return best


def _validate_operator_token(value: str | None, *, registered: frozenset[str]) -> None:
    """Reject a transformation/decomposition token that is neither a registered
    bare operator nor a valid fused ``<indexed_op>_<index>`` form.

    Raises ``ValueError`` with the same guidance shape the closed StrEnum used
    to emit, so callers see a clear allowed-token message.
    """
    if value is None or value in registered:
        return
    if _split_fused_indexed_operator(value) is not None:
        return
    raise ValueError(
        f"Invalid operator token '{value}': expected a registered operator "
        f"token or a fused '<indexed_operator>_<coordinate>' form "
        f"(e.g. 'derivative_with_respect_to_radial_coordinate')."
    )


class NonCanonicalNameError(ValueError):
    """Name parses but its token order is not the canonical form.

    The grammar locks modifier ordering the way English locks adjective
    order: non-canonical order is ungrammatical, never silently reordered.
    ``canonical_form`` carries the unique canonical spelling so downstream
    pipelines can deterministically normalize instead of guessing.
    """

    def __init__(self, name: str, canonical_form: str) -> None:
        super().__init__(
            f"non-canonical token order in '{name}': "
            f"canonical form is '{canonical_form}'"
        )
        self.name = name
        self.canonical_form = canonical_form


# ---------------------------------------------------------------------------
# IR ↔ Model adapters
# ---------------------------------------------------------------------------


def _ir_to_model_dict(ir: StandardNameIR) -> dict[str, str]:
    """Convert a parsed IR to the flat dict consumed by StandardName.model_validate."""
    d: dict[str, str] = {}

    # Check for binary operator (outermost operator with kind=BINARY)
    binary_op: OperatorApplication | None = None
    unary_ops: list[OperatorApplication] = []
    for op in ir.operators:
        if op.kind is OperatorKind.BINARY:
            binary_op = op
        else:
            unary_ops.append(op)

    if binary_op is not None:
        if any(_contains_binary_operator(argument) for argument in binary_op.args):
            raise ValueError(
                "nested binary operator tree is not representable in the "
                "flat StandardName model; use grammar.parser.parse(..., "
                "strict=True) for lossless validation"
            )
        # A flux-surface reduction is the one operator the flat model can carry
        # alongside a binary application: it occupies the free `transformation`
        # slot and always renders OUTERMOST, so no wrap order needs recording.
        # The average of a ratio (<|grad rho|^2/R^2>) is a distinct quantity from
        # the ratio of averages, and only this spelling expresses it.
        reduction_ops, _ = _flux_surface_reduction_vocab()
        reductions = [op for op in unary_ops if op.op in reduction_ops]
        others = [op for op in unary_ops if op.op not in reduction_ops]
        if others or len(reductions) > 1:
            unrepresentable = others or reductions
            raise ValueError(
                "operator(s) "
                f"{', '.join(repr(op.op) for op in unrepresentable)} wrapping "
                f"the binary operator {binary_op.op!r} are not representable in "
                "the flat StandardName model; the IR layer "
                "(grammar.parser.parse / grammar.render.compose) round-trips "
                "nested operator expressions"
            )
        if reductions:
            d["transformation"] = reductions[0].op
        # Binary operator: extract operands from args
        model_op = _BINARY_IR_TO_MODEL.get(binary_op.op, f"{binary_op.op}_of")
        d["binary_operator"] = model_op

        # Render each operand back to its string form (physical_base, secondary_base)
        if len(binary_op.args) == 2:
            d["physical_base"] = _compose_ir(binary_op.args[0])
            right_operand = binary_op.args[1]
            if right_operand.locus is not None:
                right_core = right_operand.model_copy(update={"locus": None})
                d["secondary_base"] = _compose_ir(right_core)
                _apply_locus_and_mechanism(d, right_operand)
            else:
                d["secondary_base"] = _compose_ir(right_operand)
    else:
        # Unary operators
        prefix_ops = [op for op in unary_ops if op.kind is OperatorKind.UNARY_PREFIX]
        postfix_ops = [op for op in unary_ops if op.kind is OperatorKind.UNARY_POSTFIX]
        if len(prefix_ops) > 1 or len(postfix_ops) > 1:
            raise ValueError(
                "ordered unary operator chain is not representable in the "
                "flat StandardName model; use grammar.parser.parse(..., "
                "strict=True) for lossless validation"
            )
        for op in unary_ops:
            if op.kind is OperatorKind.UNARY_PREFIX:
                d["transformation"] = op.op
            elif op.kind is OperatorKind.UNARY_POSTFIX:
                d["decomposition"] = op.op

        # When a single OUTER unary operator wraps an inner expression that
        # itself contains a bare-prefix transformation qualifier (e.g.
        # volume_averaged), the inner qualifier collides with the operator's
        # `transformation`/`decomposition` slot in the flat model and the
        # subject/qualifier split reorders tokens. Mirror the binary-operand
        # treatment: fold the inner expression (projection + qualifiers +
        # base, EXCLUDING the operator, locus and mechanism — which stay on
        # the outer model) into a single physical_base compound string via
        # the canonical renderer, so the operator slot stays free.
        #
        # The flat model has exactly one transformation/decomposition slot
        # and a flat physical_base, so this fold is sound only for a single
        # unary operator wrapping a projection-free inner expression. Inner
        # projections (radial_electric_field) and operator-of-operator nests
        # collapse a token when folded; we leave those alone so the strict
        # lossless-canonical guard rejects them rather than dropping a token.
        if _should_fold_inner_operand(ir, unary_ops):
            inner_ir = StandardNameIR(
                operators=[],
                projection=ir.projection,
                qualifiers=list(ir.qualifiers),
                base=ir.base,
                locus=None,
                mechanism=None,
            )
            d["physical_base"] = _compose_ir(inner_ir)
            return _apply_locus_and_mechanism(d, ir)

        # Projection → component or coordinate
        if ir.projection is not None:
            if ir.projection.shape is ProjectionShape.COMPONENT:
                d["component"] = ir.projection.axis
            elif ir.projection.shape is ProjectionShape.COORDINATE:
                d["coordinate"] = ir.projection.axis

        # Base + qualifiers → physical_base or geometric_base
        if ir.base.kind is BaseKind.GEOMETRY:
            d["geometric_base"] = ir.base.token
            plane_tokens = [
                qualifier.token
                for qualifier in ir.qualifiers
                if qualifier.token in _SECTION_PLANE_VALUES
            ]
            other_tokens = [
                qualifier.token
                for qualifier in ir.qualifiers
                if qualifier.token not in _SECTION_PLANE_VALUES
            ]
            if len(plane_tokens) > 1:
                raise ValueError(
                    "Two 'section_plane' tokens cannot stack in a single name; "
                    "the section_plane segment admits at most one token."
                )
            if other_tokens:
                raise ValueError(
                    "geometry carriers cannot carry non-plane prefix qualifiers; "
                    f"got {other_tokens}"
                )
            if plane_tokens:
                d["section_plane"] = plane_tokens[0]
        else:
            # physical_base: fold uncategorized qualifiers back as prefix
            subject = None
            device = None
            transformation_token = None
            aggregation = None
            section_plane = None
            geometry_representation = None
            population = None
            state = None
            orbit = None
            channel = None
            channel_qualifier = None
            zone_tokens: list[str] = []
            segment_qualifiers: list[str] = []
            base_qualifiers: list[str] = []
            for qi, q in enumerate(ir.qualifiers):
                # Aggregation, orbit, and population are orthogonal single-token
                # modifier segments; they take priority over the subject branch
                # and render before the species. Each contributes AT MOST one
                # token: a second token of the same segment is a hard error
                # (never silent last-wins — that would drop a token from the
                # name, e.g. fast_thermal_ion_density). Aggregation legitimately
                # stacks with population, so total_fast_ion_energy is valid.
                if q.token in _SECTION_PLANE_VALUES:
                    if section_plane is not None:
                        raise ValueError(
                            "Two 'section_plane' tokens cannot stack in a single name; "
                            "the section_plane segment admits at most one token."
                        )
                    section_plane = q.token
                elif q.token in _GEOMETRY_REPRESENTATION_VALUES:
                    if geometry_representation is not None:
                        raise ValueError(
                            "Two 'geometry_representation' tokens cannot stack in a "
                            "single name; the geometry_representation segment admits "
                            "at most one token."
                        )
                    geometry_representation = q.token
                elif q.token in _AGGREGATION_VALUES:
                    if aggregation is not None:
                        msg = (
                            f"Two 'aggregation' tokens ('{aggregation}' and "
                            f"'{q.token}') cannot stack in a single name; the "
                            f"aggregation segment admits at most one token."
                        )
                        raise ValueError(msg)
                    aggregation = q.token
                elif q.token in _ORBIT_VALUES:
                    if orbit is not None:
                        msg = (
                            f"Two 'orbit' tokens ('{orbit}' and '{q.token}') "
                            f"cannot stack in a single name; the orbit segment "
                            f"admits at most one token."
                        )
                        raise ValueError(msg)
                    orbit = q.token
                elif q.token in _POPULATION_VALUES:
                    if population is not None:
                        msg = (
                            f"Two 'population' tokens ('{population}' and "
                            f"'{q.token}') cannot stack in a single name; the "
                            f"population segment admits at most one token."
                        )
                        raise ValueError(msg)
                    population = q.token
                elif q.token in _REACTION_CHANNEL_VALUES and any(
                    later.token in _SUBJECT_VALUES for later in ir.qualifiers[qi + 1 :]
                ):
                    # Reactant pair acting as a reaction-channel qualifier: a
                    # product subject (e.g. neutron) follows, so the pair scopes
                    # the reaction rather than being the species. Renders in the
                    # qualifier segment, before the subject
                    # (deuterium_tritium_neutron_flux). As the sole species token
                    # (no following subject) it falls through to the subject
                    # branch below (deuterium_tritium_density).
                    segment_qualifiers.append(q.token)
                elif q.token in _SUBJECT_VALUES:
                    if subject is not None:
                        msg = (
                            f"Two 'subject' tokens ('{subject}' and "
                            f"'{q.token}') cannot stack in a single name; the "
                            f"subject segment admits at most one species "
                            f"token. Atomic multi-word subjects "
                            f"(deuterium_tritium, runaway_electron, ...) are "
                            f"single tokens and unaffected."
                        )
                        raise ValueError(msg)
                    subject = q.token
                elif q.token in _STATE_VALUES:
                    # State is a single-token subject-refinement segment; a
                    # second state token is a hard error (never silent
                    # last-wins).
                    if state is not None:
                        msg = (
                            f"Two 'state' tokens ('{state}' and "
                            f"'{q.token}') cannot stack in a single name; the "
                            f"state segment admits at most one token."
                        )
                        raise ValueError(msg)
                    state = q.token
                elif q.token in _OBJECT_VALUES:
                    device = q.token
                elif q.token in _ZONE_VALUES:
                    # Zone is multi-token: collect every zone token in PARSE
                    # order here; compose() reconstructs them in the fixed
                    # canonical intra-order, and parse_standard_name rejects a
                    # non-canonical authored order via NonCanonicalNameError.
                    zone_tokens.append(q.token)
                elif q.token in _CHANNEL_VALUES:
                    # Channel is single-token (what is transported): a name
                    # carries at most one transport channel. A second channel
                    # token is a hard error (never silent last-wins — that would
                    # drop a token from the name).
                    if channel is not None:
                        msg = (
                            f"Two 'channel' tokens ('{channel}' and "
                            f"'{q.token}') cannot stack in a single name; the "
                            f"channel segment admits at most one transport "
                            f"channel token."
                        )
                        raise ValueError(msg)
                    channel = q.token
                elif q.token in _CHANNEL_QUALIFIER_VALUES:
                    # Channel-qualifier is single-token (binds to the transport
                    # channel): a name carries at most one. A second token is a
                    # hard error (never silent last-wins — that would drop a
                    # token from the name).
                    if channel_qualifier is not None:
                        msg = (
                            f"Two 'channel_qualifier' tokens "
                            f"('{channel_qualifier}' and '{q.token}') cannot "
                            f"stack in a single name; the channel_qualifier "
                            f"segment admits at most one token."
                        )
                        raise ValueError(msg)
                    channel_qualifier = q.token
                elif (
                    q.token in _BARE_PREFIX_TRANSFORMATIONS
                    and transformation_token is None
                ):
                    transformation_token = q.token
                elif q.token in _SCOPING_QUALIFIER_VALUES:
                    # Phrase-scoping qualifier: a first-class segment token,
                    # NOT part of the base compound. It scopes over the whole
                    # species+channel+base phrase and renders outermost among
                    # the refined qualifiers
                    # (implicit_electron_energy_source_rate,
                    # incident_neutron_fluence). Kind-forming qualifiers fall
                    # through to the base-glue below (ion_atomic_mass,
                    # argon_prefill_count) — that split is the per-token
                    # verdict in scoping_qualifiers.yml.
                    segment_qualifiers.append(q.token)
                else:
                    base_qualifiers.append(q.token)
            if aggregation:
                d["aggregation"] = aggregation
            if section_plane:
                d["section_plane"] = section_plane
            if geometry_representation:
                d["geometry_representation"] = geometry_representation
            if orbit:
                d["orbit"] = orbit
            if population:
                d["population"] = population
            if subject:
                d["subject"] = subject
            if state:
                d["state"] = state
            if device:
                d["device"] = device
            if zone_tokens:
                d["zone"] = tuple(zone_tokens)
            if segment_qualifiers:
                d["qualifier"] = tuple(segment_qualifiers)
            if channel:
                d["channel"] = channel
            if channel_qualifier:
                d["channel_qualifier"] = channel_qualifier
            if transformation_token:
                d["transformation"] = transformation_token
            # Fold remaining qualifiers into physical_base as compound
            if base_qualifiers:
                d["physical_base"] = "_".join([*base_qualifiers, ir.base.token])
            else:
                d["physical_base"] = ir.base.token

    return _apply_locus_and_mechanism(d, ir)


def _contains_binary_operator(ir: StandardNameIR) -> bool:
    """Whether an IR subtree contains any binary operator application."""
    return any(
        operator.kind is OperatorKind.BINARY
        or any(_contains_binary_operator(argument) for argument in operator.args)
        for operator in ir.operators
    )


def _should_fold_inner_operand(
    ir: StandardNameIR, unary_ops: list[OperatorApplication]
) -> bool:
    """Whether to fold a unary operator's inner expression into physical_base.

    The flat :class:`StandardName` model carries a single transformation and
    a single decomposition slot. A name with an OUTER unary operator wrapping
    an inner expression that contains a bare-prefix transformation qualifier
    (``volume_averaged``, ``flux_surface_averaged``, ``normalized``, ...) only
    round-trips when the inner expression is folded into one physical_base
    string instead of being split into colliding model fields.

    The fold is sound only for exactly ONE unary operator wrapping a
    projection-free inner expression. With a projection axis or a second
    structurally-distinct operator the fold collapses a token (the flat model
    cannot represent the nest), so we decline and let the strict
    lossless-canonical guard reject the name rather than drop a token.
    """
    if len(unary_ops) != 1:
        return False
    if ir.projection is not None:
        return False
    if ir.base.kind is BaseKind.GEOMETRY:
        return False
    # Only fold when an inner bare-prefix transformation qualifier would
    # otherwise collide with the operator's transformation slot. Without one,
    # the existing subject/qualifier split already round-trips.
    return any(q.token in _BARE_PREFIX_TRANSFORMATIONS for q in ir.qualifiers)


def _apply_locus_and_mechanism(d: dict[str, str], ir: StandardNameIR) -> dict[str, str]:
    """Project the IR locus and mechanism onto the flat model dict.

    Shared by the standard subject/qualifier path and the folded-operand
    path. STRICT projection: a locus token the model cannot represent is a
    hard error — silently dropping it would lose the entire locus and collide
    with the locus-free name (e.g. safety_factor_at_<unknown> degrading to
    bare safety_factor).
    """
    if ir.locus is not None:
        token = ir.locus.token
        if ir.locus.type == LocusType.POSITION:
            if ir.locus.relation == LocusRelation.OVER:
                if token in _REGION_VALUES:
                    d["region"] = token
                else:
                    msg = (
                        f"locus token '{token}' (relation 'over') is not a "
                        f"registered 'region' segment token; cannot project "
                        f"onto the model without dropping the locus"
                    )
                    raise ValueError(msg)
            elif ir.locus.relation == LocusRelation.AT:
                if token in _POSITION_VALUES:
                    d["position"] = token
                    if ir.locus.value is not None:
                        d["position_value"] = ir.locus.value
                else:
                    msg = (
                        f"locus token '{token}' (relation 'at') is not a "
                        f"registered 'position' segment token; cannot project "
                        f"onto the model without dropping the locus"
                    )
                    raise ValueError(msg)
            elif ir.locus.relation == LocusRelation.ALONG:
                if token in _POSITION_VALUES:
                    d["path"] = token
                else:
                    msg = (
                        f"locus token '{token}' (relation 'along') is not a "
                        f"registered 'position' segment token; cannot project "
                        f"onto the model without dropping the locus"
                    )
                    raise ValueError(msg)
            else:
                if token in _POSITION_VALUES or token in _GEOMETRY_VALUES:
                    d["geometry"] = token
                else:
                    msg = (
                        f"locus token '{token}' (relation 'of') is not a "
                        f"registered 'geometry' segment token; cannot project "
                        f"onto the model without dropping the locus"
                    )
                    raise ValueError(msg)
        elif ir.locus.type == LocusType.REGION:
            if token in _REGION_VALUES:
                d["region"] = token
            else:
                msg = (
                    f"locus token '{token}' is not a registered 'region' "
                    f"segment token; cannot project onto the model without "
                    f"dropping the locus"
                )
                raise ValueError(msg)
        else:
            field_name = _LOCUS_TYPE_TO_FIELD.get(ir.locus.type)
            if field_name is None:
                msg = (
                    f"locus token '{token}' has unmapped locus type "
                    f"'{ir.locus.type.value}'; cannot project onto the model"
                )
                raise ValueError(msg)
            d[field_name] = token

        # Carry composed geometric qualifiers (inner/outer/upper/…) onto the model.
        if ir.locus.qualifiers:
            d["locus_qualifiers"] = ir.locus.qualifiers

    # Mechanism → process
    if ir.mechanism is not None:
        d["process"] = ir.mechanism.token

    return d


def _model_to_ir(model: StandardName) -> StandardNameIR:
    """Convert a StandardName model to IR for rendering."""
    operators: list[OperatorApplication] = []
    projection: AxisProjection | None = None
    qualifiers: list[Qualifier] = []
    locus: LocusRef | None = None
    mechanism: IRProcess | None = None

    # Binary operator
    if model.binary_operator:
        op_model_value = _value_of(model.binary_operator)
        ir_op = _BINARY_MODEL_TO_IR.get(op_model_value, op_model_value)

        # Look up separator from BINARY_OPERATOR_CONNECTORS
        connector = BINARY_OPERATOR_CONNECTORS.get(op_model_value)
        if connector is None:
            raise ValueError(
                f"Unknown binary operator '{op_model_value}': "
                f"no connector defined in BINARY_OPERATOR_CONNECTORS"
            )

        # Build sub-IRs for each operand by re-parsing them
        a_str = model.physical_base or ""
        b_str = _value_of(model.secondary_base) if model.secondary_base else ""
        a_ir = _parse_simple_base(a_str)
        b_ir = _parse_simple_base(b_str)

        # A transformation alongside a binary operator wraps it (the validator
        # admits only flux-surface reductions here), so it goes on the stack
        # FIRST — the operator list is outer-to-inner.
        if model.transformation:
            tf_token = _value_of(model.transformation)
            operators.append(
                OperatorApplication(
                    kind=OperatorKind.UNARY_PREFIX,
                    op=tf_token,
                    bare_prefix=tf_token in _BARE_PREFIX_TRANSFORMATIONS,
                )
            )

        operators.append(
            OperatorApplication(
                kind=OperatorKind.BINARY,
                op=ir_op,
                separator=connector,
                args=[a_ir, b_ir],
            )
        )
        # Binary uses a placeholder base
        base = QuantityOrCarrier(token="placeholder", kind=BaseKind.QUANTITY)
    else:
        # Transformation (unary prefix)
        transformation_qualifier: Qualifier | None = None
        if model.transformation:
            tf_token = _value_of(model.transformation)
            if tf_token in _BARE_PREFIX_TRANSFORMATIONS:
                # Bare-prefix transformations render as qualifiers
                # (will be prepended to base by the renderer)
                transformation_qualifier = Qualifier(token=tf_token)
            else:
                operators.append(
                    OperatorApplication(
                        kind=OperatorKind.UNARY_PREFIX,
                        op=tf_token,
                    )
                )

        # Decomposition (unary postfix)
        if model.decomposition:
            operators.append(
                OperatorApplication(
                    kind=OperatorKind.UNARY_POSTFIX,
                    op=_value_of(model.decomposition),
                )
            )

        # Projection
        if model.component:
            projection = AxisProjection(
                axis=_value_of(model.component), shape=ProjectionShape.COMPONENT
            )
        elif model.coordinate:
            projection = AxisProjection(
                axis=_value_of(model.coordinate), shape=ProjectionShape.COORDINATE
            )

        # Base
        if model.geometric_base:
            base = QuantityOrCarrier(
                token=_value_of(model.geometric_base), kind=BaseKind.GEOMETRY
            )
            if model.section_plane:
                qualifiers.append(
                    Qualifier(
                        token=_value_of(model.section_plane), category="section_plane"
                    )
                )
        elif model.physical_base:
            # Re-parse the physical_base to decompose qualifiers + base
            base, qualifiers = _decompose_physical_base(
                model.physical_base,
                model.subject,
                model.device,
                model.population,
                model.orbit,
                model.aggregation,
                model.section_plane,
                model.geometry_representation,
                model.zone,
                model.channel,
                model.channel_qualifier,
                model.qualifier,
                model.state,
            )
            # Insert bare-prefix transformation qualifier at the front
            # (transformation is outermost: <transform>_<subject>_<base>)
            if transformation_qualifier is not None:
                qualifiers.insert(0, transformation_qualifier)
        else:
            raise ValueError("Either geometric_base or physical_base must be set")

    # Locus — position field uses _at_, geometry field uses _of_, path field
    # uses _along_ for POSITION-type loci. Other fields use their fixed
    # defaults. The position field may carry a numeric parameterization
    # (position_value), rendered as _at_<position>_equal_to_<value>.
    for field_name, (default_relation, locus_type) in _FIELD_TO_LOCUS.items():
        value = getattr(model, field_name, None)
        if value is not None:
            locus = LocusRef(
                relation=default_relation,
                token=_value_of(value),
                qualifiers=tuple(model.locus_qualifiers),
                type=locus_type,
                value=model.position_value if field_name == "position" else None,
            )
            break

    # Mechanism
    if model.process:
        mechanism = IRProcess(token=_value_of(model.process))

    if model.binary_operator and locus is not None:
        binary_index = next(
            index
            for index, operator in enumerate(operators)
            if operator.kind is OperatorKind.BINARY
        )
        binary = operators[binary_index]
        right_operand = binary.args[1]
        if right_operand.locus is not None:
            raise ValueError(
                "a binary expression cannot carry a locus when its final "
                "operand already has one"
            )
        operators[binary_index] = binary.model_copy(
            update={
                "args": [
                    binary.args[0],
                    right_operand.model_copy(update={"locus": locus}),
                ]
            }
        )
        locus = None

    return StandardNameIR(
        operators=operators,
        projection=projection,
        qualifiers=qualifiers,
        base=base,
        locus=locus,
        mechanism=mechanism,
    )


def _parse_simple_base(base_str: str) -> StandardNameIR:
    """Parse a simple base string (e.g. 'electron_temperature') into an IR.

    Used for binary operator operands. Falls back to treating the whole
    string as a literal physical_base if re-parsing fails.
    """
    try:
        result = _parse_ir(base_str)
        return result.ir
    except (ParseError, ValueError):
        # Fallback: treat as a literal physical_base
        return StandardNameIR(
            base=QuantityOrCarrier(token=base_str, kind=BaseKind.QUANTITY)
        )


def _decompose_physical_base(
    physical_base: str,
    subject: Subject | None,
    device: Object | None,
    population: Population | None = None,
    orbit: Orbit | None = None,
    aggregation: Aggregation | None = None,
    section_plane: SectionPlane | None = None,
    geometry_representation: GeometryRepresentation | None = None,
    zone: tuple[Zone, ...] = (),
    channel: Channel | None = None,
    channel_qualifier: ChannelQualifier | None = None,
    segment_qualifiers: tuple[QualifierToken, ...] = (),
    state: State | None = None,
) -> tuple[QuantityOrCarrier, list[Qualifier]]:
    """Decompose a physical_base string into IR base + qualifiers.

    The physical_base may be a compound like 'magnetic_field' or
    'diamagnetic_drift_velocity'. We use the parser to decompose it correctly,
    then prepend aggregation/orbit/population/subject/device/zone as
    qualifiers, then the refined ``segment_qualifiers`` (which scope over the
    whole channel phrase), then the channel-qualifier/channel pair as the
    INNERMOST prefix tokens (just before the base, after any residual
    physical_base qualifiers).
    """
    qualifiers: list[Qualifier] = []

    # Render order outer-to-inner follows English adjective order (Forsyth's
    # royal order; Scontras et al. subjectivity hierarchy — less intrinsic
    # composes further from the noun): aggregation (quantifier), the
    # phrase-scoping qualifiers (classifying adjectives), zone (location/
    # origin), then the species block (orbit, population, subject — the
    # material/type noun adjuncts), the legacy device prefix —
    # <aggregation>_<qualifier...>_<zone...>_<orbit>_<population>_<subject>_
    # <device>_<channel_qualifier>_<channel>_<base>, e.g.
    # total_implicit_core_trapped_fast_ion_energy_source_rate.
    # Zone tokens are emitted in the FIXED canonical intra-order (Zone enum
    # order) regardless of the order they were supplied in, so a non-canonical
    # authored order canonicalizes here and is rejected by parse_standard_name.
    if section_plane:
        qualifiers.append(
            Qualifier(token=_value_of(section_plane), category="section_plane")
        )
    if aggregation:
        qualifiers.append(Qualifier(token=_value_of(aggregation)))

    # Phrase-scoping qualifiers modify the WHOLE species+channel+base phrase
    # (implicit_electron_energy_source_rate = the implicit part of the
    # electron energy source rate; incident_neutron_fluence), so they render
    # before zone and the species block. A multi-token segment is emitted in
    # the canonical qualifier-category intra-order (stable within a category),
    # so a non-canonical authored order canonicalizes here and is rejected by
    # parse_standard_name via the round-trip.
    for seg_q in sorted(
        (_value_of(sq) for sq in segment_qualifiers), key=_qualifier_order
    ):
        qualifiers.append(Qualifier(token=seg_q))

    for zone_token in sorted(
        (_value_of(z) for z in zone), key=lambda t: _ZONE_ORDER.get(t, len(_ZONE_ORDER))
    ):
        qualifiers.append(Qualifier(token=zone_token))
    if orbit:
        qualifiers.append(Qualifier(token=_value_of(orbit)))
    if population:
        qualifiers.append(Qualifier(token=_value_of(population)))
    if subject:
        qualifiers.append(Qualifier(token=_value_of(subject)))
    if state:
        # State binds tighter than the species block — rendered immediately
        # after the subject, before the legacy device prefix and the
        # channel/base: <population>_<subject>_<state>_<channel>_<base>.
        qualifiers.append(Qualifier(token=_value_of(state)))
    if device:
        qualifiers.append(Qualifier(token=_value_of(device)))

    # The channel-qualifier binds to the channel and renders immediately OUTER
    # of it (kinetic_energy_flux = channel_qualifier=kinetic + channel=energy +
    # base=flux); the channel itself renders immediately before the base.
    channel_qualifier_q = (
        Qualifier(token=_value_of(channel_qualifier))
        if channel_qualifier is not None
        else None
    )
    channel_q = Qualifier(token=_value_of(channel)) if channel is not None else None

    # Try to parse the physical_base to extract any embedded qualifiers
    try:
        result = _parse_ir(physical_base)
        # The channel renders OUTER of any residual physical_base-embedded
        # qualifiers but INNER of the refined segment qualifiers: the channel
        # names WHAT is transported and binds tightest to the base — the
        # convection velocity OF momentum (momentum_convection_velocity), the
        # decay length OF heat (heat_decay_length). The channel-qualifier
        # renders OUTER of the channel.
        if channel_qualifier_q is not None:
            qualifiers.append(channel_qualifier_q)
        if channel_q is not None:
            qualifiers.append(channel_q)
        if geometry_representation is not None:
            qualifiers.append(Qualifier(token=_value_of(geometry_representation)))
        qualifiers.extend(result.ir.qualifiers)
        return result.ir.base, qualifiers
    except (ParseError, ValueError):
        # Can't decompose: use the whole string as base. The channel-qualifier
        # and channel render just before the (compound) base.
        if channel_qualifier_q is not None:
            qualifiers.append(channel_qualifier_q)
        if channel_q is not None:
            qualifiers.append(channel_q)
        if geometry_representation is not None:
            qualifiers.append(Qualifier(token=_value_of(geometry_representation)))
        return QuantityOrCarrier(
            token=physical_base, kind=BaseKind.QUANTITY
        ), qualifiers


class StandardName(BaseModel):
    """Structured representation of a standard name."""

    model_config = ConfigDict(extra="forbid")

    component: Component | None = None
    coordinate: Coordinate | None = None
    section_plane: SectionPlane | None = None
    aggregation: Aggregation | None = None
    orbit: Orbit | None = None
    population: Population | None = None
    subject: Subject | None = None
    # State: single-token subject-refinement segment (charge_state on ions,
    # internal_state on neutrals). Binds tighter than population — renders
    # immediately AFTER the subject: <population>_<subject>_<state>_<base>.
    state: State | None = None
    device: Object | None = None
    # Zone: ordered plasma-region / geometric sub-selector PREFIX segment.
    # MULTI-token (lower_outer) unlike the single-token modifier segments, so
    # it is a tuple. Tokens are stored/validated in the FIXED canonical
    # intra-order (Zone enum order); compose() renders them between
    # subject/device and the refined qualifiers.
    zone: tuple[Zone, ...] = ()
    # Qualifier: refined base-phrase qualifier segment (implicit, incident,
    # effective, ...) from the open qualifiers.yml vocabulary. Scopes over the
    # WHOLE channel phrase (English adjective order: the qualifier modifies
    # the compound noun the channel forms with the base), so compose() renders
    # it OUTER of channel_qualifier/channel and INNER of the zone
    # (implicit_energy_source_rate, incident_kinetic_energy_flux_at_wall).
    # MULTI-token like zone, so it is a tuple; compose() emits a stacked
    # segment in canonical qualifier-category order (stable within a category),
    # so a non-canonical authored order canonicalizes and round-trips.
    qualifier: tuple[QualifierToken, ...] = ()
    # ChannelQualifier: qualifier that binds to the transport channel (kinetic,
    # plasma). SINGLE-token; compose() renders it immediately OUTER of the
    # channel (before it) and inner of the refined qualifiers. kinetic also
    # forms the atomic base kinetic_energy; the parser disambiguates by
    # longest-base match.
    channel_qualifier: ChannelQualifier | None = None
    # Channel: transport-channel PREFIX segment (heat/particle/energy/momentum —
    # WHAT is transported). SINGLE-token and INNERMOST: compose() renders it
    # immediately before the base, after the refined qualifier(s) — the
    # qualifier scopes over the whole channel phrase. energy/momentum
    # are also valid bases (standalone electron_energy, kinetic_energy); the
    # parser disambiguates by longest-base match.
    channel: Channel | None = None
    geometry_representation: GeometryRepresentation | None = None
    geometric_base: GeometricBase | None = None
    physical_base: BaseToken | None = None
    object: Object | None = None
    geometry: Position | None = None
    position: Position | None = None
    position_value: str | None = Field(
        default=None,
        description=(
            "Numeric parameterization of the position locus, rendered as "
            "at_<position>_equal_to_<position_value>. Underscores act as "
            "decimal separators (e.g. '0_95' for 0.95). Requires position."
        ),
        examples=["0_95", "1_0", "2"],
    )
    region: Region | None = None
    # Path-like position a quantity varies ALONG (a diagnostic chord or
    # traced trajectory), rendered as along_<token> — distinct from position
    # (a single point sampled AT) and geometry (an intrinsic property OF).
    # Shares the Position enum: the underlying locus_registry token just
    # also declares 'along' in its allowed_relations.
    path: Position | None = None
    # Ordered geometric qualifiers composed onto the locus FEATURE
    # (object/geometry/position/path), e.g. ('inner',) for
    # radial_coordinate_of_inner_strike_point, ('upper','outer') for
    # ...upper_outer_strike_point. Empty for a bare/non-qualifiable feature.
    # render_locus prefixes them onto the feature token; the parser canonicalises
    # order, so a non-canonically-authored name fails the compose round-trip.
    locus_qualifiers: tuple[str, ...] = ()
    process: Process | None = None
    # transformation / decomposition accept the closed registered operator
    # tokens AND fused indexed-operator tokens (``<indexed_op>_<index>``, e.g.
    # ``derivative_with_respect_to_radial_coordinate``). The codegen StrEnums
    # carry only the bare tokens, so these slots are typed ``str`` and validated
    # by the rule in ``_validate_operator_token`` rather than by enum
    # membership — see the indexed-operator note above.
    transformation: str | None = None
    decomposition: str | None = None
    binary_operator: BinaryOperator | None = None
    secondary_base: BaseToken | None = None

    @field_validator("physical_base")
    @classmethod
    def _validate_physical_base(cls, value: str | None) -> str | None:
        if value is not None and not TOKEN_PATTERN.fullmatch(value):
            msg = "physical_base segment must match the canonical token pattern"
            raise ValueError(msg)
        return value

    @field_validator("secondary_base")
    @classmethod
    def _validate_secondary_base(cls, value: str | None) -> str | None:
        if value is not None and not TOKEN_PATTERN.fullmatch(value):
            msg = "secondary_base segment must match the canonical token pattern"
            raise ValueError(msg)
        return value

    @field_validator("transformation", mode="before")
    @classmethod
    def _validate_transformation(cls, value: Any) -> str | None:
        # Accept enum members (StrEnum) by coercing to their .value first.
        token = _value_of(value) if value is not None else None
        _validate_operator_token(token, registered=_TRANSFORMATION_VALUES)
        return token

    @field_validator("decomposition", mode="before")
    @classmethod
    def _validate_decomposition(cls, value: Any) -> str | None:
        token = _value_of(value) if value is not None else None
        _validate_operator_token(token, registered=_DECOMPOSITION_VALUES)
        return token

    @field_validator("position_value")
    @classmethod
    def _validate_position_value(cls, value: str | None) -> str | None:
        if value is not None and not LOCUS_VALUE_PATTERN.fullmatch(value):
            msg = (
                f"position_value {value!r} must be a numeric literal with "
                f"underscores as decimal separators (e.g. '0_95', '1_0', '2')"
            )
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _check_position_value_requires_position(self) -> StandardName:
        if self.position_value is not None and self.position is None:
            msg = "position_value can only be set when position is set"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_exclusive(self) -> StandardName:
        for left, right in EXCLUSIVE_SEGMENT_PAIRS:
            if getattr(self, left, None) and getattr(self, right, None):
                msg = f"Segments '{left}' and '{right}' cannot both be set"
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_base_required(self) -> StandardName:
        if self.geometric_base is None and self.physical_base is None:
            msg = "Either geometric_base or physical_base must be set"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_section_plane_semantics(self) -> StandardName:
        cross_sectional = (
            self.geometric_base is not None
            and _value_of(self.geometric_base) == "cross_section"
        ) or bool(
            self.physical_base
            and (
                self.physical_base == "cross_sectional"
                or self.physical_base.startswith("cross_sectional_")
            )
        )
        if cross_sectional and self.section_plane is None and self.coordinate is None:
            raise ValueError(
                "cross-sectional identities require a section_plane; use "
                "'poloidal_plane' for a tokamak poloidal cross section"
            )
        if self.section_plane is not None and not cross_sectional:
            raise ValueError(
                "section_plane is only valid on a cross-sectional quantity or "
                "cross_section geometry carrier"
            )
        if self.section_plane is not None and (
            self.component is not None or self.coordinate is not None
        ):
            raise ValueError(
                "section_plane cannot be combined with a component or coordinate "
                "projection"
            )
        return self

    @model_validator(mode="after")
    def _check_geometry_representation_semantics(self) -> StandardName:
        if self.geometry_representation is None:
            return self
        if (
            _value_of(self.geometry_representation) != "local_circle"
            or self.physical_base != "radius"
        ):
            raise ValueError(
                "geometry_representation 'local_circle' is valid only with "
                "physical_base 'radius'"
            )
        if self.object is None:
            raise ValueError(
                "geometry_representation 'local_circle' requires an explicit "
                "owner in the object locus"
            )
        if self.component is not None or self.coordinate is not None:
            raise ValueError(
                "a local-circle radius is not a component or coordinate projection"
            )
        return self

    @field_validator("locus_qualifiers", mode="before")
    @classmethod
    def _coerce_locus_qualifiers(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        return tuple(value)

    @model_validator(mode="after")
    def _check_locus_qualifiers(self) -> StandardName:
        if not self.locus_qualifiers:
            return self
        if not (
            self.object or self.geometry or self.position or self.region or self.path
        ):
            raise ValueError(
                "locus_qualifiers require a locus feature "
                "(object / geometry / position / region / path)"
            )
        order = {q: i for i, q in enumerate(_LOCUS_QUALIFIER_ORDER)}
        for q in self.locus_qualifiers:
            if q not in order:
                raise ValueError(
                    f"unknown locus qualifier {q!r}; allowed: {_LOCUS_QUALIFIER_ORDER}"
                )
        idxs = [order[q] for q in self.locus_qualifiers]
        if idxs != sorted(idxs):
            raise ValueError(
                "locus_qualifiers must be in canonical order "
                f"{_LOCUS_QUALIFIER_ORDER}; got {self.locus_qualifiers}"
            )
        return self

    @model_validator(mode="after")
    def _check_major_radius_not_coordinate_of_locus(self) -> StandardName:
        """A point's radial (R) coordinate is ``radial_coordinate_of_<X>``.

        ``major_radius`` is reserved for the bare R0 reference and length/
        operator compounds; it must NOT carry a positional/geometry locus.
        A named feature's R coordinate is the symmetric ``radial_coordinate_of_
        <carrier>`` form (``geometric_base=radial_coordinate``), matching the
        vertical (Z) coordinate form. Rejecting ``major_radius`` + locus keeps
        one canonical spelling for the coordinate-of-a-point concept.
        """
        base = self.physical_base or ""
        if (base == "major_radius" or base.endswith("_major_radius")) and (
            self.object is not None
            or self.geometry is not None
            or self.position is not None
            or self.path is not None
        ):
            carrier = self.object or self.geometry or self.position or self.path
            carrier_val = getattr(carrier, "value", carrier)
            msg = (
                f"'{base}' cannot take a positional/geometry locus; a "
                f"point's radial coordinate is 'radial_coordinate_of_{carrier_val}' "
                "(geometric_base=radial_coordinate, symmetric with "
                "vertical_coordinate). Reserve major_radius for the bare R0 "
                "reference and length/operator compounds."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_time_base_not_process(self) -> StandardName:
        """The bare ``time`` base is the time coordinate / elapsed time.

        ``time_due_to_<process>`` is ambiguous ("time" — delay? constant?
        diffusion time?). A characteristic timescale is a named quantity and
        gets its own atomic base (``resistive_diffusion_time``,
        ``confinement_time``, ``dead_time`` …), so reject a bare ``time`` base
        that carries a ``due_to_`` process and point at the timescale base.
        """
        if self.physical_base == "time" and self.process is not None:
            proc = getattr(self.process, "value", self.process)
            msg = (
                f"bare 'time' base cannot take a due_to_{proc} process; a "
                f"characteristic timescale is a named quantity — use the "
                f"lexicalised base '{proc}_time' (e.g. resistive_diffusion_time). "
                "Reserve 'time' for the time coordinate / elapsed time."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_transformation_exclusivity(self) -> StandardName:
        """Transformation is exclusive with geometric_base only.

        A transformation (prefix operator: ``time_derivative``, ``gradient``,
        ``change_in``, ...) DOES coexist with a component or coordinate
        projection. The two occupy structurally distinct, unambiguous
        positions in the canonical string:

        * an ``_of_``-form transformation renders OUTERMOST, wrapping the
          projection — ``time_derivative_of_toroidal_current_density``
          (``gradient_of_radial_electron_temperature``). The parser peels the
          leading ``<op>_of_`` first, then resolves the projected base, so the
          round-trip is deterministic.
        * a BARE-prefix transformation (``change_in``, ``volume_averaged``,
          ...) folds into the qualifier run, so the projection stays outermost
          — ``poloidal_change_in_ion_velocity``.

        Either way the segments are unambiguous; there is no parsing collision
        to forbid. Only ``geometric_base`` stays exclusive: a geometry carrier
        has no field/operator structure for a transformation to act on, so the
        pairing is unrepresentable.
        """
        if self.transformation and self.geometric_base:
            msg = "Segments 'transformation' and 'geometric_base' cannot both be set"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_decomposition_exclusivity(self) -> StandardName:
        """Decomposition is exclusive with geometric_base.

        A decomposition (postfix operator: ``magnitude``, ``moment``, ...)
        wraps a physical_base; a geometry carrier has no vector/complex
        structure to decompose, so the pairing is unrepresentable.

        Decomposition DOES coexist with a transformation (prefix operator:
        ``maximum``, ``square``, ...). The two occupy structurally distinct,
        unambiguous positions in the canonical string — the prefix renders at
        the front (``<transform>_of_<...>``) and the postfix at the tail
        (``<...>_<decomposition>``), e.g. ``maximum_of_magnetic_field_magnitude``.
        The parser peels the trailing postfix before the leading prefix, so the
        round-trip is deterministic; there is no parsing ambiguity to forbid.
        """
        if self.decomposition and self.geometric_base:
            msg = "Segments 'decomposition' and 'geometric_base' cannot both be set"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_binary_operator_exclusivity(self) -> StandardName:
        """Binary operator is exclusive with component, transformation,
        decomposition, coordinate, subject, device, and geometric_base.

        A flux-surface reduction is the single exception in the transformation
        slot: it renders outermost over the binary application (the average of a
        ratio), so the flat model needs no wrap-order field to reproduce it.
        Every other transformation could sit either side of the operator and is
        therefore still refused.
        """
        if self.binary_operator:
            exclusive_fields = {
                "component": self.component,
                "decomposition": self.decomposition,
                "coordinate": self.coordinate,
                "subject": self.subject,
                "device": self.device,
                "geometric_base": self.geometric_base,
            }
            reduction_ops, _ = _flux_surface_reduction_vocab()
            if self.transformation and _value_of(self.transformation) not in (
                reduction_ops
            ):
                exclusive_fields["transformation"] = self.transformation
            for field_name, field_value in exclusive_fields.items():
                if field_value:
                    msg = (
                        f"Segments 'binary_operator' and '{field_name}' "
                        f"cannot both be set"
                    )
                    raise ValueError(msg)
            if not self.secondary_base:
                msg = "binary_operator requires secondary_base"
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_secondary_base_requires_binary(self) -> StandardName:
        """secondary_base can only be set with binary_operator."""
        if self.secondary_base and not self.binary_operator:
            msg = "secondary_base can only be set when binary_operator is set"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_binary_operator_connector_safety(self) -> StandardName:
        """Validate that binary operator operands don't contain the connector word."""
        if self.binary_operator and self.physical_base and self.secondary_base:
            connector = BINARY_OPERATOR_CONNECTORS.get(self.binary_operator.value, "")
            connector_sep = f"_{connector}_"
            if connector_sep in f"_{self.physical_base}_":
                msg = (
                    f"physical_base '{self.physical_base}' contains reserved "
                    f"connector word '{connector}'"
                )
                raise ValueError(msg)
            if connector_sep in f"_{self.secondary_base}_":
                msg = (
                    f"secondary_base '{self.secondary_base}' contains reserved "
                    f"connector word '{connector}'"
                )
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_generic_physical_base(self, info: ValidationInfo) -> StandardName:
        """Validate that generic physical bases have required qualification.

        Generic physical bases (area, current, power, temperature, voltage, etc.)
        are too generic to stand alone and must be qualified with subject, device,
        object, position, or geometry context.

        Examples of invalid names:
            - current (too generic)
            - temperature (too generic)
            - voltage (too generic)

        Examples of valid names:
            - plasma_current (implicit object context)
            - electron_temperature (subject qualifier)
            - poloidal_field_coil_current (device qualifier)
            - poloidal_magnetic_field_probe_voltage (device qualifier)
            - area_of_flux_loop (object qualifier)
            - pressure_at_magnetic_axis (position qualifier)
            - power_due_to_viscous_heat_flux (process qualifier)
        """
        if self.physical_base and self.physical_base in GENERIC_PHYSICAL_BASES:
            # Check if ANY qualifying segment is present
            # Transformations, binary operators, and processes also qualify generic bases
            has_qualification = any(
                [
                    self.aggregation,
                    self.section_plane,
                    self.qualifier,
                    self.orbit,
                    self.population,
                    self.subject,
                    self.device,
                    self.zone,
                    self.channel_qualifier,
                    self.channel,
                    self.geometry_representation,
                    self.object,
                    self.position,
                    self.geometry,
                    self.region,
                    self.path,
                    self.process,
                    self.transformation,
                    self.decomposition,
                    self.binary_operator,
                    bool(info.context and info.context.get("enclosing_operator")),
                ]
            )

            if not has_qualification:
                msg = (
                    f"Generic physical_base '{self.physical_base}' requires qualification. "
                    f"Generic terms like '{self.physical_base}' are ambiguous without context. "
                    f"Add a qualifying segment: subject (e.g., electron_), device (e.g., flux_loop_), "
                    f"object (e.g., of_flux_loop), position (e.g., at_magnetic_axis), "
                    f"geometry (e.g., of_plasma_boundary), region (e.g., over_halo_region), "
                    f"or path (e.g., along_line_of_sight)."
                )
                raise ValueError(msg)

        return self

    def compose(self) -> str:
        ir = _model_to_ir(self)
        return _compose_ir(ir)

    def model_dump_compact(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for key, value in self.model_dump().items():
            if value is None:
                continue
            # The zone segment is a tuple (multi-token). Omit it when empty and
            # render a non-empty zone as the canonically-ordered token run so
            # the compact dump stays a flat ``dict[str, str]`` like every other
            # segment.
            if key == "zone":
                if not value:
                    continue
                ordered = sorted(
                    (_value_of(z) for z in value),
                    key=lambda t: _ZONE_ORDER.get(t, len(_ZONE_ORDER)),
                )
                out[key] = "_".join(ordered)
                continue
            # The qualifier segment is a tuple (multi-token, authored order).
            # Omit when empty; render as a flat token run otherwise.
            if key == "qualifier":
                if not value:
                    continue
                out[key] = "_".join(_value_of(q) for q in value)
                continue
            # locus_qualifiers is a tuple (multi-token, already canonical). Omit
            # when empty; render as a flat token run otherwise.
            if key == "locus_qualifiers":
                if not value:
                    continue
                out[key] = "_".join(_value_of(q) for q in value)
                continue
            out[key] = _value_of(value)
        return out


@cache
def _flux_surface_reduction_vocab() -> tuple[frozenset[str], frozenset[str]]:
    """(reduction operator tokens, bases/carriers constant on a flux surface)."""
    from imas_standard_names.grammar.vocab_loaders import (  # noqa: PLC0415
        load_geometry_carriers,
        load_operators,
        load_physical_bases,
    )

    ops = frozenset(
        token
        for token, defn in load_operators().operators.items()
        if defn.flux_surface_reduction
    )
    flagged = frozenset(
        token
        for token, defn in load_physical_bases().bases.items()
        if defn.constant_on_flux_surface
    ) | frozenset(
        token
        for token, defn in load_geometry_carriers().carriers.items()
        if defn.constant_on_flux_surface
    )
    return ops, flagged


# Species that carry a charge (ionization) state: the bare ion tokens plus
# every element/isotope species (an element resolved by charge state is the
# impurity-transport customer — argon_charge_state_density, …). Sourced from
# the "Fusion fuel + isotopes" and "Impurity elements" sections of
# subjects.yml; the state-vocab-disjoint test guards these stay valid subjects.
_ION_LIKE_SUBJECTS: frozenset[str] = frozenset(
    {
        "ion",
        "impurity_ion",
        "ion_species",
        "impurity_species",
        # elements + isotopes
        "hydrogen",
        "hydrogenic",
        "deuterium",
        "tritium",
        "helium",
        "helium_3",
        "helium_4",
        "alpha_particle",
        "carbon",
        "nitrogen",
        "neon",
        "argon",
        "boron",
        "beryllium",
        "tungsten",
        "iron",
        "lithium",
        "oxygen",
        "krypton",
        "xenon",
    }
)
_NEUTRAL_LIKE_SUBJECTS: frozenset[str] = frozenset({"neutral", "neutral_species"})
# Which subjects each state token may resolve. internal_state also permits
# ion-like subjects (molecular ions carry internal states — a future-legal
# pairing with no catalog names yet).
_STATE_SUBJECT_COMPAT: dict[str, frozenset[str]] = {
    "charge_state": _ION_LIKE_SUBJECTS,
    "internal_state": _NEUTRAL_LIKE_SUBJECTS | _ION_LIKE_SUBJECTS,
}


def _check_state_gate(model: StandardName) -> None:
    """The state segment requires a compatible species subject.

    A state token resolves a specific state OF a species, so it is meaningless
    without a species subject (reject bare ``charge_state_density``), and the
    token must match the species: ``charge_state`` on ion-like subjects,
    ``internal_state`` on neutral-like (or, as a future pairing, ion-like)
    subjects.
    """
    if model.state is None:
        return
    state_tok = model.state.value
    subject_tok = model.subject.value if model.subject is not None else None
    if subject_tok is None:
        raise ValueError(
            f"state '{state_tok}' requires a species subject — state resolves "
            "a specific state OF a species (e.g. ion_charge_state_density); a "
            "bare state token has no referent"
        )
    allowed = _STATE_SUBJECT_COMPAT.get(state_tok, frozenset())
    if subject_tok not in allowed:
        raise ValueError(
            f"state '{state_tok}' is not valid on subject '{subject_tok}': "
            "charge_state applies to ions/elements, internal_state to neutrals "
            "(and, for molecular ions, to ions)"
        )


def compose_standard_name(parts: Mapping[str, Any] | StandardName) -> str:
    if isinstance(parts, StandardName):
        model = parts
    else:
        model = StandardName.model_validate(parts)
    _check_state_gate(model)
    ir = _model_to_ir(model)
    rendered = _compose_ir(ir)
    _parse_ir(rendered, strict=True)
    return rendered


def parse_standard_name(name: str) -> StandardName:
    """Parse a name into a validated :class:`StandardName`, or raise.

    This convenience facade returns the flat model for names that model can
    represent. Lossless :func:`imas_standard_names.grammar.parser.parse` with
    ``strict=True`` is the authoritative validity oracle, including ordered
    operator chains that cannot project into this flat model.

    Do NOT use :func:`imas_standard_names.grammar.parser.validate_round_trip`
    as a validity check: it operates on the lenient IR parser and is an
    IR-diagnostics tool, not the oracle (see docs/architecture/boundary.md).
    """
    try:
        result = _parse_ir(name, strict=True)
    except ParseError as exc:
        if isinstance(exc, _NonCanonicalParseError):
            raise NonCanonicalNameError(name, exc.canonical_form) from exc
        if exc.residue:
            from imas_standard_names.grammar.parser import (  # noqa: PLC0415
                load_default_vocabularies as _load_vocabs,
            )
            from imas_standard_names.grammar.support import (  # noqa: PLC0415
                UnknownBaseTokenError,
            )

            vocabs = _load_vocabs()
            known = tuple(sorted(vocabs.bases | vocabs.carriers))
            raise UnknownBaseTokenError(exc.residue, known) from exc
        raise
    model = StandardName.model_validate(_ir_to_model_dict(result.ir))
    _check_state_gate(model)
    # Strict canonical-form parsing: the grammar admits exactly ONE spelling
    # per name. A name whose tokens parse but sit in non-canonical order
    # (e.g. fast_trapped_ion_density vs trapped_fast_ion_density) is
    # ungrammatical — raise with the canonical form attached rather than
    # silently reordering on compose. The strict IR oracle has already checked
    # the lossless form; this projection check protects the flat facade.
    canonical = _compose_ir(_model_to_ir(model))
    if canonical != name:
        _assert_lossless_canonical(name, canonical)
        raise NonCanonicalNameError(name, canonical)
    return model


def _assert_lossless_canonical(name: str, canonical: str) -> None:
    """Refuse to offer a canonical_form that lost or gained tokens.

    Downstream pipelines auto-adopt ``NonCanonicalNameError.canonical_form``
    for deterministic normalization, so a projection path that silently drops
    a token (e.g. a last-wins device assignment) must NEVER surface its lossy
    render as the canonical form — that would launder token loss into
    persisted names. A pure reorder has identical underscore-token multisets;
    anything else raises a plain ValueError with NO canonical_form attribute.
    """
    from imas_standard_names.grammar.parser import (  # noqa: PLC0415
        load_default_vocabularies as _load_vocabs,
    )

    vocabs = _load_vocabs()
    alias_map = {**vocabs.base_aliases, **vocabs.carrier_aliases}
    normalized = f"_{name}_"
    for alias, canonical_token in sorted(
        alias_map.items(), key=lambda item: len(item[0]), reverse=True
    ):
        normalized = normalized.replace(f"_{alias}_", f"_{canonical_token}_")

    input_tokens = Counter(normalized.strip("_").split("_"))
    canonical_tokens = Counter(canonical.split("_"))
    if input_tokens == canonical_tokens:
        return
    lost = sorted((input_tokens - canonical_tokens).elements())
    gained = sorted((canonical_tokens - input_tokens).elements())
    detail = []
    if lost:
        detail.append(f"projection lost token(s) {set(lost)}")
    if gained:
        detail.append(f"projection gained token(s) {set(gained)}")
    msg = (
        f"{' and '.join(detail)} for '{name}' — name is ungrammatical "
        f"(no canonical form is offered)"
    )
    raise ValueError(msg)


__all__ = [
    "NonCanonicalNameError",
    "StandardName",
    "compose_standard_name",
    "parse_standard_name",
]
