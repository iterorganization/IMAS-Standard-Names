"""Standard name grammar parser.

Implements a multi-pass parser that turns a standard-name string
into a :class:`~imas_standard_names.grammar.ir.StandardNameIR` plus a
list of :class:`Diagnostic` records. The parser is the inverse of
:func:`imas_standard_names.grammar.render.compose`; together they form
the required round-trip pair. Diagnostic mode is liberal where explicitly
documented; ``strict=True`` validates the lossless ordered expression against
the operator-expression contract without projecting through the flat model.

Parsing is driven by closed vocabularies loaded from
``grammar/vocabularies/*.yml`` via :mod:`vocab_loaders`. Callers may
inject their own :class:`Vocabularies` bundle for testing.

Algorithm::

    1. Normalize operand-first indexed operators
    2. Preserve any tail-final postfix operator for canonical diagnosis
    3. Strip trailing _due_to_<process>                -> mechanism
    4. Resolve a leading binary expression before locus stripping
    5. Strip trailing _of_/_at_/_over_/_along_<locus>  -> locus
       (longest registry-backed match; only _at_ may
       fall back with a vocab_gap diagnostic — _over_
       and _along_ require a registered locus)
    6. Peel unary operators immediately before an of/at/due_to tail
    7. Peel outer operators right-to-outermost         -> operators
       a) unary_postfix (longest match at end)
       b) unary_prefix  (longest match `<op>_of_...`)
       c) bare prefix over a nested operator expression
       d) binary        (`<binary_op>_of_<A>_<sep>_<B>`)
       repeat until no operator peels
    8. Match residue: carrier > base > axis+resolve > qualifier+recurse
       Projection is detected inline when an axis prefix precedes
       a resolvable base (COMPONENT) or carrier (COORDINATE).
       Short form only — ``_component_of_`` and ``_coordinate_of_``
       markers are parse errors.

Liberal acceptance: the parser accepts grammatically valid forms
only. Unknown base residues raise :class:`ParseError` with top-3
edit-distance suggestions. No legacy open-fallback behaviour is retained.
"""

import re
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from difflib import get_close_matches
from functools import cache
from typing import Any

from imas_standard_names.grammar import vocab_loaders
from imas_standard_names.grammar.constants import GENERIC_PHYSICAL_BASES
from imas_standard_names.grammar.ir import (
    BARE_PREFIX_OPERATORS,
    TOKEN_PATTERN,
    AxisProjection,
    BaseKind,
    LocusRef,
    LocusRelation,
    LocusType,
    OperatorApplication,
    OperatorKind,
    Process,
    ProjectionShape,
    Qualifier,
    QuantityOrCarrier,
    StandardNameIR,
)
from imas_standard_names.grammar.model_types import Component, Object, Subject
from imas_standard_names.grammar.operator_semantics import get_operator_semantics
from imas_standard_names.grammar.render import compose

__all__ = [
    "Diagnostic",
    "ParseError",
    "ParseResult",
    "Vocabularies",
    "load_default_vocabularies",
    "parse",
    "validate_round_trip",
]


# ---------------------------------------------------------------------------
# Vocabulary bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Vocabularies:
    """Closed-vocabulary bundle consumed by the parser.

    All fields are immutable collections. Callers may synthesise a bundle
    directly for tests (bypassing YAML loading); the parser never
    introspects loader-level types.
    """

    axes: frozenset[str] = field(default_factory=frozenset)
    component_axes: frozenset[str] = field(default_factory=frozenset)
    loci: Mapping[str, tuple[LocusType, frozenset[LocusRelation]]] = field(
        default_factory=dict
    )
    operators: Mapping[str, dict[str, Any]] = field(default_factory=dict)
    bases: frozenset[str] = field(default_factory=frozenset)
    carriers: frozenset[str] = field(default_factory=frozenset)
    base_aliases: Mapping[str, str] = field(default_factory=dict)
    carrier_aliases: Mapping[str, str] = field(default_factory=dict)
    base_kinds: Mapping[str, str] = field(default_factory=dict)
    flux_function_bases: frozenset[str] = field(default_factory=frozenset)
    extremum_infix_bases: frozenset[str] = field(default_factory=frozenset)
    qualifiers: frozenset[str] = field(default_factory=frozenset)
    # token → normalized category for the genuine modifier qualifiers
    # (qualifiers.yml). Empty for tokens that only peel as qualifiers via the
    # acceptance union (operators, loci, subjects); IR metadata only.
    qualifier_categories: Mapping[str, str] = field(default_factory=dict)
    # token → grammar segment roles. This retains the closed qualifier
    # categories needed to prove binary shared-base elisions without treating
    # an arbitrary run of individually known words as a valid qualifier.
    qualifier_roles: Mapping[str, frozenset[str]] = field(default_factory=dict)
    section_planes: Mapping[str, str] = field(default_factory=dict)
    geometry_representations: frozenset[str] = field(default_factory=frozenset)
    # Ordered geometric qualifiers (canonical intra-order) that compose onto a
    # ``qualifiable`` locus feature (inner_strike_point, upper_outer_strike_point).
    locus_qualifiers: tuple[str, ...] = ()
    # Locus feature tokens that admit ``locus_qualifiers`` prefixes.
    qualifiable_loci: frozenset[str] = field(default_factory=frozenset)

    def base_universe(self) -> frozenset[str]:
        return self.bases | self.carriers

    def closed_universe(self) -> frozenset[str]:
        return (
            self.bases
            | self.carriers
            | self.qualifiers
            | frozenset(self.axes)
            | self.component_axes
            | frozenset(self.operators)
            | frozenset(self.loci)
        )


def _normalise_separator(sep: str | None) -> str | None:
    if sep is None:
        return None
    return sep.strip("_") or None


def load_default_vocabularies() -> Vocabularies:
    """Load all five grammar vocabularies from YAML into a :class:`Vocabularies`.

    Falls back to an empty set for any registry whose YAML stub is empty
    (physical_bases.yml, geometry_carriers.yml).

    Qualifiers are populated from:
    - ``Subject`` enum tokens (electron, ion, deuterium, …)
    - Physics modifier tokens (energy, particle, momentum, …) that act as
      recursive prefixes before a physical_base.
    """
    axes_reg = vocab_loaders.load_coordinate_axes()
    loci_reg = vocab_loaders.load_locus_registry()
    ops_reg = vocab_loaders.load_operators()
    bases_reg = vocab_loaders.load_physical_bases()
    carriers_reg = vocab_loaders.load_geometry_carriers()

    loci: dict[str, tuple[LocusType, frozenset[LocusRelation]]] = {}
    qualifiable_loci_set: set[str] = set()
    for token, entry in loci_reg.loci.items():
        locus_type = LocusType(entry.type)
        allowed = frozenset(LocusRelation(r) for r in entry.allowed_relations)
        loci[token] = (locus_type, allowed)
        if entry.qualifiable:
            qualifiable_loci_set.add(token)

    operators: dict[str, dict[str, Any]] = {}
    for token, entry in ops_reg.operators.items():
        operators[token] = {
            "kind": entry.kind,
            "precedence": entry.precedence,
            "separator": _normalise_separator(entry.separator),
            "indexed": entry.indexed,
            "index_params": entry.index_params,
            "returns": entry.returns,
            "arg_types": entry.arg_types,
            "flux_surface_reduction": entry.flux_surface_reduction,
            "extremum_reduction": entry.extremum_reduction,
        }

    # Build qualifier set: Subject tokens + Object tokens + YAML-loaded
    # modifier prefixes.  Tokens that are also in bases/carriers are safe —
    # the parser tries full base match first; qualifiers only strip
    # recursively when the full string is not itself a registered base or
    # carrier.
    subject_quals = frozenset(s.value for s in Subject)
    object_quals = frozenset(o.value for o in Object)
    modifier_quals = vocab_loaders.load_qualifiers()
    # Aggregation (total/net) + population (energy-state) + orbit (transit
    # class) modifiers peel like qualifiers; the StandardName model retains
    # them in the dedicated ``aggregation`` / ``population`` / ``orbit``
    # single-token segments.
    aggregation_quals = vocab_loaders.load_aggregations()
    population_quals = vocab_loaders.load_populations()
    orbit_quals = vocab_loaders.load_orbits()
    # State-resolution tokens (charge_state, internal_state) peel like
    # qualifiers; the model retains the single token in the ``state`` segment.
    state_quals = vocab_loaders.load_states()
    section_plane_values = vocab_loaders.load_section_planes()
    section_plane_quals = {f"{plane}_plane": plane for plane in section_plane_values}
    representation_quals = vocab_loaders.load_geometry_representations()

    # Add unary_prefix operator tokens as qualifiers so that "bare" prefix
    # operators (those that attach without _of_, like volume_averaged,
    # normalized, flux_surface_averaged) can be stripped during qualifier
    # matching.  Operators that DO use _of_ form are peeled first by
    # _peel_outer_operator and never reach the qualifier stage.
    prefix_op_quals = frozenset(
        name
        for name, meta in operators.items()
        if meta.get("kind") == OperatorKind.UNARY_PREFIX.value
    )

    # Zone tokens (core, edge, inner, outer, lower, ...) are an ordered prefix
    # segment; they peel like qualifiers and the model retains them in the
    # ``zone`` segment.
    zone_quals = frozenset(vocab_loaders.load_zones())

    # Channel tokens (heat, particle, energy, momentum) name what is
    # transported. They peel like qualifiers (innermost prefix, just before the
    # base); the model retains the single token in the ``channel`` segment.
    # energy/momentum are also bases — the parser tries the full base match
    # first, so standalone energy/momentum resolve as base and only the
    # *_flux/*_diffusivity/... compounds strip the channel.
    channel_quals = frozenset(vocab_loaders.load_channels())

    # Channel-qualifier tokens (kinetic, plasma) bind to the transport channel.
    # They peel like qualifiers (outer of the channel, inner of the zone); the
    # model retains the single token in the ``channel_qualifier`` segment.
    # kinetic also forms the atomic base kinetic_energy — the parser tries the
    # longest base match first, so electron_kinetic_energy resolves as the base
    # while ion_kinetic_energy_flux strips channel_qualifier=kinetic.
    channel_qualifier_quals = frozenset(vocab_loaders.load_channel_qualifiers())

    qualifiers = (
        subject_quals
        | object_quals
        | modifier_quals
        | aggregation_quals
        | population_quals
        | orbit_quals
        | state_quals
        | section_plane_quals.keys()
        | representation_quals
        | prefix_op_quals
        | zone_quals
        | channel_quals
        | channel_qualifier_quals
    )
    qualifier_roles: dict[str, set[str]] = {}

    def register_role(tokens: frozenset[str], role: str) -> None:
        for token in tokens:
            qualifier_roles.setdefault(token, set()).add(role)

    register_role(subject_quals, "subject")
    register_role(object_quals, "object")
    register_role(aggregation_quals, "aggregation")
    register_role(population_quals, "population")
    register_role(orbit_quals, "orbit")
    register_role(state_quals, "state")
    register_role(frozenset(section_plane_quals), "section_plane")
    register_role(representation_quals, "geometry_representation")
    register_role(zone_quals, "zone")
    register_role(channel_quals, "channel")
    register_role(channel_qualifier_quals, "channel_qualifier")
    modifier_categories = vocab_loaders.load_qualifier_categories()
    for token, category in modifier_categories.items():
        qualifier_roles.setdefault(token, set()).add(f"qualifier:{category}")

    return Vocabularies(
        axes=frozenset(axes_reg.axes),
        component_axes=frozenset(member.value for member in Component),
        loci=loci,
        operators=operators,
        bases=frozenset(bases_reg.bases),
        carriers=frozenset(carriers_reg.carriers),
        base_aliases={
            alias: token
            for token, definition in bases_reg.bases.items()
            for alias in definition.aliases
        },
        carrier_aliases={
            alias: token
            for token, definition in carriers_reg.carriers.items()
            for alias in definition.aliases
        },
        base_kinds={
            token: definition.kind for token, definition in bases_reg.bases.items()
        }
        | dict.fromkeys(carriers_reg.carriers, "geometry"),
        flux_function_bases=frozenset(
            token
            for token, definition in bases_reg.bases.items()
            if definition.constant_on_flux_surface
        )
        | frozenset(
            token
            for token, definition in carriers_reg.carriers.items()
            if definition.constant_on_flux_surface
        ),
        extremum_infix_bases=frozenset(
            token
            for token, definition in bases_reg.bases.items()
            if definition.extremum_is_transformation
        ),
        qualifiers=qualifiers,
        qualifier_categories=modifier_categories,
        qualifier_roles={
            token: frozenset(roles) for token, roles in qualifier_roles.items()
        },
        section_planes=section_plane_quals,
        geometry_representations=representation_quals,
        locus_qualifiers=tuple(loci_reg.locus_qualifiers),
        qualifiable_loci=frozenset(qualifiable_loci_set),
    )


_DEFAULT_CACHE: Vocabularies | None = None


def _default_vocabs() -> Vocabularies:
    global _DEFAULT_CACHE
    if _DEFAULT_CACHE is None:
        _DEFAULT_CACHE = load_default_vocabularies()
    return _DEFAULT_CACHE


# ---------------------------------------------------------------------------
# Diagnostics / result types
# ---------------------------------------------------------------------------


@dataclass
class Diagnostic:
    """A single parser/validator diagnostic entry.

    Contract: ``category`` is one of
    ``"non_canonical"``, ``"vocab_gap"``, or ``"ambiguity"``; ``layer`` is
    ``"parser"`` or ``"validator"``; ``severity`` is ``"info"``,
    ``"warning"``, or ``"error"``.
    """

    category: str
    layer: str
    message: str
    suggestion: str | None = None
    severity: str = "info"


@dataclass
class ParseResult:
    ir: StandardNameIR
    diagnostics: list[Diagnostic] = field(default_factory=list)


class ParseError(ValueError):
    """Raised when the parser cannot produce a valid IR."""

    def __init__(
        self,
        message: str,
        *,
        suggestions: list[str] | None = None,
        residue: str | None = None,
    ) -> None:
        super().__init__(message)
        self.suggestions: list[str] = list(suggestions or [])
        self.residue: str | None = residue


class _NonCanonicalParseError(ParseError):
    """Strict rejection carrying the unique canonical flat spelling."""

    def __init__(self, name: str, canonical_form: str) -> None:
        super().__init__(
            f"name is not canonical: flat segment order renders as {canonical_form!r}"
        )
        self.name = name
        self.canonical_form = canonical_form


@dataclass(frozen=True)
class _CachedParseError:
    message: str
    suggestions: tuple[str, ...]
    residue: str | None
    name: str | None = None
    canonical_form: str | None = None

    @classmethod
    def from_exception(cls, error: ParseError) -> "_CachedParseError":
        return cls(
            str(error),
            tuple(error.suggestions),
            error.residue,
            getattr(error, "name", None),
            getattr(error, "canonical_form", None),
        )

    def raise_fresh(self) -> None:
        if self.name is not None and self.canonical_form is not None:
            raise _NonCanonicalParseError(self.name, self.canonical_form)
        raise ParseError(
            self.message,
            suggestions=list(self.suggestions),
            residue=self.residue,
        )


@dataclass
class _ParseContext:
    vocabs: Vocabularies
    cache: dict[tuple[str, bool], ParseResult | _CachedParseError] = field(
        default_factory=dict
    )


_ACTIVE_PARSE_CONTEXT: ContextVar[_ParseContext | None] = ContextVar(
    "_ACTIVE_PARSE_CONTEXT", default=None
)


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------


def _strip_mechanism(s: str) -> tuple[Process | None, str]:
    marker = "_due_to_"
    idx = s.rfind(marker)
    if idx <= 0:
        return None, s
    token = s[idx + len(marker) :]
    if not token or not TOKEN_PATTERN.match(token):
        return None, s
    head = s[:idx]
    if not head:
        return None, s
    return Process(token=token), head


# Value-parameterized at-locus: ``at_<token>_equal_to_<value>`` where <value>
# is a numeric literal with underscores as decimal separators (0_95, 1_0, 2).
_LOCUS_VALUE_SUFFIX = re.compile(
    r"^(?P<head>[a-z][a-z0-9_]*)_equal_to_(?P<value>\d+(?:_\d+)?)$"
)


def _match_locus_feature(
    token: str, v: Vocabularies
) -> tuple[str, tuple[str, ...], LocusType, frozenset[LocusRelation]] | None:
    """Resolve a locus token to ``(feature, qualifiers, type, relations)``.

    Direct registry hit first (bare feature or non-qualifiable flat token); then
    the compositional form — strip leading geometric qualifiers off a
    ``qualifiable`` feature (``inner_strike_point`` -> ``strike_point`` + ``inner``;
    ``upper_outer_strike_point`` -> ``strike_point`` + ``upper``, ``outer``).
    Qualifiers are returned in canonical intra-order; a non-canonically-authored
    input therefore fails the compose round-trip and is rejected as non-canonical.
    """
    if token in v.loci:
        lt, allowed = v.loci[token]
        return token, (), lt, allowed
    if not v.locus_qualifiers or not v.qualifiable_loci:
        return None
    qset = set(v.locus_qualifiers)
    order = {q: i for i, q in enumerate(v.locus_qualifiers)}
    quals: list[str] = []
    rest = token
    while "_" in rest:
        head, _, tail = rest.partition("_")
        if head not in qset:
            break
        quals.append(head)
        rest = tail
        if rest in v.qualifiable_loci:
            lt, allowed = v.loci[rest]
            canon = tuple(sorted(quals, key=lambda q: order[q]))
            return rest, canon, lt, allowed
    return None


def _strip_locus(
    s: str, v: Vocabularies
) -> tuple[LocusRef | None, str, list[Diagnostic]]:
    """Strip a trailing locus suffix.

    Preference order: rightmost registry-backed ``_<rel>_<token>`` match,
    including the value-parameterized form ``_at_<token>_equal_to_<value>``
    (the ``_equal_to_<value>`` suffix is split off BEFORE registry lookup;
    only position-typed registry tokens admit a value). ``_at_`` has no
    operator collisions, so an unregistered token still strips with a
    ``vocab_gap`` diagnostic. ``_over_`` and ``_along_`` require a
    registered locus — an unregistered token is left in the residue so the
    unknown-base match rejects the name rather than fabricating a locus.
    ``_of_`` without a registry hit is left for operator peeling
    can resolve it as a binary-operator template.
    """

    diagnostics: list[Diagnostic] = []

    # 1. Registry-backed rightmost match (direct feature or composed
    #    <qualifier>..._<feature>, e.g. inner_strike_point).
    best: tuple[str, int, str, str | None, tuple[str, ...]] | None = None
    for rel in ("over", "at", "along", "of"):
        marker = f"_{rel}_"
        idx = s.rfind(marker)
        while idx > 0:
            token = s[idx + len(marker) :]
            m = _match_locus_feature(token, v) if token else None
            if m is not None:
                if best is None or idx > best[1]:
                    best = (rel, idx, m[0], None, m[1])
                break
            # Value-parameterized position: at_<token>_equal_to_<value>.
            # Split the value suffix BEFORE the registry lookup; only
            # position-typed tokens admit a value (relation 'at').
            if rel == "at" and token:
                value_match = _LOCUS_VALUE_SUFFIX.match(token)
                if value_match:
                    hm = _match_locus_feature(value_match.group("head"), v)
                    if hm is not None and hm[2] is LocusType.POSITION:
                        if best is None or idx > best[1]:
                            best = (
                                rel,
                                idx,
                                hm[0],
                                value_match.group("value"),
                                hm[1],
                            )
                        break
            idx = s.rfind(marker, 0, idx)

    if best is not None:
        rel_str, idx, token, value, quals = best
        locus_type, allowed = v.loci[token]
        relation = LocusRelation(rel_str)
        if relation not in allowed:
            allowed_names = sorted(r.value for r in allowed)
            diagnostics.append(
                Diagnostic(
                    category="non_canonical",
                    layer="parser",
                    message=(
                        f"relation '_{rel_str}_' not permitted for locus "
                        f"{token!r} (type={locus_type.value}); "
                        f"allowed: {allowed_names}"
                    ),
                    severity="warning",
                )
            )
            return None, s, diagnostics
        locus = LocusRef(
            relation=relation,
            token=token,
            qualifiers=quals,
            type=locus_type,
            value=value,
        )
        return locus, s[:idx], diagnostics

    # 2. Unregistered-but-unambiguous fallback for _at_ only.
    #    Skip if the core that would remain is a known qualifier or operator
    #    token — that indicates the _at_ is part of a compound token, not a
    #    locus marker (e.g. maximum_over_flux_surface for the analogous _over_
    #    case).
    #
    #    The _over_ relation does not take this fallback: it is valid solely
    #    for region-typed loci (the locus_registry compatibility matrix), and
    #    those are matched by the registry-backed pass. An unregistered
    #    _over_<X> would otherwise
    #    fabricate a spurious region locus (e.g. velocity_over_magnetic_field),
    #    masking the correct construction (ratio_of_velocity_to_magnetic_field).
    #    Leaving it in the residue makes the base match fail → ParseError.
    for rel, default_type in (("at", LocusType.POSITION),):
        marker = f"_{rel}_"
        idx = s.rfind(marker)
        if idx <= 0:
            continue
        token = s[idx + len(marker) :]
        if not token or not TOKEN_PATTERN.match(token):
            continue
        core = s[:idx]
        # Check if the whole string up to and including the marker token
        # is a registered qualifier/operator (e.g. "maximum_over_flux_surface")
        if any(
            q.startswith(core + marker.rstrip("_"))
            for q in v.qualifiers
            if len(q) > len(core)
        ):
            continue
        try:
            locus = LocusRef(
                relation=LocusRelation(rel),
                token=token,
                type=default_type,
            )
        except Exception:
            continue
        diagnostics.append(
            Diagnostic(
                category="vocab_gap",
                layer="parser",
                message=(
                    f"locus token {token!r} not in locus_registry "
                    f"(defaulted type={default_type.value})"
                ),
                severity="info",
            )
        )
        return locus, s[:idx], diagnostics

    return None, s, diagnostics


def _longest_match(s: str, candidates: frozenset[str] | set[str]) -> str | None:
    """Return the longest candidate in ``candidates`` that matches.

    ``candidates`` should be raw tokens; ``s`` is the full string we are
    searching. This helper is used for operator detection where we match
    against an exact equality family, not a prefix/suffix — callers
    compose the full boundary marker themselves.
    """
    best: str | None = None
    for token in candidates:
        if token == s and (best is None or len(token) > len(best)):
            best = token
    return best


def _peel_outer_operator(
    s: str,
    v: Vocabularies,
    diagnostics: list[Diagnostic] | None = None,
) -> tuple[OperatorApplication | None, str, list[StandardNameIR]]:
    """Peel ONE outer operator off ``s``.

    Returns (op_application, new_inner_string_if_unary, binary_args).
    For unary operators, ``binary_args`` is empty and the caller keeps
    parsing ``new_inner_string_if_unary``. For binary operators the inner
    string is empty and ``binary_args`` holds the two parsed sub-IRs; the
    caller attaches them to the op_application and stops operator peeling
    (a binary operator has no further prefix/postfix beyond its args).
    """

    # A complete registered base/qualifier/projection expression has priority
    # over every operator-shaped substring inside it. This protects atomic
    # suffixes such as magnetic_moment and compound axes such as
    # normalized_toroidal from reinterpretation as operators.
    if _resolves_without_postfix(s, v):
        return None, s, []

    # Split operators by kind.
    postfix_ops = {
        name
        for name, meta in v.operators.items()
        if meta.get("kind") == OperatorKind.UNARY_POSTFIX.value
    }
    prefix_ops = {
        name
        for name, meta in v.operators.items()
        if meta.get("kind") == OperatorKind.UNARY_PREFIX.value
    }
    binary_ops = {
        name
        for name, meta in v.operators.items()
        if meta.get("kind") == OperatorKind.BINARY.value
    }

    # a) unary postfix: s ends with "_<op>", longest op first. A postfix at the
    # tail of an explicit prefix or binary form belongs to that operator's
    # operand, so leave it for the operand's own parse rather than hoisting it
    # outside the leading application.
    postfix_match = (
        None
        if _resolves_without_postfix(s, v)
        else _longest_suffix_match(s, postfix_ops)
    )
    if postfix_match is not None and _postfix_belongs_to_an_operator_operand(s, v):
        postfix_match = None
    if postfix_match is not None:
        new_s = s[: -len(postfix_match) - 1]  # drop "_<op>"
        if new_s:
            return (
                OperatorApplication(kind=OperatorKind.UNARY_POSTFIX, op=postfix_match),
                new_s,
                [],
            )

    # b) unary prefix: s starts with "<op>_of_"
    prefix_match = _longest_prefix_operator_match(s, prefix_ops)
    if prefix_match is not None:
        new_s = s[len(prefix_match) + len("_of_") :]
        if new_s:
            return (
                OperatorApplication(kind=OperatorKind.UNARY_PREFIX, op=prefix_match),
                new_s,
                [],
            )

    # b1) indexed unary prefix: s starts with "<op>_<coord>_of_" where <op> is
    # an indexed prefix operator (index_params) and <coord> is a registered
    # coordinate token. The bound index is fused into the operator token
    # (<op>_<coord>) so the canonical renderer reproduces the prefix form
    # "<op>_<coord>_of_<inner>" verbatim.
    indexed_match = _longest_indexed_prefix_operator_match(s, prefix_ops, v)
    if indexed_match is not None:
        fused_op, consumed_len = indexed_match
        new_s = s[consumed_len + len("_of_") :]
        if new_s:
            return (
                OperatorApplication(kind=OperatorKind.UNARY_PREFIX, op=fused_op),
                new_s,
                [],
            )

    # b2) bare unary prefix: These operators (normalized, volume_averaged, etc.)
    # fall through to the qualifier + base matching stage and are handled
    # by the IR→Model adapter in model.py. We do not peel them here because
    # they can form compound axes (e.g. normalized_radial) that projection
    # stripping needs to see intact.

    # b3) bare unary prefix wrapping an explicit operator expression:
    # "<bare_prefix_op>_<unary_or_binary_application>". The fall-through in b2
    # relies on qualifier + base matching, but an operator application is not a
    # base. Peeling keeps the outer reduction first-class and preserves the
    # nested operator order; bare_prefix reproduces the joiner-free spelling.
    bare_over_operator = _longest_bare_prefix_over_operator_match(
        s, prefix_ops, binary_ops, v
    )
    if bare_over_operator is not None:
        return (
            OperatorApplication(
                kind=OperatorKind.UNARY_PREFIX,
                op=bare_over_operator,
                bare_prefix=True,
            ),
            s[len(bare_over_operator) + 1 :],
            [],
        )

    binary = _peel_binary_operator(s, v, diagnostics)
    if binary is not None:
        return binary

    return None, s, []


def _peel_binary_operator(
    s: str,
    v: Vocabularies,
    diagnostics: list[Diagnostic] | None = None,
) -> tuple[OperatorApplication, str, list[StandardNameIR]] | None:
    """Resolve one leading binary application, including recursive operands.

    Binary operands own their complete recursive expression, including a
    terminal locus.  Resolving this form before the enclosing parser strips a
    locus prevents the final operand's suffix from being hoisted onto the
    binary expression, whose surface spelling cannot preserve that distinction.
    """

    binary_ops = _binary_operator_tokens(v)
    for op in sorted(binary_ops, key=len, reverse=True):
        prefix = f"{op}_of_"
        if not s.startswith(prefix):
            continue
        rest = s[len(prefix) :]
        sep = v.operators[op].get("separator")
        if sep is None:
            continue
        sep_marker = f"_{sep}_"
        # Collect rightmost-first candidates, then prefer the first split
        # whose operands both resolve strictly. A connector word may occur
        # inside a registered operand (for example signal_to_noise_ratio);
        # accepting literal fallbacks at the first split would cut that base
        # apart before the parser reaches its registered boundary.
        candidates: list[tuple[str, str]] = []
        sep_idx = rest.rfind(sep_marker)
        while sep_idx > 0:
            a_str = rest[:sep_idx]
            b_str = rest[sep_idx + len(sep_marker) :]
            if not a_str or not b_str:
                break
            candidates.append((a_str, b_str))
            sep_idx = rest.rfind(sep_marker, 0, sep_idx)

        for a_str, b_str in candidates:
            try:
                a_ir = parse(a_str, vocabs=v).ir
                b_ir = parse(b_str, vocabs=v).ir
            except ParseError:
                continue
            if a_ir is not None and b_ir is not None:
                return (
                    OperatorApplication(
                        kind=OperatorKind.BINARY,
                        op=op,
                        separator=sep,
                        args=[a_ir, b_ir],
                    ),
                    "",
                    [a_ir, b_ir],
                )

        # No fully registered split resolved. Retain the liberal IR fallback
        # for diagnostics; the strict validity oracle gates its vocabulary.
        for a_str, b_str in candidates:
            a_ir = _try_parse_or_literal(a_str, v, diagnostics)
            b_ir = _try_parse_or_literal(b_str, v, diagnostics)
            if a_ir is not None and b_ir is not None:
                return (
                    OperatorApplication(
                        kind=OperatorKind.BINARY,
                        op=op,
                        separator=sep,
                        args=[a_ir, b_ir],
                    ),
                    "",
                    [a_ir, b_ir],
                )

    return None


def _peel_leading_binary_expression(
    s: str,
    v: Vocabularies,
    diagnostics: list[Diagnostic] | None = None,
) -> list[OperatorApplication] | None:
    """Resolve a binary application and any explicit unary wrappers.

    The returned stack is outermost first. A unary-only expression returns
    ``None`` so its trailing decorators continue through the flat parser path.
    """

    operators: list[OperatorApplication] = []
    residue = s
    while True:
        operator, new_residue, _ = _peel_outer_operator(residue, v, diagnostics)
        if operator is None:
            return None
        operators.append(operator)
        if operator.kind is OperatorKind.BINARY:
            return operators
        residue = new_residue


def _peel_trailing_postfix_operator(
    s: str, v: Vocabularies
) -> tuple[OperatorApplication | None, str]:
    """Peel ONE trailing unary-postfix operator off the END of ``s``.

    A postfix decomposition operator renders at the very tail of the canonical
    string (``{core}_{op}``), after any locus/mechanism suffix. Peeling it
    before the mechanism/locus strips keeps those strips from greedily
    absorbing the operator token into a fabricated process/locus token.

    Returns ``(None, s)`` when no postfix operator suffix is present.
    """
    if _resolves_without_postfix(s, v):
        return None, s

    postfix_ops = {
        name
        for name, meta in v.operators.items()
        if meta.get("kind") == OperatorKind.UNARY_POSTFIX.value
    }
    match = _longest_suffix_match(s, postfix_ops)
    if match is None:
        return None, s
    new_s = s[: -len(match) - 1]  # drop "_<op>"
    if not new_s:
        return None, s
    return (
        OperatorApplication(kind=OperatorKind.UNARY_POSTFIX, op=match),
        new_s,
    )


def _peel_repositioned_tail_operator(
    s: str, v: Vocabularies
) -> tuple[OperatorApplication | Qualifier | None, str]:
    """Peel one unary operator placed between its base and trailing locus."""

    if _resolves_without_postfix(s, v):
        return None, s

    prefix_operators = {
        name
        for name, meta in v.operators.items()
        if meta.get("kind") == OperatorKind.UNARY_PREFIX.value
    }
    if _spells_leading_operator_application(
        s, prefix_operators, _binary_operator_tokens(v), v
    ):
        return None, s

    unary_operators = {
        name: meta
        for name, meta in v.operators.items()
        if meta.get("kind")
        in {OperatorKind.UNARY_PREFIX.value, OperatorKind.UNARY_POSTFIX.value}
    }
    unary_tokens = set(unary_operators)
    binary_operators = _binary_operator_tokens(v)

    @cache
    def resolves_operand(candidate: str) -> bool:
        if _resolves_without_postfix(candidate, v):
            return True
        if _spells_leading_operator_application(
            candidate, prefix_operators, binary_operators, v
        ):
            return True
        return any(
            resolves_operand(candidate[: -len(token) - 1])
            for token in unary_tokens
            if candidate.endswith(f"_{token}") and len(candidate) > len(token) + 1
        )

    match = None
    new_s = s
    for token in sorted(unary_tokens, key=len, reverse=True):
        marker = f"_{token}"
        if not s.endswith(marker) or len(s) <= len(marker):
            continue
        candidate = s[: -len(marker)]
        if resolves_operand(candidate):
            match = token
            new_s = candidate
            break
    if match is None:
        return None, s

    kind = OperatorKind(unary_operators[match]["kind"])
    if kind is OperatorKind.UNARY_PREFIX and match in BARE_PREFIX_OPERATORS:
        return Qualifier(token=match), new_s
    return OperatorApplication(kind=kind, op=match), new_s


def _resolves_without_postfix(s: str, v: Vocabularies) -> bool:
    """Whether the complete spelling resolves without a postfix peel."""
    try:
        _match_base_with_qualifiers(s, v)
    except ParseError:
        return False
    return True


def _longest_suffix_match(s: str, tokens: set[str]) -> str | None:
    best: str | None = None
    for tok in tokens:
        marker = f"_{tok}"
        if s.endswith(marker) and len(s) > len(marker):
            if best is None or len(tok) > len(best):
                best = tok
    return best


def _longest_prefix_operator_match(s: str, tokens: set[str]) -> str | None:
    best: str | None = None
    for tok in tokens:
        marker = f"{tok}_of_"
        if s.startswith(marker) and len(s) > len(marker):
            if best is None or len(tok) > len(best):
                best = tok
    return best


def _coordinate_universe(v: Vocabularies) -> frozenset[str]:
    """Tokens admissible as an indexed-operator coordinate index.

    The ``coord`` index of operators like ``derivative_with_respect_to`` is
    drawn from the coordinate / flux-coordinate vocabulary: the geometry
    carriers (``radial_coordinate``, ``toroidal_flux_coordinate``,
    ``normalized_poloidal_flux_coordinate``, …) plus the bare coordinate axes
    (``radial``, ``poloidal``, …).
    """
    return v.carriers | frozenset(v.carrier_aliases) | frozenset(v.axes)


def _rewrite_operand_first_indexed_operator(s: str, v: Vocabularies) -> str:
    """Rewrite an operand-first coordinate derivative to the internal form.

    The lossless IR keeps a coordinate-indexed operator as one fused token.
    Canonical surface spelling places the complete operand before that index,
    so parsing temporarily restores the fused prefix spelling consumed by the
    existing operator peel. The renderer converts the fused token back to the
    canonical operand-first spelling.
    """
    candidates: list[tuple[str, str, str]] = []
    for op, meta in v.operators.items():
        fixed_operator, fixed_relation, fixed_index = op.partition("_with_respect_to_")
        if fixed_relation and fixed_operator and fixed_index:
            candidates.append((op, fixed_operator, fixed_index))
        if (
            meta.get("kind") != OperatorKind.UNARY_PREFIX.value
            or not meta.get("indexed")
            or list(meta.get("index_params") or []) != ["coord"]
        ):
            continue
        operator, relation, _ = op.partition("_with_respect_to")
        if not relation or not operator:
            continue
        for coord in _coordinate_universe(v):
            canonical_coord = v.carrier_aliases.get(coord, coord)
            candidates.append((f"{op}_{canonical_coord}", operator, coord))

    for fused_op, operator, index in sorted(
        candidates, key=lambda candidate: len(candidate[2]), reverse=True
    ):
        suffix = f"_with_respect_to_{index}"
        if not s.endswith(suffix):
            continue
        operand_end = len(s) - len(suffix)
        marker = f"{operator}_of_"
        operator_start = s.rfind(marker, 0, operand_end)
        if operator_start < 0:
            continue
        operand_start = operator_start + len(marker)
        operand = s[operand_start:operand_end]
        if not operand:
            continue
        return f"{s[:operator_start]}{fused_op}_of_{operand}"
    return s


def _longest_indexed_prefix_operator_match(
    s: str, prefix_ops: set[str], v: Vocabularies
) -> tuple[str, int] | None:
    """Match ``<op>_<coord>_of_`` for an indexed unary-prefix operator.

    ``<op>`` must be an indexed prefix operator (``index_params`` declared with
    a single ``coord`` parameter) and ``<coord>`` must be a registered
    coordinate token (see :func:`_coordinate_universe`). Returns the fused
    operator token ``<op>_<coord>`` together with the byte length consumed up
    to (but excluding) the ``_of_`` separator, or ``None`` when no indexed
    operator binds.

    The longest fused match wins (operator length first, then coordinate
    length) so an overlapping plain-prefix match never shadows it.
    """
    coords = _coordinate_universe(v)
    best: tuple[str, int] | None = None
    for op in prefix_ops:
        meta = v.operators.get(op, {})
        if not meta.get("indexed"):
            continue
        params = meta.get("index_params") or []
        # Only the single-coordinate index form is supported in the prefix
        # position (``<op>_<coord>_of_<base>``).
        if list(params) != ["coord"]:
            continue
        op_prefix = f"{op}_"
        if not s.startswith(op_prefix):
            continue
        remainder = s[len(op_prefix) :]
        of_idx = remainder.find("_of_")
        if of_idx <= 0:
            continue
        coord = remainder[:of_idx]
        if coord not in coords:
            continue
        canonical_coord = v.carrier_aliases.get(coord, coord)
        fused = f"{op}_{canonical_coord}"
        consumed = len(f"{op}_{coord}")
        if best is None or consumed > best[1]:
            best = (fused, consumed)
    return best


_BARE_PREFIX_OPERATORS_LONGEST_FIRST: tuple[str, ...] = tuple(
    sorted(BARE_PREFIX_OPERATORS, key=len, reverse=True)
)


def _postfix_belongs_to_an_operator_operand(s: str, v: Vocabularies) -> bool:
    """Whether a trailing postfix sits inside a leading operator application.

    ``square_of_magnetic_field_magnitude`` is the square of the field
    magnitude, not the magnitude of its square. The same rule keeps a trailing
    postfix inside the right operand of a binary application. A bare reduction
    may wrap either form, so look through a bare prefix only when its remainder
    is itself an explicit operator application.
    """
    binary_ops = _binary_operator_tokens(v)
    prefix_ops = {
        name
        for name, meta in v.operators.items()
        if meta.get("kind") == OperatorKind.UNARY_PREFIX.value
    }
    if _spells_leading_operator_application(s, prefix_ops, binary_ops, v):
        return True
    postfix_ops = {
        name
        for name, meta in v.operators.items()
        if meta.get("kind") == OperatorKind.UNARY_POSTFIX.value
    }
    postfix = _longest_suffix_match(s, postfix_ops)
    if postfix is not None:
        undecorated = s[: -len(postfix) - 1]
        if undecorated in v.base_universe():
            return False
    for op in _BARE_PREFIX_OPERATORS_LONGEST_FIRST:
        head = f"{op}_"
        if not s.startswith(head):
            continue
        return _spells_nested_operator_application(
            s[len(head) :], prefix_ops, binary_ops, v
        )
    return False


def _binary_operator_tokens(v: Vocabularies) -> set[str]:
    """Registered binary operator tokens."""
    return {
        name
        for name, meta in v.operators.items()
        if meta.get("kind") == OperatorKind.BINARY.value
    }


def _spells_binary_application(s: str, binary_ops: set[str], v: Vocabularies) -> bool:
    """Whether ``s`` spells ``<binary_op>_of_<A>_<sep>_<B>``.

    A cheap string test — it does not parse the operands, so a true result means
    "shaped like a binary application", not "resolves". Callers use it to choose
    a peel order, and the operand parse in the binary peel is the real gate.
    """
    for op in binary_ops:
        prefix = f"{op}_of_"
        if not s.startswith(prefix):
            continue
        sep = v.operators[op].get("separator")
        if sep and f"_{sep}_" in s[len(prefix) :]:
            return True
    return False


def _spells_nested_operator_application(
    s: str,
    prefix_ops: set[str],
    binary_ops: set[str],
    v: Vocabularies,
) -> bool:
    """Whether ``s`` is a registered explicit operator application."""
    if _spells_leading_operator_application(s, prefix_ops, binary_ops, v):
        return True

    postfix_ops = {
        name
        for name, meta in v.operators.items()
        if meta.get("kind") == OperatorKind.UNARY_POSTFIX.value
    }
    postfix = _longest_suffix_match(s, postfix_ops)
    if postfix is None:
        return False
    operand = s[: -len(postfix) - 1]
    # Locus/mechanism tails are stripped before ordinary operator peeling.
    # Do not steal those forms into a bare-prefix tree here; this predicate is
    # for an unambiguously nested postfix expression.
    return bool(operand) and not any(
        marker in operand
        for marker in ("_at_", "_over_", "_along_", "_due_to_", "_of_")
    )


def _spells_leading_operator_application(
    s: str,
    prefix_ops: set[str],
    binary_ops: set[str],
    v: Vocabularies,
) -> bool:
    """Whether ``s`` starts with a registered prefix or binary application."""
    if _spells_binary_application(s, binary_ops, v):
        return True
    if _longest_prefix_operator_match(s, prefix_ops) is not None:
        return True
    return _longest_indexed_prefix_operator_match(s, prefix_ops, v) is not None


def _longest_bare_prefix_over_operator_match(
    s: str, prefix_ops: set[str], binary_ops: set[str], v: Vocabularies
) -> str | None:
    """Longest bare prefix whose remainder spells an explicit operator form.

    Restricting the match to an operator remainder keeps this from stealing the
    qualifier reading of an ordinary name: in
    ``flux_surface_averaged_electron_density`` the remainder is a base, so the
    operator stays a qualifier.
    """
    best: str | None = None
    for op in prefix_ops & BARE_PREFIX_OPERATORS:
        head = f"{op}_"
        if not s.startswith(head):
            continue
        if not _spells_nested_operator_application(
            s[len(head) :], prefix_ops, binary_ops, v
        ):
            continue
        if best is None or len(op) > len(best):
            best = op
    return best


def _try_parse_or_literal(
    s: str,
    v: Vocabularies,
    diagnostics: list[Diagnostic] | None = None,
) -> StandardNameIR | None:
    """Try to parse ``s`` as a full standard name; fall back to a literal base.

    Returns ``None`` only when ``s`` is syntactically invalid (not
    snake_case). For valid-looking tokens that don't match the closed
    vocabulary, returns a literal ``QuantityOrCarrier`` so binary operator
    operands with unregistered compound bases (e.g. ``magnetic_pressure``)
    are accepted.
    """
    try:
        return parse(s, vocabs=v).ir
    except ParseError:
        if TOKEN_PATTERN.match(s):
            if diagnostics is not None:
                diagnostics.append(
                    Diagnostic(
                        category="vocab_gap",
                        layer="parser",
                        message=(
                            f"binary operand {s!r} used the literal-base fallback; "
                            "the validity oracle will require registered or "
                            "qualifier-elided operand vocabulary"
                        ),
                        severity="warning",
                    )
                )
            return StandardNameIR(
                base=QuantityOrCarrier(token=s, kind=BaseKind.QUANTITY)
            )
        return None


def _match_base_with_qualifiers(
    s: str, v: Vocabularies, *, _allow_projection: bool = True
) -> tuple[QuantityOrCarrier, list[Qualifier], AxisProjection | None]:
    """Match ``s`` as ``[axis_][qualifier_]*(base|carrier)``.

    Resolution priority: carrier > base > axis > qualifier.

    When ``_allow_projection`` is True (the default), an axis prefix
    followed by a resolvable base/carrier is interpreted as a projection:
    axis + quantity base → COMPONENT, axis + carrier → COORDINATE.
    Nested projections (projection inside a projection) are blocked by
    recursing with ``_allow_projection=False``.

    Returns ``(base_or_carrier, qualifiers, projection_or_none)``.
    """

    if s in v.carrier_aliases:
        return (
            QuantityOrCarrier(token=v.carrier_aliases[s], kind=BaseKind.GEOMETRY),
            [],
            None,
        )
    if s in v.base_aliases:
        return (
            QuantityOrCarrier(token=v.base_aliases[s], kind=BaseKind.QUANTITY),
            [],
            None,
        )
    if s in v.carriers:
        return QuantityOrCarrier(token=s, kind=BaseKind.GEOMETRY), [], None
    if s in v.bases:
        return QuantityOrCarrier(token=s, kind=BaseKind.QUANTITY), [], None

    parts = s.split("_")

    # A plane token may also be a projection axis (poloidal). When the
    # remainder is explicitly cross-sectional, the section-plane reading is
    # authoritative and must win before projection matching.
    for split in range(len(parts) - 1, 0, -1):
        prefix = "_".join(parts[:split])
        rest = "_".join(parts[split:])
        section_plane = v.section_planes.get(prefix)
        if section_plane is None or not rest:
            continue
        try:
            base, deeper, inner_proj = _match_base_with_qualifiers(
                rest, v, _allow_projection=False
            )
        except ParseError:
            continue
        cross_sectional = base.token == "cross_section" or any(
            qualifier.token == "cross_sectional" for qualifier in deeper
        )
        if inner_proj is not None or not cross_sectional:
            continue
        return (
            base,
            [Qualifier(token=section_plane, category="section_plane"), *deeper],
            None,
        )

    # --- Priority 3: axis prefix → projection ---
    if _allow_projection:
        projection_axes = v.axes | v.component_axes
        for split in range(len(parts) - 1, 0, -1):
            prefix = "_".join(parts[:split])
            rest = "_".join(parts[split:])
            if prefix not in projection_axes or not rest:
                continue
            try:
                base, quals, inner_proj = _match_base_with_qualifiers(
                    rest, v, _allow_projection=False
                )
            except ParseError:
                continue
            if inner_proj is not None:
                continue  # nested projections not allowed
            shape = (
                ProjectionShape.COORDINATE
                if base.kind is BaseKind.GEOMETRY
                else ProjectionShape.COMPONENT
            )
            allowed_axes = (
                v.axes
                if shape is ProjectionShape.COORDINATE
                else (v.component_axes or v.axes)
            )
            if prefix not in allowed_axes:
                continue
            return base, quals, AxisProjection(axis=prefix, shape=shape)

    # --- Priority 4: qualifier prefix ---
    for split in range(len(parts) - 1, 0, -1):
        prefix = "_".join(parts[:split])
        rest = "_".join(parts[split:])
        if prefix not in v.qualifiers:
            continue
        if not rest:
            continue
        try:
            base, deeper, proj = _match_base_with_qualifiers(
                rest, v, _allow_projection=_allow_projection
            )
        except ParseError:
            continue
        return (
            base,
            [
                Qualifier(token=prefix, category=v.qualifier_categories.get(prefix)),
                *deeper,
            ],
            proj,
        )

    suggestions = get_close_matches(s, list(v.base_universe()), n=3)
    raise ParseError(
        f"residue {s!r} does not match any physical_base or geometry_carrier; "
        f"nearest candidates: {suggestions or '(none)'}",
        suggestions=suggestions,
        residue=s,
    )


# ---------------------------------------------------------------------------
# Strict ordered-IR validation
# ---------------------------------------------------------------------------


def _operator_metadata(
    operator: OperatorApplication, v: Vocabularies
) -> tuple[str, Mapping[str, Any]]:
    """Resolve an IR operator to its registered token and metadata."""
    direct = v.operators.get(operator.op)
    if direct is not None:
        return operator.op, direct

    for token in sorted(v.operators, key=len, reverse=True):
        meta = v.operators[token]
        marker = f"{token}_"
        if not operator.op.startswith(marker) or not meta.get("indexed"):
            continue
        index = operator.op[len(marker) :]
        if list(meta.get("index_params") or []) == ["coord"] and index in (
            _coordinate_universe(v)
        ):
            return token, meta

    raise ParseError(f"operator {operator.op!r} is not registered")


def _strict_operator_spelling(ir: StandardNameIR, v: Vocabularies) -> None:
    """Enforce one canonical spelling for every registered operator."""
    inner_bare_precedence: list[tuple[str, int]] = []
    for qualifier in ir.qualifiers:
        meta = v.operators.get(qualifier.token)
        if (
            meta is not None
            and meta.get("kind") == OperatorKind.UNARY_PREFIX.value
            and qualifier.token not in BARE_PREFIX_OPERATORS
            and not meta.get("extremum_reduction")
        ):
            raise ParseError(
                f"operator spelling for {qualifier.token!r} requires "
                f"'{qualifier.token}_of_<operand>'; the glued form is not canonical"
            )
        if qualifier.token in BARE_PREFIX_OPERATORS and meta is not None:
            inner_bare_precedence.append(
                (qualifier.token, int(meta.get("precedence", 0)))
            )

    binary_seen = False
    prefix_precedence: list[tuple[str, int]] = []
    for index, operator in enumerate(ir.operators):
        registered_token, meta = _operator_metadata(operator, v)
        declared_kind = meta.get("kind")
        if declared_kind != operator.kind.value:
            raise ParseError(
                f"operator {operator.op!r} has kind {operator.kind.value!r}, "
                f"but the registry declares {declared_kind!r}"
            )
        if operator.kind is OperatorKind.BINARY:
            if binary_seen or index != len(ir.operators) - 1:
                raise ParseError(
                    f"binary operator {operator.op!r} must terminate its "
                    "enclosing outer-to-inner operator chain"
                )
            binary_seen = True
            registered_separator = meta.get("separator")
            if registered_separator != operator.separator:
                raise ParseError(
                    f"binary operator {operator.op!r} requires separator "
                    f"{registered_separator!r}, got {operator.separator!r}"
                )
        elif operator.kind is OperatorKind.UNARY_PREFIX:
            prefix_precedence.append((registered_token, int(meta.get("precedence", 0))))
            canonical_bare = registered_token in BARE_PREFIX_OPERATORS
            if operator.bare_prefix != canonical_bare:
                form = (
                    f"{registered_token}_<operand>"
                    if canonical_bare
                    else f"{registered_token}_of_<operand>"
                )
                raise ParseError(
                    f"operator spelling for {registered_token!r} must be {form!r}"
                )
        for argument in operator.args:
            _strict_operator_spelling(argument, v)

    # Prefix precedence governs prefix nesting only. Postfix operators have an
    # authored tail position that already fixes their binding (both
    # square_of_field_magnitude and square_of_field_fourier_coefficient are
    # canonical), while a binary application terminates the local chain.
    # Explicit prefixes wrap the qualifier-held bare prefixes on the base, so
    # they precede them in the actual outer-to-inner order.
    ordered_precedence = [*prefix_precedence, *inner_bare_precedence]
    for outer, inner in zip(ordered_precedence, ordered_precedence[1:], strict=False):
        if outer[1] < inner[1]:
            raise ParseError(
                f"operator {outer[0]!r} with precedence {outer[1]} cannot "
                f"wrap {inner[0]!r} with precedence {inner[1]}"
            )


def _operand_is_resolved(ir: StandardNameIR, v: Vocabularies) -> bool:
    """Whether an operator operand bottoms out in registered base vocabulary."""
    binary = next(
        (operator for operator in ir.operators if operator.kind is OperatorKind.BINARY),
        None,
    )
    if binary is not None:
        return all(_operand_is_resolved(argument, v) for argument in binary.args)
    if ir.base.token not in v.base_universe():
        return False
    return all(
        _operand_is_resolved(argument, v)
        for operator in ir.operators
        for argument in operator.args
    )


def _qualifier_segmentations(
    spelling: str, v: Vocabularies
) -> list[tuple[tuple[str, frozenset[str]], ...]]:
    """Return at most two closed-vocabulary qualifier segmentations."""
    words = tuple(spelling.split("_"))
    candidates = [
        (tuple(token.split("_")), token, roles)
        for token, roles in v.qualifier_roles.items()
        if roles
    ]
    by_start: dict[int, list[tuple[tuple[str, frozenset[str]], ...]]] = {
        len(words): [()]
    }
    for start in range(len(words) - 1, -1, -1):
        solutions: list[tuple[tuple[str, frozenset[str]], ...]] = []
        for token_words, token, roles in candidates:
            end = start + len(token_words)
            if words[start:end] != token_words:
                continue
            for suffix in by_start.get(end, []):
                solutions.append(((token, roles), *suffix))
                if len(solutions) == 2:
                    break
            if len(solutions) == 2:
                break
        by_start[start] = solutions
    return by_start.get(0, [])


def _has_unambiguous_qualifier_roles(
    segmentation: tuple[tuple[str, frozenset[str]], ...],
) -> bool:
    """Whether one segmentation maps uniquely to compatible segment roles."""
    if any(len(roles) != 1 for _, roles in segmentation):
        return False
    roles = [next(iter(token_roles)) for _, token_roles in segmentation]
    single_cardinality = {
        "aggregation",
        "population",
        "orbit",
        "state",
        "subject",
        "object",
        "channel",
        "channel_qualifier",
        "section_plane",
        "geometry_representation",
    }
    return all(
        roles.count(role) <= 1
        for role in single_cardinality
        | {r for r in roles if r.startswith("qualifier:")}
    )


def _operand_is_qualifier_elision(
    ir: StandardNameIR, sibling: StandardNameIR, v: Vocabularies
) -> bool:
    """Whether an unresolved operand safely elides a shared sibling base."""
    if (
        ir.operators
        or ir.projection is not None
        or ir.locus is not None
        or ir.mechanism is not None
        or sibling.operators
        or sibling.projection is not None
        or sibling.locus is not None
        or sibling.mechanism is not None
        or sibling.base.token not in v.base_universe()
    ):
        return False

    segmentations = _qualifier_segmentations(ir.base.token, v)
    if len(segmentations) != 1 or not _has_unambiguous_qualifier_roles(
        segmentations[0]
    ):
        return False

    # Prove the elision by restoring the sibling's registered base and running
    # the same strict validity oracle used for authored names.
    expanded = f"{ir.base.token}_{sibling.base.token}"
    try:
        parse(expanded, vocabs=v, strict=True)
    except ParseError:
        return False
    return True


def _strict_binary_operands(
    ir: StandardNameIR, v: Vocabularies, allowed_elisions: set[int]
) -> None:
    """Validate closed binary operands and record safe qualifier elisions."""
    for operator in ir.operators:
        if operator.kind is not OperatorKind.BINARY:
            for argument in operator.args:
                _strict_binary_operands(argument, v, allowed_elisions)
            continue

        left, right = operator.args
        left_resolved = _operand_is_resolved(left, v)
        right_resolved = _operand_is_resolved(right, v)
        if not left_resolved:
            if right_resolved and _operand_is_qualifier_elision(left, right, v):
                allowed_elisions.add(id(left))
            else:
                raise ParseError(
                    f"binary operand {compose(left)!r} is not registered",
                    residue=left.base.token,
                )
        if not right_resolved:
            if left_resolved and _operand_is_qualifier_elision(right, left, v):
                allowed_elisions.add(id(right))
            else:
                raise ParseError(
                    f"binary operand {compose(right)!r} is not registered",
                    residue=right.base.token,
                )
        _strict_binary_operands(left, v, allowed_elisions)
        _strict_binary_operands(right, v, allowed_elisions)


def _operator_accepts(actual: str, allowed: list[str]) -> bool:
    """Whether an inferred operand kind satisfies a registry constraint."""
    if actual in allowed:
        return True
    # Base metadata records structural rank, not whether a scalar/vector value
    # is real or complex. A complex-only decomposition therefore cannot be
    # disproved from a registered scalar/vector kind.
    if allowed == ["complex"] and actual in {"scalar", "vector", "scalar_or_vector"}:
        return True
    if "scalar_or_vector" in allowed and actual in {"scalar", "vector"}:
        return True
    if actual == "scalar_or_vector" and {"scalar", "vector"} & set(allowed):
        return False
    return False


def _operator_result_kind(meta: Mapping[str, Any], operand_kinds: list[str]) -> str:
    """Infer the output kind declared by an operator application."""
    declared = meta.get("returns")
    if declared in {None, "scalar_or_vector", "rate"}:
        concrete = {kind for kind in operand_kinds if kind in {"scalar", "vector"}}
        if len(concrete) == 1:
            return concrete.pop()
        return "scalar_or_vector"
    return str(declared)


def _strict_expression_kind(
    ir: StandardNameIR,
    v: Vocabularies,
    allowed_elisions: set[int],
    *,
    enclosing_operator: bool = False,
) -> str:
    """Validate one ordered expression and return its inferred result kind."""
    binary = next(
        (operator for operator in ir.operators if operator.kind is OperatorKind.BINARY),
        None,
    )
    if binary is None:
        if ir.base.token in v.base_universe():
            current_kind = v.base_kinds.get(
                ir.base.token,
                "geometry" if ir.base.kind is BaseKind.GEOMETRY else "scalar_or_vector",
            )
        elif id(ir) in allowed_elisions:
            current_kind = "scalar_or_vector"
        else:
            raise ParseError(f"base token {ir.base.token!r} is not registered")
    else:
        current_kind = "scalar_or_vector"

    has_local_operator = bool(ir.operators) or any(
        qualifier.token in BARE_PREFIX_OPERATORS for qualifier in ir.qualifiers
    )
    if (
        ir.base.token in GENERIC_PHYSICAL_BASES
        and not enclosing_operator
        and not has_local_operator
        and not ir.qualifiers
        and ir.projection is None
        and ir.locus is None
        and ir.mechanism is None
    ):
        raise ParseError(f"generic base {ir.base.token!r} requires qualification")

    # Bare-prefix operators live in the qualifier-held inner expression.
    # Evaluate them before walking the explicit outer-to-inner stack in reverse.
    for qualifier in reversed(ir.qualifiers):
        if qualifier.token not in BARE_PREFIX_OPERATORS:
            continue
        meta = v.operators[qualifier.token]
        if current_kind == "geometry":
            raise ParseError(
                f"operator {qualifier.token!r} cannot apply to a geometry carrier"
            )
        allowed = list(meta.get("arg_types") or [])
        if allowed and not _operator_accepts(current_kind, allowed):
            raise ParseError(
                f"operator {qualifier.token!r} requires one of {allowed}, "
                f"got {current_kind!r}"
            )
        current_kind = _operator_result_kind(meta, [current_kind])

    for operator in reversed(ir.operators):
        _, meta = _operator_metadata(operator, v)
        if operator.kind is OperatorKind.BINARY:
            operand_kinds = [
                _strict_expression_kind(
                    argument,
                    v,
                    allowed_elisions,
                    enclosing_operator=True,
                )
                for argument in operator.args
            ]
        elif operator.args:
            operand_kinds = [
                _strict_expression_kind(
                    operator.args[0],
                    v,
                    allowed_elisions,
                    enclosing_operator=True,
                )
            ]
        else:
            operand_kinds = [current_kind]

        if "geometry" in operand_kinds:
            raise ParseError(
                f"operator {operator.op!r} cannot apply to a geometry carrier"
            )
        allowed = list(meta.get("arg_types") or [])
        if allowed:
            for actual in operand_kinds:
                if not _operator_accepts(actual, allowed):
                    raise ParseError(
                        f"operator {operator.op!r} requires one of {allowed}, "
                        f"got {actual!r}"
                    )
        current_kind = _operator_result_kind(meta, operand_kinds)

    return current_kind


def _strict_flux_surface_reductions(
    ir: StandardNameIR,
    v: Vocabularies,
    *,
    inherited: frozenset[str] = frozenset(),
) -> None:
    """Reject reductions recursively applied to flux-function bases."""
    reductions = {
        token
        for token, meta in v.operators.items()
        if meta.get("flux_surface_reduction")
    }
    active = set(inherited)
    active.update(
        qualifier.token for qualifier in ir.qualifiers if qualifier.token in reductions
    )
    binary_seen = False
    for operator in ir.operators:
        if operator.op in reductions:
            active.add(operator.op)
        if operator.kind is OperatorKind.BINARY:
            binary_seen = True
            for argument in operator.args:
                _strict_flux_surface_reductions(
                    argument, v, inherited=frozenset(active)
                )
        else:
            for argument in operator.args:
                _strict_flux_surface_reductions(
                    argument, v, inherited=frozenset(active)
                )
    if not binary_seen and active and ir.base.token in v.flux_function_bases:
        operator = sorted(active)[0]
        raise ParseError(
            f"operator {operator!r} cannot apply to {ir.base.token!r}: "
            "the base is constant on a flux surface"
        )


def _temporal_change_operators() -> tuple[str, ...]:
    """Registered operators declaring a temporal-change effect."""
    return tuple(
        sorted(
            token
            for token in vocab_loaders.load_operators().operators
            if "temporal_change" in get_operator_semantics(token)
        )
    )


def _rate_bases(v: Vocabularies) -> tuple[str, ...]:
    """Registered bases that already name a quantity per unit time."""
    return tuple(
        sorted(
            token
            for token in v.base_universe()
            if token == "rate" or token.endswith("_rate")
        )
    )


def _strict_temporal_denominator(ir: StandardNameIR, v: Vocabularies) -> None:
    """Reject a ratio that divides a quantity by the time base.

    Dividing by elapsed time is a temporal derivative, not a ratio of two
    quantities, so the spelling must use a temporal-change operator or a base
    that already carries the per-time reading.
    """
    for operator in ir.operators:
        if operator.kind is OperatorKind.BINARY and operator.separator == "to":
            denominator = operator.args[1]
            if denominator.base.token == "time" and not denominator.operators:
                operators = ", ".join(_temporal_change_operators())
                bases = ", ".join(_rate_bases(v))
                raise ParseError(
                    f"operator {operator.op!r} cannot divide "
                    f"{compose(operator.args[0])!r} by 'time': a quantity per "
                    "unit time is a temporal derivative, not a ratio; spell it "
                    f"with a temporal-change operator ({operators}) or with a "
                    f"rate base ({bases})"
                )
        for argument in operator.args:
            _strict_temporal_denominator(argument, v)


def _strict_extremum_infix(ir: StandardNameIR, v: Vocabularies) -> None:
    """Reject an extremum adjective where an operator spelling is required."""
    extremum_tokens = {
        token for token, meta in v.operators.items() if meta.get("extremum_reduction")
    } | {"peak"}
    infix = {qualifier.token for qualifier in ir.qualifiers} & extremum_tokens
    if infix and ir.base.token in v.extremum_infix_bases:
        token = sorted(infix)[0]
        raise ParseError(
            f"qualifier {token!r} cannot be an infix inside {ir.base.token!r}: "
            "an extremum of a flux is a reduction transformation"
        )
    for operator in ir.operators:
        for argument in operator.args:
            _strict_extremum_infix(argument, v)


def _strict_state_semantics(ir: StandardNameIR, v: Vocabularies) -> None:
    """Apply the shared state-to-subject compatibility contract to lossless IR."""
    state_tokens = [
        qualifier.token
        for qualifier in ir.qualifiers
        if "state" in v.qualifier_roles.get(qualifier.token, ())
    ]
    if state_tokens:
        subject_tokens = [
            qualifier.token
            for qualifier in ir.qualifiers
            if "subject" in v.qualifier_roles.get(qualifier.token, ())
        ]
        from imas_standard_names.grammar.model import (  # noqa: PLC0415
            _STATE_SUBJECT_COMPAT,
        )

        state = state_tokens[0]
        subject = subject_tokens[0] if len(subject_tokens) == 1 else None
        if subject not in _STATE_SUBJECT_COMPAT.get(state, frozenset()):
            raise ParseError(f"state {state!r} requires one compatible species subject")
    for operator in ir.operators:
        for argument in operator.args:
            _strict_state_semantics(argument, v)


def _strict_flat_segment_semantics(
    name: str,
    ir: StandardNameIR,
    *,
    enclosing_operator: bool = False,
) -> None:
    """Reuse flat-model validators whenever the ordered IR is projectable."""
    from imas_standard_names.grammar.model import (  # noqa: PLC0415
        StandardName,
        _assert_lossless_canonical,
        _check_state_gate,
        _ir_to_model_dict,
        _model_to_ir,
    )

    for operator in ir.operators:
        for argument in operator.args:
            _strict_flat_segment_semantics(
                compose(argument),
                argument,
                enclosing_operator=True,
            )

    try:
        model_data = _ir_to_model_dict(ir)
    except ValueError as error:
        if "not representable in the flat StandardName model" in str(error):
            binary = next(
                (
                    operator
                    for operator in ir.operators
                    if operator.kind is OperatorKind.BINARY
                ),
                None,
            )
            if binary is not None:
                root_shell = ir.model_copy(update={"operators": []})
                _strict_operator_free_core(
                    root_shell,
                    operator_qualified=True,
                )
                if ir.operators != [binary]:
                    binary_core = ir.model_copy(update={"operators": [binary]})
                    _strict_flat_segment_semantics(
                        compose(binary_core),
                        binary_core,
                        enclosing_operator=True,
                    )
            else:
                core = ir.model_copy(update={"operators": []})
                _strict_operator_free_core(
                    core,
                    operator_qualified=bool(ir.operators) or enclosing_operator,
                )
            return
        raise ParseError(str(error)) from error
    try:
        model = StandardName.model_validate(
            model_data,
            context={"enclosing_operator": enclosing_operator},
        )
        _check_state_gate(model)
    except ValueError as error:
        raise ParseError(str(error)) from error

    canonical = compose(_model_to_ir(model))
    if canonical == name:
        return
    try:
        _assert_lossless_canonical(name, canonical)
    except ValueError as error:
        raise ParseError(str(error)) from error
    raise _NonCanonicalParseError(name, canonical)


def _strict_operator_free_core(
    core: StandardNameIR,
    *,
    operator_qualified: bool,
) -> None:
    """Validate a wrapped core without flattening its ordered operator stack."""
    from imas_standard_names.grammar.model import (  # noqa: PLC0415
        StandardName,
        _assert_lossless_canonical,
        _check_state_gate,
        _ir_to_model_dict,
        _model_to_ir,
    )

    core_name = compose(core)
    try:
        model_data = _ir_to_model_dict(core)
        model = StandardName.model_validate(
            model_data,
            context={"enclosing_operator": operator_qualified},
        )
        _check_state_gate(model)
    except ValueError as error:
        raise ParseError(str(error)) from error

    canonical = compose(_model_to_ir(model))
    if canonical == core_name:
        return
    try:
        _assert_lossless_canonical(core_name, canonical)
    except ValueError as error:
        raise ParseError(str(error)) from error
    raise ParseError(
        f"wrapped core is not canonical: flat segment order renders as {canonical!r}"
    )


def _strict_validate(name: str, ir: StandardNameIR, v: Vocabularies) -> None:
    """Validate the lossless ordered IR without projecting to the flat model."""
    _strict_operator_spelling(ir, v)
    allowed_elisions: set[int] = set()
    _strict_binary_operands(ir, v, allowed_elisions)
    _strict_flux_surface_reductions(ir, v)
    _strict_temporal_denominator(ir, v)
    _strict_extremum_infix(ir, v)
    _strict_expression_kind(ir, v, allowed_elisions)
    _strict_state_semantics(ir, v)
    _strict_flat_segment_semantics(name, ir)
    rendered = compose(ir)
    if rendered != name:
        raise ParseError(
            f"name is not canonical: rendered ordered expression is {rendered!r}"
        )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _parse_uncached(
    name: str,
    vocabs: Vocabularies,
    *,
    strict: bool,
) -> ParseResult:
    """Parse one memoization-cache miss."""
    v = vocabs
    diagnostics: list[Diagnostic] = []
    s = _rewrite_operand_first_indexed_operator(name, v)

    # Tail-final postfix acceptance pass. Peeling a postfix token before the
    # mechanism/locus strips prevents those strips from absorbing it into a
    # fabricated process or locus token. Canonical composition moves this
    # operator before the tail; lenient parsing retains the authored IR so the
    # strict canonical check can report the unique rendering.
    trailing_postfix: list[OperatorApplication] = []
    while True:
        op_app, new_s = _peel_trailing_postfix_operator(s, v)
        if op_app is None:
            break
        if _postfix_belongs_to_an_operator_operand(s, v):
            # Leave it for the leading application's operand parse; hoisting it
            # here would reverse the authored operator order.
            break
        trailing_postfix.append(op_app)
        s = new_s

    # Mechanism pass.
    mechanism, s = _strip_mechanism(s)

    # A binary application has precedence over a trailing locus: the suffix
    # may belong to its recursively parsed final operand.  Parsing the binary
    # first is the only lossless interpretation because an enclosing binary
    # locus has the same flat spelling and the renderer rejects that ambiguous
    # IR shape.
    leading_expression = _peel_leading_binary_expression(s, v, diagnostics)
    if leading_expression is not None:
        ir = StandardNameIR(
            operators=[*trailing_postfix, *leading_expression],
            base=QuantityOrCarrier(token="placeholder", kind=BaseKind.QUANTITY),
            mechanism=mechanism,
        )
        result = ParseResult(ir=ir, diagnostics=diagnostics)
        if strict:
            _strict_validate(name, result.ir, v)
        return result

    # Locus pass.
    locus, s, locus_diags = _strip_locus(s, v)
    diagnostics.extend(locus_diags)

    # A unary operator on an ordinary base moves immediately before an
    # ``of``/``at`` locus or ``due_to`` mechanism. Peel that suffix before
    # resolving the base so the parser reconstructs the same outer-to-inner
    # operator stack that the renderer consumed. Bare prefix operators retain
    # their qualifier-held IR representation.
    has_repositioning_tail = mechanism is not None or (
        locus is not None and locus.relation in {LocusRelation.OF, LocusRelation.AT}
    )
    repositioned_operators: list[OperatorApplication] = []
    repositioned_qualifiers: list[Qualifier] = []
    if has_repositioning_tail:
        while True:
            operator, new_s = _peel_repositioned_tail_operator(s, v)
            if operator is None:
                break
            if isinstance(operator, Qualifier):
                repositioned_qualifiers.append(operator)
            else:
                repositioned_operators.append(operator)
            s = new_s

    # Operator-peeling pass (outermost layer after locus/mechanism).
    # Trailing postfix operators peeled first are the outermost wrap and
    # precede anything peeled here (a prefix/binary operator-of form).
    operator_stack: list[OperatorApplication] = list(trailing_postfix)
    binary_terminator: OperatorApplication | None = None
    while True:
        op_app, new_s, _ = _peel_outer_operator(s, v, diagnostics)
        if op_app is None:
            break
        if op_app.kind is OperatorKind.BINARY:
            binary_terminator = op_app
            s = ""
            break
        operator_stack.append(op_app)
        s = new_s

    operator_stack.extend(repositioned_operators)

    if binary_terminator is not None:
        # Binary consumed everything. Base/qualifiers/projection must be empty.
        if s:
            raise ParseError(
                f"binary operator {binary_terminator.op!r} cannot combine with "
                "residue; got unexpected trailing content"
            )
        # Synthesize a placeholder base so the outer IR validates. The
        # binary operator lives on the outer IR's operators stack and its
        # args carry the real structure. The placeholder is never rendered.
        # The full operator stack — trailing postfix plus any
        # prefix operators peeled before the binary terminator — wraps the
        # binary result, outermost first.
        ir = StandardNameIR(
            operators=[*operator_stack, binary_terminator],
            base=QuantityOrCarrier(token="placeholder", kind=BaseKind.QUANTITY),
            locus=locus,
            mechanism=mechanism,
        )
        result = ParseResult(ir=ir, diagnostics=diagnostics)
        if strict:
            _strict_validate(name, result.ir, v)
        return result

    # Base-resolution pass: carrier > base > axis (projection) > qualifier.
    if not s:
        raise ParseError(
            "empty residue after peeling operators and decorators",
        )
    base, qualifiers, projection = _match_base_with_qualifiers(s, v)
    qualifiers = [*repositioned_qualifiers, *qualifiers]

    ir = StandardNameIR(
        operators=operator_stack,
        projection=projection,
        qualifiers=qualifiers,
        base=base,
        locus=locus,
        mechanism=mechanism,
    )
    result = ParseResult(ir=ir, diagnostics=diagnostics)
    if strict:
        _strict_validate(name, result.ir, v)
    return result


def parse(
    name: str,
    vocabs: Vocabularies | None = None,
    *,
    strict: bool = False,
) -> ParseResult:
    """Parse ``name`` into a :class:`ParseResult`.

    Raises :class:`ParseError` when the residue cannot be resolved against the
    closed base vocabulary. With ``strict=True``, this is the authoritative
    validity oracle for both flat and ordered grammar: it validates registry
    metadata, closed operand vocabulary, segment semantics, generic-base
    qualification, recursive flux-surface semantics, and canonical spelling
    without projecting through the flat :class:`StandardName` facade.

    Recursive binary exploration is memoized by substring and validation mode,
    so ambiguous invalid connector chains do not repeatedly parse the same
    candidate operands.
    """
    if not isinstance(name, str) or not name:
        raise ParseError("name must be a non-empty string")
    if not TOKEN_PATTERN.match(name):
        raise ParseError(
            f"name {name!r} is not a valid grammar token (must be lowercase snake_case)"
        )

    v = vocabs if vocabs is not None else _default_vocabs()
    context = _ACTIVE_PARSE_CONTEXT.get()
    context_token = None
    if context is None or context.vocabs is not v:
        context = _ParseContext(vocabs=v)
        context_token = _ACTIVE_PARSE_CONTEXT.set(context)

    key = (name, strict)
    try:
        cached = context.cache.get(key)
        if isinstance(cached, _CachedParseError):
            cached.raise_fresh()
        if cached is not None:
            return cached
        try:
            result = _parse_uncached(name, v, strict=strict)
        except ParseError as error:
            context.cache[key] = _CachedParseError.from_exception(error)
            raise
        context.cache[key] = result
        return result
    finally:
        if context_token is not None:
            _ACTIVE_PARSE_CONTEXT.reset(context_token)


def validate_round_trip(name: str, vocabs: Vocabularies | None = None) -> bool:
    """Return ``True`` iff ``compose(parse(name).ir) == name``.

    Raises :class:`ParseError` when the name fails to parse. Otherwise
    compares the rendered form against the input byte-for-byte.

    IR-diagnostics tool only — not a validity oracle. It runs on the lenient
    IR parser and answers "does this name render back to itself?", which is
    weaker than validity: it does not enforce segment compatibility, the
    generic-base gate, or the flux-surface reduction gate. Use :func:`parse`
    with ``strict=True`` as the validity oracle. Use
    :func:`imas_standard_names.grammar.model.parse_standard_name` only when a
    validated name must also project into the flat model. Use this helper only
    to locate IR parse/compose round-trip drift.
    """

    result = parse(name, vocabs=vocabs)
    try:
        rendered = compose(result.ir)
    except Exception:
        return False
    return rendered == name
