"""Grammar intermediate representation (IR).

Pydantic v2 models for the 5-group IR:

    StandardNameIR := {
        operators:  [OperatorApplication],       # outer-to-inner stack
        projection: AxisProjection | None,
        qualifiers: [Qualifier],
        base:       QuantityBase | GeometryCarrier,
        locus:      LocusRef | None,
        mechanism:  Process | None,
    }

This module defines *shape only*. It does not resolve tokens against
vocabulary YAMLs — that wiring happens in the vocabulary loaders and parser
(parser). Validators here enforce structural invariants (non-empty tokens,
consistent projection/base pairing, legal locus-relation type matrix, no
empty operator arg lists) and the `_of_` disambiguation assertions.

The canonical renderer lives in :mod:`imas_standard_names.grammar.render`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "LOCUS_VALUE_PATTERN",
    "TOKEN_PATTERN",
    "AxisProjection",
    "BaseKind",
    "LocusRef",
    "LocusRelation",
    "LocusType",
    "OperatorApplication",
    "OperatorKind",
    "ProjectionShape",
    "Qualifier",
    "QuantityOrCarrier",
    "StandardNameIR",
    "LOCUS_RELATION_MATRIX",
    "BARE_PREFIX_OPERATORS",
    "BINARY_SEPARATORS",
    "assert_binary_has_separator",
    "assert_locus_is_trailing",
    "assert_operator_of_form",
]


# A grammar token is lowercase ASCII snake_case. No leading/trailing underscore,
# no digits-only segments (``m_2`` is allowed; ``2`` alone is not).
TOKEN_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")

# A locus value is a numeric literal with underscores as decimal separators:
# ``0_95`` (= 0.95), ``1_0`` (= 1.0), ``2`` (= 2). At most one decimal point.
LOCUS_VALUE_PATTERN = re.compile(r"^\d+(?:_\d+)?$")


def _validate_token(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):  # pragma: no cover - pydantic coerces first
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")
    if not value:
        raise ValueError(f"{field_name} must be a non-empty token")
    if not TOKEN_PATTERN.match(value):
        raise ValueError(
            f"{field_name} must be lowercase snake_case matching "
            f"{TOKEN_PATTERN.pattern!r}; got {value!r}"
        )
    return value


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class OperatorKind(StrEnum):
    """Operator arity / position in the rendered form."""

    UNARY_PREFIX = "unary_prefix"
    UNARY_POSTFIX = "unary_postfix"
    BINARY = "binary"


class ProjectionShape(StrEnum):
    """Axis projection shape — selects component vs coordinate template."""

    COMPONENT = "component"
    COORDINATE = "coordinate"


class BaseKind(StrEnum):
    """The two disjoint base categories.

    A :class:`QuantityOrCarrier` carries a ``kind`` discriminating between
    a physical quantity (e.g. ``temperature``) and a geometry carrier
    (e.g. ``position``, ``outline``). The distinction controls which
    :class:`ProjectionShape` may attach to the base.
    """

    QUANTITY = "quantity"
    GEOMETRY = "geometry"


class LocusRelation(StrEnum):
    """Preposition used in a locus suffix."""

    OF = "of"
    AT = "at"
    OVER = "over"
    ALONG = "along"


class LocusType(StrEnum):
    """Typed locus registry classification."""

    ENTITY = "entity"
    POSITION = "position"
    REGION = "region"
    GEOMETRY = "geometry"


# Locus relation compatibility matrix (see vocabularies/locus_registry.yml).
# ``along`` is a third preposition on POSITION-typed loci, alongside ``of``
# (intrinsic geometry) and ``at`` (field evaluated there): it names a
# path-like locus (line_of_sight, pellet_path) that a quantity varies ALONG
# rather than sits AT a single point of.
LOCUS_RELATION_MATRIX: dict[LocusType, frozenset[LocusRelation]] = {
    LocusType.ENTITY: frozenset({LocusRelation.OF}),
    LocusType.POSITION: frozenset(
        {LocusRelation.OF, LocusRelation.AT, LocusRelation.ALONG}
    ),
    LocusType.REGION: frozenset({LocusRelation.OVER}),
    LocusType.GEOMETRY: frozenset({LocusRelation.OF}),
}


# Allowed separators for binary operators (see vocabularies/operators.yml).
BINARY_SEPARATORS: frozenset[str] = frozenset({"and", "to"})


# Unary-prefix operators that spell WITHOUT the ``_of_`` joiner
# (``flux_surface_averaged_electron_density``, not
# ``flux_surface_averaged_of_electron_density``). The domain reductions among
# them — the precedence-30 members, see the placement policy in
# ``vocabularies/operators.yml`` — lead their name and their operand runs to
# the end of the string, projection and trailing locus included, so they reach
# the IR on the operator stack carrying ``bare_prefix``. Never as a qualifier:
# a qualifier renders after the base and inside the tail, which states a
# narrower scope than the reduction has, and one token with two readings makes
# two names for one quantity. The lower-precedence members modify the base
# rather than reducing it, so they do reach the IR as qualifiers on the base.
BARE_PREFIX_OPERATORS: frozenset[str] = frozenset(
    {
        "accumulated",
        "change_in",
        "cumulative",
        "cumulative_inside_flux_surface",
        "flux_surface_averaged",
        "gyroaveraged",
        "line_averaged",
        "line_integrated",
        "maximum_over_flux_surface",
        "minimum_over_flux_surface",
        "normalized",
        "per_poloidal_mode",
        "per_toroidal_and_poloidal_mode_number",
        "per_toroidal_mode",
        "perturbed",
        "surface_integrated",
        "time_averaged",
        "volume_averaged",
        "volume_integrated",
    }
)


# ---------------------------------------------------------------------------
# Leaf models
# ---------------------------------------------------------------------------


class Qualifier(BaseModel):
    """A species or source-entity prefix qualifier.

    Qualifiers are plain prefix tokens drawn from closed vocabularies. The
    actual vocabulary binding (species vs source_entity) is resolved in
    the loaders; at the IR level we record only the raw token string. A
    ``category`` field is provided for forward-compatibility so parsers may
    tag the token with its vocabulary.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    token: str = Field(description="Snake_case qualifier token.")
    category: str | None = Field(
        default=None,
        description="Optional vocabulary category (e.g. 'species').",
    )

    @field_validator("token")
    @classmethod
    def _check_token(cls, value: str) -> str:
        return _validate_token(value, field_name="qualifier token")

    @field_validator("category")
    @classmethod
    def _check_category(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return _validate_token(value, field_name="qualifier category")


class AxisProjection(BaseModel):
    """``<axis>_component`` or ``<axis>_coordinate`` projection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    axis: str = Field(description="Coordinate axis token (e.g. 'radial').")
    shape: ProjectionShape

    @field_validator("axis")
    @classmethod
    def _check_axis(cls, value: str) -> str:
        return _validate_token(value, field_name="projection axis")


class QuantityOrCarrier(BaseModel):
    """The IR ``base`` slot — either a quantity or a geometry carrier."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    token: str = Field(description="Base token, e.g. 'temperature' or 'position'.")
    kind: BaseKind

    @field_validator("token")
    @classmethod
    def _check_token(cls, value: str) -> str:
        return _validate_token(value, field_name="base token")


class LocusRef(BaseModel):
    """Typed locus reference with a relation preposition.

    ``value`` carries an optional numeric parameterization rendered as
    ``_equal_to_<value>`` after the locus token (underscore as decimal
    separator, e.g. ``0_95`` for 0.95):
    ``safety_factor_at_normalized_poloidal_magnetic_flux_equal_to_0_95``.
    Only position-typed loci with the ``at`` relation admit a value.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    relation: LocusRelation
    token: str = Field(
        description="Locus registry FEATURE token (e.g. 'strike_point')."
    )
    qualifiers: tuple[str, ...] = Field(
        default=(),
        description=(
            "Ordered geometric qualifiers composed onto a qualifiable feature "
            "(e.g. ('inner',) for inner_strike_point, ('upper','outer') for "
            "upper_outer_strike_point). Empty for a bare or non-qualifiable feature."
        ),
    )
    type: LocusType
    value: str | None = Field(
        default=None,
        description=(
            "Numeric literal parameterizing the locus (underscores as decimal "
            "separators, e.g. '0_95'); rendered as '_equal_to_<value>'."
        ),
    )

    @field_validator("token")
    @classmethod
    def _check_token(cls, value: str) -> str:
        return _validate_token(value, field_name="locus token")

    @field_validator("qualifiers")
    @classmethod
    def _check_qualifiers(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for q in value:
            _validate_token(q, field_name="locus qualifier")
        return value

    @field_validator("value")
    @classmethod
    def _check_value(cls, value: str | None) -> str | None:
        if value is not None and not LOCUS_VALUE_PATTERN.fullmatch(value):
            raise ValueError(
                f"locus value {value!r} must be a numeric literal with "
                f"underscores as decimal separators (e.g. '0_95', '1_0', '2')"
            )
        return value

    @model_validator(mode="after")
    def _check_relation_matrix(self) -> LocusRef:
        allowed = LOCUS_RELATION_MATRIX[self.type]
        if self.relation not in allowed:
            allowed_names = ", ".join(sorted(r.value for r in allowed))
            raise ValueError(
                f"locus type {self.type.value!r} does not permit relation "
                f"{self.relation.value!r}; allowed: {allowed_names}"
            )
        if self.value is not None and (
            self.type is not LocusType.POSITION or self.relation is not LocusRelation.AT
        ):
            raise ValueError(
                f"locus value {self.value!r} is only permitted on "
                f"position-typed loci with the 'at' relation "
                f"(got type={self.type.value!r}, relation={self.relation.value!r})"
            )
        return self


class Process(BaseModel):
    """A mechanism / causal process attached via ``_due_to_``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    token: str = Field(description="Process token from processes.yml.")

    @field_validator("token")
    @classmethod
    def _check_token(cls, value: str) -> str:
        return _validate_token(value, field_name="process token")


# ---------------------------------------------------------------------------
# Recursive operator application + top-level IR
# ---------------------------------------------------------------------------


class OperatorApplication(BaseModel):
    """Application of one operator to its argument(s).

    Recursive: each argument is itself a :class:`StandardNameIR`. The
    operator stack in ``StandardNameIR.operators`` is outer-to-inner; the
    renderer walks it by peeling the first element and recursing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: OperatorKind
    op: str = Field(description="Bare operator token (e.g. 'magnitude').")
    args: list[StandardNameIR] = Field(
        default_factory=list,
        description=(
            "Operator arguments. Unary operators: exactly 1 arg or empty "
            "(empty = operator applied to the enclosing IR's base). "
            "Binary operators: exactly 2 args."
        ),
    )
    # Forward-compat: binary operators resolve their separator through the
    # registry; at the IR level callers may pin it explicitly.
    separator: Literal["and", "to"] | None = Field(
        default=None,
        description="Binary-operator separator; required for kind=binary.",
    )
    bare_prefix: bool = Field(
        default=False,
        description=(
            "Render this unary_prefix operator as '<op>_<operand>' rather than "
            "'<op>_of_<operand>'. Only tokens in BARE_PREFIX_OPERATORS may set "
            "it, and both spellings of such a token denote distinct names."
        ),
    )

    @field_validator("op")
    @classmethod
    def _check_op(cls, value: str) -> str:
        return _validate_token(value, field_name="operator token")

    @model_validator(mode="after")
    def _check_arity(self) -> OperatorApplication:
        if self.kind is OperatorKind.BINARY:
            if len(self.args) != 2:
                raise ValueError(
                    f"binary operator {self.op!r} requires exactly 2 args, "
                    f"got {len(self.args)}"
                )
            if self.separator is None:
                raise ValueError(
                    f"binary operator {self.op!r} requires an explicit "
                    f"separator ('and' or 'to')"
                )
            if self.separator not in BINARY_SEPARATORS:
                raise ValueError(
                    f"binary operator separator must be one of "
                    f"{sorted(BINARY_SEPARATORS)}, got {self.separator!r}"
                )
        else:
            # Unary operators carry 0 or 1 arg. 0 = applied to the enclosing
            # IR's base (the common case during parse-peeling); 1 = an
            # explicitly nested sub-IR. More than 1 arg is structurally
            # invalid for unary kinds.
            if len(self.args) > 1:
                raise ValueError(
                    f"unary operator {self.op!r} accepts at most 1 arg, "
                    f"got {len(self.args)}"
                )
            if self.separator is not None:
                raise ValueError(
                    f"unary operator {self.op!r} must not carry a separator"
                )
        if self.bare_prefix:
            if self.kind is not OperatorKind.UNARY_PREFIX:
                raise ValueError(
                    f"bare_prefix applies to the unary_prefix rendering template "
                    f"only; operator {self.op!r} has kind {self.kind.value}"
                )
            if self.op not in BARE_PREFIX_OPERATORS:
                raise ValueError(
                    f"operator {self.op!r} has no bare spelling; bare_prefix is "
                    f"restricted to {sorted(BARE_PREFIX_OPERATORS)}"
                )
        return self


class StandardNameIR(BaseModel):
    """Top-level 5-group IR for a standard name."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    operators: list[OperatorApplication] = Field(
        default_factory=list,
        description="Operator stack, outer-to-inner.",
    )
    projection: AxisProjection | None = None
    qualifiers: list[Qualifier] = Field(
        default_factory=list,
        description="Species / source-entity prefix qualifiers.",
    )
    base: QuantityOrCarrier
    locus: LocusRef | None = None
    mechanism: Process | None = None

    @model_validator(mode="after")
    def _check_projection_base_pairing(self) -> StandardNameIR:
        if self.projection is None:
            return self
        if (
            self.projection.shape is ProjectionShape.COORDINATE
            and self.base.kind is not BaseKind.GEOMETRY
        ):
            raise ValueError(
                "projection shape 'coordinate' requires a geometry-carrier base; "
                f"got base kind {self.base.kind.value!r}"
            )
        if (
            self.projection.shape is ProjectionShape.COMPONENT
            and self.base.kind is not BaseKind.QUANTITY
        ):
            raise ValueError(
                "projection shape 'component' requires a quantity base; "
                f"got base kind {self.base.kind.value!r}"
            )
        return self

    @model_validator(mode="after")
    def _check_qualifier_ordering_and_uniqueness(self) -> StandardNameIR:
        tokens = [q.token for q in self.qualifiers]
        if len(tokens) != len(set(tokens)):
            dupes = sorted({t for t in tokens if tokens.count(t) > 1})
            raise ValueError(f"duplicate qualifiers: {dupes}")
        return self


# Rebuild to resolve the forward reference inside OperatorApplication.args.
OperatorApplication.model_rebuild()
StandardNameIR.model_rebuild()


# ---------------------------------------------------------------------------
# Operator-form assertion helpers
# ---------------------------------------------------------------------------


def assert_operator_of_form(
    op: OperatorApplication,
    registry: Mapping[str, Any] | None = None,
) -> None:
    """Assert ``op`` obeys the operator-form rule.

    A ``unary_prefix`` operator must resolve against ``registry`` (when
    provided) to a registered prefix operator. ``registry`` is expected to
    map operator tokens to metadata dicts with a ``kind`` field matching
    :class:`OperatorKind`. When ``registry`` is ``None`` only the shape of
    ``op`` is checked (non-empty bare token, correct kind).

    Raises :class:`ValueError` on mismatch.
    """

    if op.kind is not OperatorKind.UNARY_PREFIX:
        raise ValueError(
            f"assert_operator_of_form expects a unary_prefix operator; "
            f"got kind={op.kind.value!r}"
        )
    if registry is None:
        return
    entry = registry.get(op.op)
    if entry is None:
        raise ValueError(
            f"operator {op.op!r} is not registered in the operator registry"
        )
    declared_kind = entry.get("kind") if isinstance(entry, Mapping) else None
    if declared_kind != OperatorKind.UNARY_PREFIX.value:
        raise ValueError(
            f"operator {op.op!r} is registered with kind {declared_kind!r}, "
            f"not {OperatorKind.UNARY_PREFIX.value!r}"
        )


def assert_binary_has_separator(
    op: OperatorApplication,
    registry: Mapping[str, Any] | None = None,
) -> None:
    """Assert a binary operator carries (and agrees with) a separator.

    The IR validator already enforces that ``kind=binary`` implies a
    non-``None`` separator in {``and``, ``to``}. This helper additionally
    cross-checks the registry when provided: the registry's declared
    separator for ``op.op`` must equal ``op.separator``.
    """

    if op.kind is not OperatorKind.BINARY:
        raise ValueError(
            f"assert_binary_has_separator expects a binary operator; "
            f"got kind={op.kind.value!r}"
        )
    if op.separator is None or op.separator not in BINARY_SEPARATORS:
        raise ValueError(
            f"binary operator {op.op!r} has invalid separator "
            f"{op.separator!r}; must be one of {sorted(BINARY_SEPARATORS)}"
        )
    if registry is None:
        return
    entry = registry.get(op.op)
    if entry is None:
        raise ValueError(f"binary operator {op.op!r} is not registered")
    if not isinstance(entry, Mapping):
        return
    declared = entry.get("separator")
    if declared is not None and declared != op.separator:
        raise ValueError(
            f"binary operator {op.op!r} separator mismatch: IR={op.separator!r}, "
            f"registry={declared!r}"
        )


def assert_locus_is_trailing(rendered: str, ir: StandardNameIR) -> None:
    """Assert the rendered locus is the final locus segment.

    Given the final rendered string and its IR, verify the locus suffix
    occupies the trailing-position slot (before an optional ``_due_to_``
    mechanism tail). Intended as a safety net inside :func:`compose`; also
    usable from parser tests.

    Raises :class:`ValueError` when the locus does not appear in the
    expected trailing slot.
    """

    if ir.locus is None:
        return
    relation = ir.locus.relation.value
    token = ir.locus.token
    if ir.locus.qualifiers:
        token = "_".join((*ir.locus.qualifiers, token))
    locus_segment = f"_{relation}_{token}"
    if ir.locus.value is not None:
        locus_segment += f"_equal_to_{ir.locus.value}"

    tail = rendered
    if ir.mechanism is not None:
        mech_segment = f"_due_to_{ir.mechanism.token}"
        if not tail.endswith(mech_segment):
            raise ValueError(
                f"mechanism suffix {mech_segment!r} is not the trailing "
                f"segment of rendered form {rendered!r}"
            )
        tail = tail[: -len(mech_segment)]
    if not tail.endswith(locus_segment):
        raise ValueError(
            f"locus suffix {locus_segment!r} is not trailing in rendered "
            f"form {rendered!r} (after stripping mechanism)"
        )
