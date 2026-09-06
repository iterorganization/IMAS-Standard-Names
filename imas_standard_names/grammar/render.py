"""Grammar canonical renderer (strict generator).

Implements :func:`compose` — a pure function that maps a
:class:`StandardNameIR` to its single canonical string form. There are no
fallbacks: malformed IR raises :class:`RenderError`.

The renderer is deliberately isolated from vocabulary resolution. It
consumes validated IR structures (see :mod:`imas_standard_names.grammar.ir`)
and emits token strings; vocabulary lookups happen at parse time.

See the rendering templates in ``vocabularies/operators.yml``.
"""

from collections.abc import Iterable

from imas_standard_names.grammar.ir import (
    BARE_PREFIX_OPERATORS,
    AxisProjection,
    LocusRef,
    OperatorApplication,
    OperatorKind,
    Process,
    Qualifier,
    StandardNameIR,
    assert_binary_has_separator,
    assert_locus_is_trailing,
    assert_operator_of_form,
)

__all__ = [
    "RenderError",
    "compose",
    "render_mechanism",
    "render_operators",
    "render_projection",
    "render_qualifiers",
    "render_locus",
]


class RenderError(ValueError):
    """Raised when the generator cannot produce a canonical form."""


# ---------------------------------------------------------------------------
# Leaf renderers
# ---------------------------------------------------------------------------


def render_projection(projection: AxisProjection | None) -> str:
    """Render a projection as the axis prefix token.

    Both COMPONENT and COORDINATE shapes render identically as just
    ``<axis>`` — the ``_component_of_`` / ``_coordinate_of_`` long forms
    are removed. The caller joins this with the base via ``_``.

    Returns an empty string when ``projection`` is ``None``.
    """

    if projection is None:
        return ""
    return projection.axis


def render_qualifiers(qualifiers: Iterable[Qualifier]) -> str:
    """Render qualifiers in parse order (insertion order).

    Qualifiers are emitted in the order they appear in the IR list,
    which matches the left-to-right order extracted by the parser.
    This ensures parse→compose round-trip fidelity.

    A section-plane qualifier uses an explicit ``_plane`` marker. The marker
    keeps the semantic value ``poloidal`` distinct from the existing
    ``poloidal`` axis projection while leaving the IR value vocabulary-focused.

    Returns an empty string when no qualifiers are present. Otherwise
    returns the tokens joined by ``_`` with no leading or trailing
    underscore; the caller is responsible for gluing it onto the base.
    """

    tokens = [
        f"{qualifier.token}_plane"
        if qualifier.category == "section_plane"
        else qualifier.token
        for qualifier in qualifiers
    ]
    return "_".join(tokens)


def render_locus(locus: LocusRef | None) -> str:
    """Render a locus as ``_<relation>_[<qualifiers>_]<token>[_equal_to_<value>]``.

    Ordered geometric ``qualifiers`` (e.g. ``('inner',)``, ``('upper','outer')``)
    are rendered as a prefix on the feature token — ``_of_inner_strike_point``,
    ``_of_upper_outer_strike_point``. Returns an empty string when ``locus`` is
    ``None``. Relation/type compatibility and value constraints are enforced by
    :class:`LocusRef`'s own validators.
    """

    if locus is None:
        return ""
    token = locus.token
    if locus.qualifiers:
        token = "_".join((*locus.qualifiers, token))
    rendered = f"_{locus.relation.value}_{token}"
    if locus.value is not None:
        rendered += f"_equal_to_{locus.value}"
    return rendered


def render_mechanism(mechanism: Process | None) -> str:
    """Render a mechanism as ``_due_to_<process>``.

    Returns an empty string when ``mechanism`` is ``None``.
    """

    if mechanism is None:
        return ""
    return f"_due_to_{mechanism.token}"


# ---------------------------------------------------------------------------
# Base + inner-IR rendering
# ---------------------------------------------------------------------------


def _render_base_with_decorators(ir: StandardNameIR) -> str:
    """Render the projection + qualifiers + base + locus + mechanism core.

    This function does not include operator decoration. The canonical composer
    may render it whole or render a tail-free copy before inserting operators.
    """

    parts: list[str] = []

    # Short form: both COMPONENT and COORDINATE render as ``<axis>_<rest>``
    projection_str = render_projection(ir.projection)
    if projection_str:
        parts.append(projection_str)

    qualifiers_str = render_qualifiers(ir.qualifiers)
    if qualifiers_str:
        parts.append(qualifiers_str)

    parts.append(ir.base.token)

    core = "_".join(parts)

    # Locus and mechanism are suffixes; their leading underscore is baked in.
    core += render_locus(ir.locus)
    core += render_mechanism(ir.mechanism)
    return core


# ---------------------------------------------------------------------------
# Operator stack rendering
# ---------------------------------------------------------------------------


def _render_operator_stack(
    operators: list[OperatorApplication],
    inner: str,
    enclosing_ir: StandardNameIR,
    *,
    inside_composite_operand: bool,
) -> str:
    """Recursively apply the operator stack outer-to-inner.

    ``operators[0]`` is the outermost operator. ``inner`` is the rendered
    form to be decorated by the remaining stack. ``enclosing_ir`` is the
    IR whose base produced ``inner`` — used for diagnostics only.
    """

    if not operators:
        return inner

    op = operators[0]
    rest = operators[1:]

    if op.kind is OperatorKind.UNARY_PREFIX:
        # If the operator carries an explicit sub-IR arg, render that arg
        # as the operator's operand instead of the inner stream. This is
        # how nested operator trees are represented (args: [sub_ir]).
        if op.args:
            operand = _compose(
                op.args[0], inside_composite_operand=inside_composite_operand
            )
        else:
            operand = _render_operator_stack(
                rest,
                inner,
                enclosing_ir,
                inside_composite_operand=inside_composite_operand,
            )
            rest = []
        if op.bare_prefix:
            # Joiner-free spelling (flux_surface_averaged_ratio_of_A_to_B). The
            # IR restricts the flag to operators that have a bare spelling.
            outer = f"{op.op}_{operand}"
        else:
            assert_operator_of_form(op, registry=None)
            indexed_parts = op.op.partition("_with_respect_to_")
            if indexed_parts[1] and indexed_parts[0] and indexed_parts[2]:
                operator, _, index = indexed_parts
                outer = f"{operator}_of_{operand}_with_respect_to_{index}"
            else:
                outer = f"{op.op}_of_{operand}"
        # Any remaining operators in ``rest`` still need to wrap the result.
        return _render_operator_stack(
            rest,
            outer,
            enclosing_ir,
            inside_composite_operand=inside_composite_operand,
        )

    if op.kind is OperatorKind.UNARY_POSTFIX:
        if op.args:
            operand = _compose(
                op.args[0], inside_composite_operand=inside_composite_operand
            )
        else:
            operand = _render_operator_stack(
                rest,
                inner,
                enclosing_ir,
                inside_composite_operand=inside_composite_operand,
            )
            rest = []
        outer = f"{operand}_{op.op}"
        return _render_operator_stack(
            rest,
            outer,
            enclosing_ir,
            inside_composite_operand=inside_composite_operand,
        )

    if op.kind is OperatorKind.BINARY:
        assert_binary_has_separator(op, registry=None)
        if len(op.args) != 2:  # pragma: no cover - guarded by IR validator
            raise RenderError(
                f"binary operator {op.op!r} requires 2 args, got {len(op.args)}"
            )
        a = _compose(op.args[0], inside_composite_operand=True)
        b = _compose(op.args[1], inside_composite_operand=True)
        outer = f"{op.op}_of_{a}_{op.separator}_{b}"
        return _render_operator_stack(
            rest,
            outer,
            enclosing_ir,
            inside_composite_operand=inside_composite_operand,
        )

    raise RenderError(  # pragma: no cover - StrEnum is exhaustive
        f"unknown operator kind {op.kind!r} for operator {op.op!r}"
    )


def render_operators(
    operators: list[OperatorApplication],
    inner: str,
    enclosing_ir: StandardNameIR | None = None,
) -> str:
    """Public operator-stack renderer.

    ``enclosing_ir`` is optional and used only for error messages. When the
    caller has no enclosing IR (e.g. a test calling this helper directly)
    a placeholder IR may be omitted.
    """

    if not operators:
        return inner
    if enclosing_ir is None:
        # Build a trivial no-op context: we need only identity for diagnostics.
        enclosing_ir = operators[0].args[0] if operators[0].args else None  # type: ignore[assignment]
    return _render_operator_stack(  # type: ignore[arg-type]
        operators,
        inner,
        enclosing_ir,
        inside_composite_operand=False,
    )


def _contains_composite_expression(ir: StandardNameIR) -> bool:
    """Return whether ``ir`` contains a binary operand tree."""

    for operator in ir.operators:
        if operator.kind is OperatorKind.BINARY:
            return True
        if any(_contains_composite_expression(argument) for argument in operator.args):
            return True
    return False


def _has_repositioning_tail(ir: StandardNameIR) -> bool:
    """Return whether ``ir`` carries an ``of``, ``at``, or ``due_to`` tail."""

    if ir.mechanism is not None:
        return True
    return ir.locus is not None and ir.locus.relation.value in {"of", "at"}


def _leading_bare_operators(
    ir: StandardNameIR,
) -> tuple[list[Qualifier], list[Qualifier]]:
    """Split leading bare operators from ordinary base qualifiers."""

    operator_qualifiers: list[Qualifier] = []
    ordinary_qualifiers = list(ir.qualifiers)
    while ordinary_qualifiers and ordinary_qualifiers[0].token in BARE_PREFIX_OPERATORS:
        operator_qualifiers.append(ordinary_qualifiers.pop(0))
    return operator_qualifiers, ordinary_qualifiers


def _can_reposition_operators(
    ir: StandardNameIR,
    *,
    inside_composite_operand: bool,
) -> bool:
    """Return whether local unary operators may sit immediately before the tail."""

    if inside_composite_operand or not _has_repositioning_tail(ir):
        return False
    if _contains_composite_expression(ir):
        return False
    return all(
        operator.kind is not OperatorKind.BINARY
        and not operator.args
        and "_with_respect_to_" not in operator.op
        and operator.op not in BARE_PREFIX_OPERATORS
        for operator in ir.operators
    )


def _compose(
    ir: StandardNameIR,
    *,
    inside_composite_operand: bool,
) -> str:
    """Render one IR while retaining composite-operand context recursively."""

    operator_qualifiers, ordinary_qualifiers = _leading_bare_operators(ir)
    can_reposition = _can_reposition_operators(
        ir, inside_composite_operand=inside_composite_operand
    )
    has_local_operator = bool(ir.operators or operator_qualifiers)

    if can_reposition and has_local_operator:
        base_ir = ir.model_copy(
            update={
                "operators": [],
                "qualifiers": ordinary_qualifiers,
                "locus": None,
                "mechanism": None,
            }
        )
        rendered = _render_base_with_decorators(base_ir)
        operator_tokens = [
            *(operator.op for operator in ir.operators),
            *(qualifier.token for qualifier in operator_qualifiers),
        ]
        for token in reversed(operator_tokens):
            rendered += f"_{token}"
        rendered += render_locus(ir.locus)
        rendered += render_mechanism(ir.mechanism)
        return rendered

    inner = _render_base_with_decorators(ir)
    return _render_operator_stack(
        list(ir.operators),
        inner,
        ir,
        inside_composite_operand=inside_composite_operand,
    )


# ---------------------------------------------------------------------------
# Top-level compose
# ---------------------------------------------------------------------------


def compose(ir: StandardNameIR) -> str:
    """Render ``ir`` into its single canonical string form.

    Raises :class:`RenderError` when the IR is structurally inconsistent
    beyond what the Pydantic validators already catch (e.g. a trailing
    locus that the operator stack has displaced).
    """

    if not isinstance(ir, StandardNameIR):  # pragma: no cover - type guard
        raise RenderError(
            f"compose() expects a StandardNameIR, got {type(ir).__name__}"
        )

    if ir.locus is not None and any(
        operator.kind is OperatorKind.BINARY for operator in ir.operators
    ):
        raise RenderError(
            "a locus on an enclosing binary expression is ambiguous with a "
            "locus on its final operand; attach the locus to an operand"
        )

    try:
        rendered = _compose(ir, inside_composite_operand=False)
    except RenderError:
        raise
    except ValueError as exc:
        raise RenderError(str(exc)) from exc

    # Binary operators replace `inner` entirely, losing any mechanism that was
    # rendered into `inner` by _render_base_with_decorators. Re-append it from
    # the top-level IR. A top-level locus is rejected above because its flat
    # spelling cannot distinguish it from a final-operand locus.
    if any(op.kind is OperatorKind.BINARY for op in ir.operators):
        rendered += render_mechanism(ir.mechanism)

    # Safety net: enforce the trailing-locus rule on the final string.
    # When the outermost operator pushes text after the locus suffix, the
    # resulting name violates the trailing-position invariant and must be
    # rejected rather than emitted.
    if ir.locus is not None and (
        not ir.operators
        or _can_reposition_operators(ir, inside_composite_operand=False)
    ):
        try:
            assert_locus_is_trailing(rendered, ir)
        except ValueError as exc:
            raise RenderError(str(exc)) from exc
    return rendered
