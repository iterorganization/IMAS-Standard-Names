"""Model-layer ↔ parser-layer parity for every registered operator.

The strict IR parser (``parse(..., strict=True)``/``render.compose``) is the
lossless validity oracle. The flat ``StandardName`` facade
(``parse_standard_name``/``compose_standard_name``) projects the subset it can
represent. The flat facade stores indexed prefix operators such as
``derivative_with_respect_to_<coord>`` as fused tokens that are not members of
the closed ``Transformation`` / ``Decomposition`` StrEnums. The facade and
ordered IR must nevertheless render the same canonical name.

This module provides an exhaustive, data-driven parity check: for
every operator declared in ``operators.yml``, build a representative canonical
name and assert that

    compose_standard_name(parse_standard_name(name)) == name      (model round-trip)
    compose_standard_name(parse_standard_name(name)) == render(parse(name).ir)  (parity)

Driving the parametrization from the loaded operator registry keeps coverage
complete as that registry changes and makes any disagreement fail at the
grammar boundary.
"""

from __future__ import annotations

import pytest

from imas_standard_names import StandardNameIR, compose, parse
from imas_standard_names.grammar.ir import OperatorKind
from imas_standard_names.grammar.model import (
    compose_standard_name,
    parse_standard_name,
)
from imas_standard_names.grammar.parser import (
    Vocabularies,
    load_default_vocabularies,
)

# Representative bases. A scalar base for most operators; a vector base for the
# scalar-extraction postfixes (magnitude, real_part, imaginary_part) that
# require vector/complex structure to act on.
_SCALAR_BASE = "pressure"
_VECTOR_BASE = "magnetic_field"
_VECTOR_ONLY_POSTFIX = frozenset({"magnitude", "real_part", "imaginary_part"})


def _candidate_names(op: str, meta: dict) -> list[str]:
    """Representative canonical name(s) to try for one operator.

    Returns an ordered list of candidates; the first that round-trips through
    both layers is the one asserted on. The variants cover the structural
    forms: indexed-prefix (``derivative_of_<base>_with_respect_to_<coord>``),
    bare-vs-``_of_``
    prefix, postfix tail, and binary ``<op>_of_<A>_<sep>_<B>``.
    """
    kind = meta["kind"]
    if kind == OperatorKind.BINARY.value:
        sep = meta.get("separator")
        return [f"{op}_of_velocity_{sep}_magnetic_field"] if sep else []
    if kind == OperatorKind.UNARY_POSTFIX.value:
        base = _VECTOR_BASE if op in _VECTOR_ONLY_POSTFIX else _SCALAR_BASE
        return [f"{base}_{op}"]
    if kind == OperatorKind.UNARY_PREFIX.value:
        if meta.get("flux_surface_reduction"):
            # pressure is a flux function (constant_on_flux_surface), so the
            # flux-surface reductions gate it out — use a surface-varying base.
            return [f"{op}_of_temperature", f"{op}_temperature"]
        if meta.get("indexed") and list(meta.get("index_params") or []) == ["coord"]:
            operator, relation, _ = op.partition("_with_respect_to")
            return [f"{operator}_of_{_SCALAR_BASE}{relation}_radial_coordinate"]
        # Some prefix operators render with ``_of_`` (gradient_of_pressure);
        # the bare-prefix family renders without it (normalized_pressure). Try
        # both and keep whichever round-trips.
        return [f"{op}_of_{_SCALAR_BASE}", f"{op}_{_SCALAR_BASE}"]
    return []


@pytest.fixture(scope="module")
def vocabs() -> Vocabularies:
    return load_default_vocabularies()


def _all_operators() -> list[tuple[str, dict]]:
    return sorted(load_default_vocabularies().operators.items())


@pytest.mark.parametrize(
    "op,meta", _all_operators(), ids=lambda x: x if isinstance(x, str) else ""
)
def test_every_operator_model_parser_parity(
    op: str, meta: dict, vocabs: Vocabularies
) -> None:
    """Each registered operator round-trips at the model layer and the model
    path agrees with the parser path."""
    candidates = _candidate_names(op, meta)
    assert candidates, f"no candidate name form for operator {op!r} ({meta['kind']})"

    last_error: Exception | None = None
    for name in candidates:
        try:
            model_round_trip = compose_standard_name(parse_standard_name(name))
            parser_render = compose(parse(name, vocabs=vocabs).ir)
        except Exception as exc:  # noqa: BLE001 — record and try the next form
            last_error = exc
            continue
        if model_round_trip == name and parser_render == name:
            # Parity: the two layers produce the identical canonical string.
            assert model_round_trip == parser_render
            return

    pytest.fail(
        f"operator {op!r} ({meta['kind']}) did not round-trip through the model "
        f"layer for any candidate {candidates!r}; last error: {last_error!r}"
    )


def test_public_ir_api_preserves_outermost_first_operator_chain() -> None:
    name = "flux_surface_averaged_inverse_of_square_of_major_radius"
    parsed = parse(name, strict=True)

    assert isinstance(parsed.ir, StandardNameIR)
    assert [operator.op for operator in parsed.ir.operators] == [
        "flux_surface_averaged",
        "inverse",
        "square",
    ]
    assert compose(parsed.ir) == name


def test_strict_ir_is_validity_oracle_when_flat_facade_cannot_project() -> None:
    name = "gradient_of_time_derivative_of_electron_temperature"

    assert compose(parse(name, strict=True).ir) == name
    with pytest.raises(ValueError, match="flat StandardName model"):
        parse_standard_name(name)


def test_flat_facade_uses_strict_validity_contract() -> None:
    name = "inverse_of_volume_averaged_electron_temperature"

    with pytest.raises(ValueError, match="precedence"):
        parse_standard_name(name)


def test_strict_oracle_reuses_flat_segment_semantics() -> None:
    with pytest.raises(ValueError, match="compatible species subject"):
        parse("charge_state_density", strict=True)
