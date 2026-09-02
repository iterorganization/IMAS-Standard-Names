"""SPA dataset builder for the IMAS Standard Names catalog.

Converts a directory of per-domain YAML files (the published catalog
format) into a single JSON dataset consumed by the redesigned SPA. The
SPA loads:

* ``CATALOG_VERSION`` — human-readable catalog identifier
* ``CATEGORIES`` — list of ``{id, label, count}`` per physics_domain
* ``GRAMMAR_VOCAB`` — token lists per UI vocabulary section
* ``STANDARD_TERMS`` — governed compositional terms and definitions
* ``NAMES`` — flat array of records with the grammar-derived ``parse``
  decomposition pre-computed by the ISN Python parser (no JS
  heuristic).

The output is consumed by the SPA's read-only renderer. Each NAMES
record carries:

* identity: ``name``, ``category``, ``group``, ``parent``
* algebraic ``algebra`` (``scalar | vector | tensor | complex | metadata``)
  declared on the catalog entry
* physical metadata: ``unit``, ``tags``, ``axis``, ``locus``
* prose: ``short`` (description), ``long`` (documentation minus the
  ``Sign convention:`` paragraph), ``sign`` (the extracted paragraph)
* navigation: ``seeAlso`` (links normalised, ``name:`` prefix stripped),
  ``arguments`` (just the argument names), ``sources`` (source-system bindings
  plus any metadata resolved from their pinned source version),
  ``superseded_by`` (name of replacement or
  ``null``), ``deprecates`` (name being deprecated or ``null``)
* ``parse`` — a list of role/text/note segments (operators, qualifiers,
  axis, base, locus, process) for the UI to render as chips.

Status filtering
----------------
``build_site_dataset`` emits every entry whose normalised status is one
of the four canonical values: ``active``, ``draft``, ``deprecated``,
``superseded``.

Legacy status values are normalised before filtering:

* ``"drafted"``   → ``"draft"``
* ``"accepted"``  → ``"active"``
* ``"published"`` → ``"active"``

Unknown values are logged as warnings and the entry is dropped.
"""

import importlib
import json
import logging
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from imas_standard_names.grammar import Subject, vocab_loaders
from imas_standard_names.grammar.model_types import Aggregation, Orbit, Population
from imas_standard_names.grammar.parser import (
    ParseError,
    compose,
    parse,
)
from imas_standard_names.grammar.terms import standard_terms
from imas_standard_names.models import (
    StandardNameCatalogManifest,
    StandardNameEntryBase,
)

_log = logging.getLogger(__name__)

__all__ = [
    "build_site_dataset",
    "write_site_dataset",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# Prefix operator tokens classified as "reductions" — used by the grammar
# vocab builder below (``_build_grammar_vocab``) to populate the
# ``"reduction"`` chip rail in the SPA.  NOT used for display_kind
# (which has been removed; only ``algebra`` is emitted on each record).
_REDUCTION_PREFIX_OPS: frozenset[str] = frozenset(
    {
        "maximum",
        "minimum",
        "maximum_over_flux_surface",
        "minimum_over_flux_surface",
        "volume_integrated",
        "surface_integrated",
        "line_integrated",
        "accumulated",
        "cumulative",
        "cumulative_inside_flux_surface",
        "time_averaged",
        "root_mean_square",
    }
)


_SUBJECT_TOKENS: frozenset[str] = frozenset(member.value for member in Subject)
_ORBIT_TOKENS: frozenset[str] = frozenset(member.value for member in Orbit)
_POPULATION_TOKENS: frozenset[str] = frozenset(member.value for member in Population)
_AGGREGATION_TOKENS: frozenset[str] = frozenset(member.value for member in Aggregation)
# Zone (ordered plasma-region / geometric sub-selector prefix, possibly
# multiple per name) and channel (single transport-channel prefix) tokens.
# Both peel through ``ir.qualifiers`` in the parser; the dataset classifies
# them into dedicated ``zone`` / ``channel`` parse roles so the SPA renders
# them as their own grammar segments in canonical position.
_ZONE_TOKENS: frozenset[str] = frozenset(vocab_loaders.load_zones())
_CHANNEL_TOKENS: frozenset[str] = frozenset(vocab_loaders.load_channels())
_CHANNEL_QUALIFIER_TOKENS: frozenset[str] = frozenset(
    vocab_loaders.load_channel_qualifiers()
)


# Coordinate-axis ordering for sort_axis_index emission.
# An axis-projected family member sorts by this index within the component
# tier. Its catalog kind remains independent: a component value can be scalar
# even though its parent quantity is a vector.
_AXIS_ORDER: dict[str, int] = {
    "radial": 0,
    "toroidal": 1,
    "vertical": 2,
    "poloidal": 3,
    "parallel": 4,
    "perpendicular": 5,
}

# Operator tokens that signal a domain-wide aggregation (tier 3). The
# parser surfaces these as ``qualifier_tokens``; we recompute the
# "reduction qualifier" flag inline since :class:`_GrammarFacets` no
# longer caches it (was used by the retired display_kind heuristic).
_AGGREGATION_PREFIXES: frozenset[str] = frozenset(
    {
        "total",
        "minimum",
        "maximum",
        "effective",
    }
)


def _extract_subject(qualifier_tokens: tuple[str, ...]) -> str | None:
    """Return the first qualifier token matching the Subject enum, if any.

    Drives the SPA's subject filter so users can slice by species
    (``electron``, ``ion``, ``deuterium``, …) without resorting to
    free-text search.
    """
    for token in qualifier_tokens:
        if token in _SUBJECT_TOKENS:
            return token
    return None


# Match the ``Sign convention: Positive ...`` paragraph. We accept both
# real newline separators (``\n\n``) and literal backslash-n escapes
# (``\\n\\n``) — a handful of catalog entries were YAML-encoded with
# single quotes that left the escapes uninterpreted. The sentence runs
# until the next paragraph break (real ``\n\n`` or literal ``\\n\\n``)
# or the end of the documentation string.
_SIGN_CONVENTION_RE = re.compile(
    r"(?:^|\n\n|\\n\\n)"
    r"Sign convention:\s*Positive"
    r"(?:(?!\n\n|\\n\\n).)*"
    r"(?=$|\n\n|\\n\\n)",
    re.DOTALL,
)


_NAME_LINK_RE = re.compile(r"^name:([a-z0-9_]+)$")


# ---------------------------------------------------------------------------
# Helpers — humanisation
# ---------------------------------------------------------------------------


# Selected abbreviations matching the SPA prototype's compact labels.
_LABEL_ABBREVIATIONS: dict[str, str] = {
    "measurement": "Meas.",
    "electromagnetic": "EM",
}


def _humanise_domain(slug: str) -> str:
    """Convert a physics_domain slug into a human-readable label.

    Most slugs ``snake_case`` → ``Title Case`` ("auxiliary_heating" →
    "Auxiliary Heating"). Selected long words are abbreviated to match
    the SPA prototype's compact sidebar labels.
    """
    if not slug:
        return ""
    words: list[str] = []
    for token in slug.split("_"):
        if not token:
            continue
        abbreviated = _LABEL_ABBREVIATIONS.get(token)
        if abbreviated is not None:
            words.append(abbreviated)
        else:
            words.append(token.capitalize())
    return " ".join(words)


def _humanise(token: str) -> str:
    """Convert a snake_case token to space-separated lowercase words.

    Used for group titles ("magnetic_field" → "magnetic field") so the
    SPA can cluster sibling names without further string handling.
    """
    return token.replace("_", " ") if token else ""


# ---------------------------------------------------------------------------
# Helpers — sign convention / documentation
# ---------------------------------------------------------------------------


def _extract_sign(documentation: str) -> tuple[str, str | None]:
    """Strip the Sign convention paragraph and return (long, sign).

    The validator enforces ``Sign convention:`` as a standalone
    paragraph; we capture the entire sentence (which begins with
    ``Positive``) and remove it (and the leading blank line) from the
    main documentation text.
    """
    if not documentation:
        return documentation or "", None

    match = _SIGN_CONVENTION_RE.search(documentation)
    if match is None:
        return documentation, None

    # Strip leading separators (real ``\n\n`` or literal ``\\n\\n``) so
    # the captured sentence starts cleanly with "Sign convention:".
    sign_text = match.group(0)
    sign_text = re.sub(r"^(?:\\n\\n|\n\n)", "", sign_text).strip()
    # Strip the "Sign convention: " prefix so the SPA gets just the
    # human-readable rule. Keep the trailing period.
    sign_value = re.sub(
        r"^Sign convention:\s+", "", sign_text, count=1, flags=re.IGNORECASE
    ).strip()

    # Remove the matched span (including its leading separator) from
    # the documentation, then collapse any resulting triple newline.
    start, end = match.span()
    stripped = documentation[:start] + documentation[end:]
    stripped = re.sub(r"\n{3,}", "\n\n", stripped).strip()
    return stripped, sign_value or None


# ---------------------------------------------------------------------------
# Helpers — links / sources / arguments
# ---------------------------------------------------------------------------


def _normalise_see_also(links: list[str] | None) -> list[str]:
    """Filter ``links`` to internal ``name:foo`` refs (returned without prefix)."""
    if not links:
        return []
    result: list[str] = []
    for link in links:
        if not isinstance(link, str):
            continue
        match = _NAME_LINK_RE.match(link.strip())
        if match:
            result.append(match.group(1))
    return result


@cache
def _dd_factory(dd_version: str) -> Any:
    """Return the cached imas-python factory for one pinned DD version."""
    imas = importlib.import_module("imas")
    return imas.IDSFactory(dd_version)


@cache
def _dd_metadata_root(dd_version: str, ids_name: str) -> Any:
    """Return the metadata root for one IDS in a pinned DD version."""
    return _dd_factory(dd_version).new(ids_name).metadata


def _resolve_dd_source(path: str, dd_version: str) -> dict[str, Any]:
    """Resolve one pinned DD source through imas-python metadata."""
    ids_name, separator, relative_path = path.partition("/")
    if not separator or not ids_name or not relative_path:
        raise ValueError(f"DD path must include an IDS name and leaf path: {path!r}")

    ids_path = importlib.import_module("imas.ids_path").IDSPath(relative_path)
    metadata = ids_path.goto_metadata(_dd_metadata_root(dd_version, ids_name))
    parent = metadata._parent
    parent_relative_path = str(parent.path)
    parent_path = (
        f"{ids_name}/{parent_relative_path}" if parent_relative_path else ids_name
    )
    data_type = getattr(metadata.data_type, "value", str(metadata.data_type))
    return {
        "leaf_definition": metadata.documentation,
        "parent_path": parent_path,
        "parent_definition": parent.documentation,
        "data_type": data_type,
        "unit": metadata.units,
        "coordinates": [str(coordinate) for coordinate in metadata.coordinates],
        "resolution_source": "imas-python",
        "resolution_status": "resolved",
    }


def _legacy_source_projection(raw: dict[str, Any]) -> dict[str, Any]:
    """Project legacy inline DD fields when no pinned version is available."""
    projected: dict[str, Any] = {}
    authoritative = raw.get("dd_documentation")
    if not isinstance(authoritative, dict):
        authoritative = {}
    enhanced = raw.get("enhanced_context")
    if not isinstance(enhanced, dict):
        enhanced = {}
    aliases = {
        "leaf_definition": ("leaf_definition", "documentation"),
        "parent_path": ("parent_path",),
        "parent_definition": ("parent_definition", "parent_documentation"),
        "data_type": ("data_type",),
        "unit": ("unit",),
        "coordinates": ("coordinates",),
        "lifecycle": ("lifecycle", "dd_lifecycle"),
        "enhanced_context": ("enhanced_context",),
        "enhancement_kind": ("enhancement_kind",),
    }
    for public_key, candidate_keys in aliases.items():
        value = next(
            (raw.get(key) for key in candidate_keys if raw.get(key) not in (None, "")),
            None,
        )
        if value is not None:
            projected[public_key] = value
    context = projected.get("enhanced_context")
    if isinstance(context, dict):
        description = context.get("description")
        if isinstance(description, str) and description:
            projected["enhanced_context"] = description
        else:
            projected.pop("enhanced_context")
    nested_authoritative = {
        "leaf_definition": authoritative.get("leaf"),
        "parent_path": authoritative.get("parent_path"),
        "parent_definition": authoritative.get("parent"),
        "data_type": authoritative.get("data_type"),
        "unit": authoritative.get("unit"),
        "coordinates": authoritative.get("coordinates"),
        "lifecycle": authoritative.get("lifecycle_status"),
        "lifecycle_version": authoritative.get("lifecycle_version"),
    }
    for key, value in nested_authoritative.items():
        if value not in (None, "", []):
            projected[key] = value
    if enhanced.get("description"):
        projected["enhanced_context"] = enhanced["description"]
    if enhanced.get("kind"):
        projected["enhancement_kind"] = enhanced["kind"]
    if projected:
        projected["resolution_source"] = "catalog-inline"
        projected["resolution_status"] = "legacy-inline"
    return projected


def _normalise_sources(sources: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Resolve source bindings and strip producer-only provenance fields.

    Retired ``dd_path`` / ``dd_version`` records are translated at the read
    boundary. Only ``imas-dd`` dispatches to imas-python; every other kind is
    preserved as an unresolved binding for a source-specific resolver.
    """
    if not sources:
        return []
    normalised: list[dict[str, Any]] = []
    for raw in sources:
        if not isinstance(raw, dict):
            continue
        kind = raw.get("kind") or ""
        ref = raw.get("ref") or raw.get("dd_path") or raw.get("signal_id") or ""
        if not ref:
            ident = raw.get("id") or ""
            if isinstance(ident, str) and ident.startswith("dd:"):
                kind = kind or "imas-dd"
                ref = ident[len("dd:") :]
        if not kind and raw.get("dd_path") not in (None, ""):
            kind = "imas-dd"
        if not kind or not ref:
            continue
        version = raw.get("version") or raw.get("dd_version")
        projected: dict[str, Any] = {
            "kind": str(kind),
            "ref": str(ref),
        }
        if version not in (None, ""):
            version_text = str(version)
            projected["version"] = version_text
        else:
            version_text = ""

        if kind == "imas-dd" and version_text:
            try:
                projected.update(_resolve_dd_source(str(ref), version_text))
            except Exception as exc:
                projected.update(
                    {
                        "resolution_source": "imas-python",
                        "resolution_status": "unresolved",
                        "resolution_error": f"{type(exc).__name__}: {exc}",
                    }
                )
        elif kind == "imas-dd":
            projected.update(_legacy_source_projection(raw))
        else:
            projected.update(
                {
                    "resolution_status": "unresolved",
                    "resolution_error": f"No resolver registered for source kind {kind!r}",
                }
            )
        normalised.append(projected)
    return normalised


def _normalise_arguments(arguments: list[dict[str, Any]] | None) -> list[str]:
    """Flatten ``ArgumentRef`` entries to their ``name`` strings."""
    if not arguments:
        return []
    flat: list[str] = []
    for arg in arguments:
        if isinstance(arg, dict):
            ref = arg.get("name")
            if isinstance(ref, str) and ref:
                flat.append(ref)
        elif isinstance(arg, str):
            flat.append(arg)
    return flat


# ---------------------------------------------------------------------------
# Helpers — grammar IR
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _GrammarFacets:
    """Facets derived from the grammar IR for a single name.

    These drive ``parent``, ``axis``, ``locus``, grammar-derived
    ``tags``, and ``parse`` segments. Holding them in a single dataclass
    keeps the helper logic testable and clear.
    """

    parsed: bool
    parse_segments: list[dict[str, str]]
    base_token: str | None
    axis: str | None
    locus_token: str | None
    has_projection: bool
    has_locus: bool
    qualifier_tokens: tuple[str, ...]
    operator_tokens: tuple[str, ...]
    has_mechanism: bool


def _derive_grammar_facets(name: str) -> _GrammarFacets:
    """Parse ``name`` and extract everything the SPA record needs.

    Falls back to an "unparseable" record so the dataset never crashes
    on a malformed entry. The single ``parse`` segment in that case
    lets the SPA still render a (visually distinct) chip.
    """
    try:
        ir = parse(name).ir
    except ParseError:
        return _GrammarFacets(
            parsed=False,
            parse_segments=[
                {
                    "role": "unparseable",
                    "text": name,
                    "note": "Parser could not decompose",
                }
            ],
            base_token=None,
            axis=None,
            locus_token=None,
            has_projection=False,
            has_locus=False,
            qualifier_tokens=(),
            operator_tokens=(),
            has_mechanism=False,
        )

    segments: list[dict[str, str]] = []

    # Prefix operators (outermost first). The IR stores operators in
    # outer-to-inner order, which is the order we want to render.
    operator_tokens: list[str] = []
    for op in ir.operators:
        operator_tokens.append(op.op)
        if op.kind.value == "unary_prefix":
            segments.append(
                {
                    "role": "operator",
                    "text": op.op,
                    "note": "Prefix operator",
                }
            )
        elif op.kind.value == "binary":
            segments.append(
                {
                    "role": "operator",
                    "text": op.op,
                    "note": "Binary operator",
                }
            )
        else:
            # postfix — keep in the operators bucket but render later
            # (after the base) so its position in the chip strip reflects
            # the canonical written form.
            pass

    # Axis projection (before qualifiers in canonical render).
    axis: str | None = None
    if ir.projection is not None:
        axis = ir.projection.axis
        shape = ir.projection.shape.value
        segments.append(
            {
                "role": "axis",
                "text": axis,
                "note": f"Axis {shape}",
            }
        )

    # Qualifiers (insertion order matches parse order).
    qualifier_tokens: list[str] = []
    for qualifier in ir.qualifiers:
        token = qualifier.token
        qualifier_tokens.append(token)
        # Classify into dedicated single-token segments so the SPA renders
        # distinct, filterable cards in the canonical decomposition.
        if token in _AGGREGATION_TOKENS:
            role, note = "aggregation", "Aggregation (total / net)"
        elif token in _ORBIT_TOKENS:
            role, note = "orbit", "Orbit / transit class"
        elif token in _POPULATION_TOKENS:
            role, note = "population", "Species population (energy-state, …)"
        elif token in _SUBJECT_TOKENS:
            role, note = "subject", "Species the quantity applies to"
        elif token in _ZONE_TOKENS:
            role, note = "zone", "Plasma-region / geometric sub-selector"
        elif token in _CHANNEL_QUALIFIER_TOKENS:
            role, note = "channel_qualifier", "Channel qualifier (binds to channel)"
        elif token in _CHANNEL_TOKENS:
            role, note = "channel", "Transport channel (what is transported)"
        else:
            role, note = "qualifier", "Qualifier"
        segments.append(
            {
                "role": role,
                "text": token,
                "note": note,
            }
        )

    # Base (always present).
    base_token = ir.base.token
    segments.append(
        {
            "role": "base",
            "text": base_token,
            "note": ir.base.kind.value,
        }
    )

    # Postfix operators render after the base in canonical form.
    for op in ir.operators:
        if op.kind.value == "unary_postfix":
            segments.append(
                {
                    "role": "operator",
                    "text": op.op,
                    "note": "Postfix operator",
                }
            )

    # Locus (trailing position).
    locus_token: str | None = None
    if ir.locus is not None:
        locus_token = ir.locus.token
        relation = ir.locus.relation.value
        locus_text = f"{relation}_{ir.locus.token}"
        if ir.locus.value is not None:
            # Value-parameterized position: at_<token>_equal_to_<value>.
            locus_text += f"_equal_to_{ir.locus.value}"
        segments.append(
            {
                "role": "locus",
                "text": locus_text,
                "note": ir.locus.type.value,
            }
        )

    # Mechanism (always last when present).
    has_mechanism = ir.mechanism is not None
    if has_mechanism:
        segments.append(
            {
                "role": "process",
                "text": f"due_to_{ir.mechanism.token}",
                "note": "Mechanism",
            }
        )

    return _GrammarFacets(
        parsed=True,
        parse_segments=segments,
        base_token=base_token,
        axis=axis,
        locus_token=locus_token,
        has_projection=ir.projection is not None,
        has_locus=ir.locus is not None,
        qualifier_tokens=tuple(qualifier_tokens),
        operator_tokens=tuple(operator_tokens),
        has_mechanism=has_mechanism,
    )


def _sort_tier(
    name: str,
    algebra: str,
    parent: str | None,
    facets: _GrammarFacets,
) -> int:
    """Compute the canonical sort tier for a name (0–7).

    Drives result-list ordering inside a cluster group, so the family
    reads top-to-bottom as base → components → magnitude → aggregation
    → operator-derived → locus-evaluated → metadata → variant.

    The classifier derives from grammar IR fields where available; tiers 2
    (magnitude / norm) and 4 (gradient / shear / etc.) use the
    parser's ``operator_tokens`` as the primary signal, with substring
    tests on the name string as a defensive fallback for unparseable
    names.
    """
    if algebra == "metadata":
        return 6
    if parent is None and algebra in {"vector", "tensor", "complex"}:
        return 0
    if facets.has_projection:
        return 1
    # Tier 2 — magnitude / norm. Driven by the parser's classified
    # postfix operator tokens, not by name-string regex.
    if any(op in {"magnitude", "norm"} for op in facets.operator_tokens):
        return 2
    # Regex fallback for unparseable names.
    if "_magnitude_" in f"_{name}_" or "_norm_" in f"_{name}_":
        return 2
    # Tier 3 — aggregation. Either a known reduction prefix (from the
    # ``_REDUCTION_PREFIX_OPS`` set already used by the vocab builder)
    # appears as a qualifier, or the name carries one of the
    # subject-style aggregation prefixes (total/minimum/maximum/…).
    if any(q in _REDUCTION_PREFIX_OPS for q in facets.qualifier_tokens):
        return 3
    if any(name.startswith(p + "_") for p in _AGGREGATION_PREFIXES):
        return 3
    # Tier 4 — operator-style derived. Primary: parser-classified
    # operator tokens. Fallback: substring tests for unparseable names.
    if any(
        op in {"gradient", "shear", "divergence", "curl", "density"}
        for op in facets.operator_tokens
    ):
        return 4
    for token in ("_gradient", "_shear", "_divergence", "_curl", "_density"):
        if token in f"_{name}_":
            return 4
    # Tier 5 — point evaluation at a locus.
    if facets.has_locus:
        return 5
    # Tier 7 — variant / unclassified scalars (base scalars, etc.).
    return 7


def _sort_axis_index(facets: _GrammarFacets) -> int:
    """Return the axis index (0–5) for a projection name, 99 otherwise."""
    if facets.axis is None:
        return 99
    return _AXIS_ORDER.get(facets.axis, 99)


# ---------------------------------------------------------------------------
# Helpers — parent / group
# ---------------------------------------------------------------------------


def _parent_token(
    name: str, facets: _GrammarFacets, entry: dict[str, Any] | None = None
) -> str | None:
    """Return the structural parent name (one layer peeled), if any.

    Resolution order — preferring the canonical pipeline-derived parent
    over heuristic local reconstruction:

    1. **``arguments[0].name``** from the YAML entry, when present and
       not a self-loop. The catalog exporter emits ``arguments`` from
       the graph's outgoing ``COMPONENT_OF`` edges (one per name, for
       unary peels; per-role for binary), which is the canonical
       structural-parent signal. Using it here means imas-codex's
       single source of truth — the derivation module — governs what
       the SPA shows as parent.

    2. **One-layer IR peel** (operator → projection → qualifier →
       locus) when no ``arguments`` field is provided (standalone
       catalog use, or pre-pipeline entries). This mirrors the
       imas-codex derivation so the SPA stays self-consistent.

    3. **``None``** for true leaves and unparseable names.

    A direct ``facets.base_token`` shortcut would jump past every structural
    layer and make `upper_elongation_of_plasma_boundary` report `elongation`
    as its parent. Peeling exactly one layer preserves the boundary variants;
    recursion is implicit because each parent recomputes its own peel.
    """
    # --- (1) Canonical pipeline-derived parent from YAML arguments ---
    if entry is not None:
        canonical = _arguments_parent(name, entry)
        if canonical is not None:
            return canonical

    # --- (2) Local IR peel — matches imas-codex/derivation.py ordering ---
    local = _local_ir_peel(name)
    if local is not None:
        return local

    # --- (3) Legacy fallback (preserved for unparseable names) ---
    if not facets.parsed or facets.base_token is None:
        return None
    if facets.base_token == name:
        return None
    return facets.base_token


def _arguments_parent(name: str, entry: dict[str, Any]) -> str | None:
    """Extract the first non-self argument target from the YAML entry."""
    args = entry.get("arguments")
    if not isinstance(args, list):
        return None
    for arg in args:
        if not isinstance(arg, dict):
            continue
        target = arg.get("name")
        if isinstance(target, str) and target and target != name:
            return target
    return None


def _local_ir_peel(name: str) -> str | None:
    """Peel ONE structural layer from *name* using the ISN IR parser.

    Mirrors imas-codex/imas_codex/standard_names/derivation.py logic
    so that catalogs built outside the imas-codex pipeline still
    surface accurate parents in the SPA.

    Returns ``None`` when the name is a leaf, unparseable, or when
    the peeled inner name fails to round-trip.
    """
    try:
        result = parse(name)
    except Exception:
        return None

    ir = result.ir
    stripped = None
    if ir.operators:
        # Outermost operator: drop the head of ir.operators
        stripped = ir.model_copy(update={"operators": ir.operators[1:]})
    elif ir.projection is not None:
        stripped = ir.model_copy(update={"projection": None})
    elif ir.qualifiers:
        # Outermost qualifier — covers upper/lower/inner/outer/electron/ion/…
        stripped = ir.model_copy(update={"qualifiers": ir.qualifiers[1:]})
    elif ir.locus is not None:
        stripped = ir.model_copy(update={"locus": None})
    else:
        return None  # leaf

    try:
        inner = compose(stripped)
    except Exception:
        return None

    if not inner or inner == name:
        return None
    return inner


def _group_title(name: str, facets: _GrammarFacets) -> str:
    """Compute the locus-first group title for SPA list clustering.

    Priority order:

    1. Locus token — siblings clustered by structural locality
    2. Base token — siblings clustered by physical quantity
    3. Falls back to ``"other quantities"`` for unparseable names.

    The returned string is humanised (snake_case → space-separated
    lowercase) so the SPA can render it directly without further string
    handling.
    """
    if facets.parsed:
        if facets.locus_token is not None:
            return _humanise(facets.locus_token)
        if facets.base_token is not None:
            return _humanise(facets.base_token)
    return "other quantities"


# ---------------------------------------------------------------------------
# Helpers — tag derivation
# ---------------------------------------------------------------------------


def _derive_tags(
    entry_tags: list[str] | None,
    facets: _GrammarFacets,
) -> list[str]:
    """Return the entry's explicit tags, falling back to grammar-derived ones.

    Many catalog entries omit the ``tags`` field. The SPA still wants
    something descriptive; we synthesise a small set of tags from the
    IR (e.g. ``component``, ``locus``, ``averaged``, ``magnitude``) to
    keep the sidebar useful when the catalog has not yet been hand-tagged.
    """
    if entry_tags:
        return [str(t) for t in entry_tags if isinstance(t, str)]

    derived: list[str] = []
    if facets.has_projection:
        derived.append("component")
    if facets.has_locus:
        derived.append("locus")
    if "magnitude" in facets.operator_tokens:
        derived.append("magnitude")
    if any(q.endswith("averaged") for q in facets.qualifier_tokens):
        derived.append("averaged")
    if any(q.endswith("integrated") for q in facets.qualifier_tokens):
        derived.append("integrated")
    if facets.has_mechanism:
        derived.append("mechanism")
    return derived


# ---------------------------------------------------------------------------
# Record builder
# ---------------------------------------------------------------------------


def _normalise_status(raw: str | None) -> str | None:
    """Map legacy status values to the canonical set; return None to drop.

    Canonical values (pass through unchanged):
        ``"draft"``, ``"active"``, ``"deprecated"``, ``"superseded"``

    Legacy mappings:
        ``"drafted"``   → ``"draft"``
        ``"accepted"``  → ``"active"``
        ``"published"`` → ``"active"``

    Unknown values are logged as warnings and ``None`` is returned so
    the entry is silently dropped from the emitted dataset.
    """
    if not raw:
        return "draft"
    _LEGACY: dict[str, str] = {
        "drafted": "draft",
        "accepted": "active",
        "published": "active",
    }
    _CANONICAL: frozenset[str] = frozenset(
        {"draft", "active", "deprecated", "superseded"}
    )
    if raw in _CANONICAL:
        return raw
    mapped = _LEGACY.get(raw)
    if mapped is not None:
        return mapped
    _log.warning("unknown status %r — entry will be dropped", raw)
    return None


def _build_record(entry: dict[str, Any]) -> dict[str, Any]:
    """Build one NAMES record from a parsed YAML entry.

    Emits ``algebra`` (scalar / vector / tensor / complex / metadata)
    from the catalog entry's ``kind`` field.  The synthetic
    ``display_kind`` / ``kind`` fields are no longer emitted; the SPA
    reads only ``algebra`` for kind-based filtering and badging.

    The ``status`` field on the entry is expected to have already been
    normalised by ``_normalise_status`` before this function is called.
    """
    name = str(entry.get("name") or "")
    domain_type = StandardNameEntryBase.model_fields["physics_domain"].annotation
    category = domain_type(entry["physics_domain"])
    facets = _derive_grammar_facets(name)
    algebra = entry.get("kind") or "scalar"
    parent = _parent_token(name, facets, entry)
    sort_tier = _sort_tier(name, str(algebra), parent, facets)
    sort_axis_index = _sort_axis_index(facets)
    group = _group_title(name, facets)
    tags = _derive_tags(entry.get("tags"), facets)

    # Status has already been normalised by build_site_dataset.
    status = entry.get("status") or "draft"
    # Subject — first qualifier token matching the closed Subject enum
    # (electron, ion, deuterium, …). Drives the SPA's subject filter so
    # users can slice by species without text-search.
    subject = _extract_subject(facets.qualifier_tokens)

    short = entry.get("description") or ""
    long_text, sign = _extract_sign(entry.get("documentation") or "")
    see_also = _normalise_see_also(entry.get("links"))
    sources = _normalise_sources(entry.get("sources"))
    arguments = _normalise_arguments(entry.get("arguments"))
    superseded_by = entry.get("superseded_by") or None
    deprecates = entry.get("deprecates") or None

    record: dict[str, Any] = {
        "name": name,
        "category": str(category),
        "group": group,
        "parent": parent,
        "algebra": str(algebra),
        "status": str(status),
        "subject": str(subject) if subject else None,
        "unit": entry.get("unit") or "",
        "tags": tags,
        "short": short,
        "long": long_text,
        "sign": sign,
        "seeAlso": see_also,
        "arguments": arguments,
        "sources": sources,
        "superseded_by": superseded_by,
        "deprecates": deprecates,
        "parse": facets.parse_segments,
        "sort_tier": sort_tier,
        "sort_axis_index": sort_axis_index,
    }
    if facets.axis is not None:
        record["axis"] = facets.axis
    if facets.locus_token is not None:
        record["locus"] = facets.locus_token
    return record


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_entries(catalog_path: Path) -> list[dict[str, Any]]:
    """Load every standard name YAML file under ``catalog_path``.

    Accepts the published per-domain list-of-entries layout. Returns
    a flat list ordered by file name then in-file order.
    """
    entries: list[dict[str, Any]] = []
    if not catalog_path.exists():
        return entries
    yaml_files = sorted(
        list(catalog_path.rglob("*.yml")) + list(catalog_path.rglob("*.yaml"))
    )
    for yaml_file in yaml_files:
        try:
            data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            continue
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "name" in item:
                    entries.append(item)
        elif isinstance(data, dict) and "name" in data:
            entries.append(data)
    return entries


def _git_output(repo_root: Path, *args: str) -> str:
    """Return text emitted by a read-only Git command."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ValueError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout


def _resolve_catalog_git_context(catalog_path: Path) -> tuple[Path, Path]:
    """Return the catalog repository root and catalog-relative path."""
    catalog_path = catalog_path.resolve()
    repo_root = Path(
        _git_output(catalog_path, "rev-parse", "--show-toplevel").strip()
    ).resolve()
    try:
        relative_path = catalog_path.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"catalog path {catalog_path} is outside {repo_root}") from exc
    return repo_root, relative_path


def _resolve_commit(repo_root: Path, ref: str) -> str:
    """Resolve ``ref`` to an immutable commit identity."""
    return _git_output(
        repo_root,
        "rev-parse",
        "--verify",
        "--end-of-options",
        f"{ref}^{{commit}}",
    ).strip()


def _git_file_text(repo_root: Path, commit: str, relative_path: Path) -> str | None:
    """Read one file from a commit, returning ``None`` when absent."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{commit}:{relative_path.as_posix()}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout
    if (
        "does not exist" in result.stderr
        or "exists on disk, but not in" in result.stderr
    ):
        return None
    detail = result.stderr.strip() or result.stdout.strip()
    raise ValueError(f"git show {commit}:{relative_path.as_posix()} failed: {detail}")


def _load_entries_at_commit(
    repo_root: Path,
    catalog_relative_path: Path,
    commit: str,
) -> list[dict[str, Any]]:
    """Load catalog entries from ``commit`` without changing the checkout."""
    output = _git_output(
        repo_root,
        "ls-tree",
        "-r",
        "--name-only",
        commit,
        "--",
        catalog_relative_path.as_posix(),
    )
    entries: list[dict[str, Any]] = []
    for file_name in sorted(output.splitlines()):
        if Path(file_name).suffix not in {".yml", ".yaml"}:
            continue
        text = _git_file_text(repo_root, commit, Path(file_name))
        if text is None:
            continue
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError:
            continue
        if isinstance(data, list):
            entries.extend(
                item for item in data if isinstance(item, dict) and "name" in item
            )
        elif isinstance(data, dict) and "name" in data:
            entries.append(data)
    return entries


def _manifest_from_data(data: Any) -> StandardNameCatalogManifest | None:
    """Validate manifest data while preserving the builder's tolerant behavior."""
    if not isinstance(data, dict):
        return None
    try:
        return StandardNameCatalogManifest(**data)
    except Exception:
        return None


def _load_manifest_at_commit(
    repo_root: Path,
    catalog_relative_path: Path,
    commit: str,
) -> StandardNameCatalogManifest | None:
    """Load the catalog manifest associated with ``catalog_relative_path``."""
    candidates = [
        catalog_relative_path.parent / "catalog.yml",
        catalog_relative_path / "catalog.yml",
    ]
    for candidate in candidates:
        text = _git_file_text(repo_root, commit, candidate)
        if text is None:
            continue
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError:
            continue
        if manifest := _manifest_from_data(data):
            return manifest
    return None


def _load_manifest(catalog_path: Path) -> StandardNameCatalogManifest | None:
    """Load ``catalog.yml`` if present alongside (or in) the catalog dir.

    Looks first at ``catalog_path.parent/catalog.yml`` (the standard
    layout — manifest at repo root, entries under ``standard_names/``)
    then at ``catalog_path/catalog.yml`` (entries under repo root).
    Returns ``None`` when the manifest is missing or invalid; the
    dataset builder still works without it.
    """
    candidates = [
        catalog_path.parent / "catalog.yml",
        catalog_path / "catalog.yml",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            data = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            continue
        if manifest := _manifest_from_data(data):
            return manifest
    return None


# ---------------------------------------------------------------------------
# Categories / grammar vocab
# ---------------------------------------------------------------------------


def _build_categories(names: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate NAMES into ``[{id, label, count}]`` sorted by count desc.

    Ties break alphabetically on the slug for determinism.
    """
    counts: Counter[str] = Counter(record.get("category", "") for record in names)
    rows: list[tuple[str, int]] = sorted(
        counts.items(), key=lambda pair: (-pair[1], pair[0])
    )
    return [
        {
            "id": slug,
            "label": _humanise_domain(slug),
            "count": count,
        }
        for slug, count in rows
        if slug
    ]


def _flat_vocab_tokens(filename: str) -> list[str]:
    """Read a flat-list vocabulary YAML and return its tokens in file order.

    The closed-vocabulary files ``components.yml``, ``subjects.yml``,
    ``regions.yml``, and ``processes.yml`` are bare YAML lists of string
    tokens (``- token`` with optional inline comments). File order is
    preserved because it groups tokens meaningfully (the dropdown re-sorts
    by catalog usage at render time, so emission order is cosmetic).
    Returns ``[]`` on any read failure so the dataset never crashes on a
    missing or malformed vocabulary file.
    """
    try:
        data = vocab_loaders._load_yaml(filename)
    except Exception:
        return []
    if isinstance(data, dict):
        # A ``{key: [...]}`` wrapper — take the first list-valued entry.
        for value in data.values():
            if isinstance(value, list):
                return [str(token) for token in value]
        return [str(token) for token in data]
    if isinstance(data, list):
        return [str(token) for token in data]
    return []


def _build_grammar_vocab() -> dict[str, list[dict[str, Any]]]:
    """Build the SPA's ``GRAMMAR_VOCAB`` from the canonical vocabularies.

    The Grammar composer view consumes one entry list per closed-vocabulary
    segment. Every entry is an object carrying at least a ``token``; richer
    segments add the metadata the composer needs:

    * ``operators`` — ``kind`` drives prefix (``op_of_…``) vs postfix
      (``…_op``) rendering; ``returns`` is the result kind.
    * ``locus_registry`` — ``type`` and ``relations`` (the allowed
      ``of``/``at``/``over`` connectors) drive the locus connector switch.
    * ``physical_bases`` / ``geometry_carriers`` — ``kind`` and ``aliases``
      classify the base and pair it with its projection segment.
    * ``physics_domains`` — ``category`` matches the ``category`` field on
      each name record.

    Each section is built independently and tolerates a loader failure by
    falling back to an empty list, so a single malformed vocabulary file
    never sinks the whole dataset build.
    """

    def _safe(fn, default):  # type: ignore[no-untyped-def]
        try:
            return fn()
        except Exception:
            return default

    operators = _safe(vocab_loaders.load_operators, None)
    loci = _safe(vocab_loaders.load_locus_registry, None)
    bases = _safe(vocab_loaders.load_physical_bases, None)
    carriers = _safe(vocab_loaders.load_geometry_carriers, None)
    axes = _safe(vocab_loaders.load_coordinate_axes, None)
    aggregations = _safe(vocab_loaders.load_aggregations, frozenset())
    orbits = _safe(vocab_loaders.load_orbits, frozenset())
    populations = _safe(vocab_loaders.load_populations, frozenset())
    # Zone tokens preserve their canonical intra-order (the loader returns an
    # ORDERED tuple — vertical → radial → region → face); channel tokens keep
    # their locked file order (heat, particle, energy, momentum). Emit in that
    # order so the SPA's segment + picker rows read in canonical sequence.
    zones = _safe(vocab_loaders.load_zones, ())
    channel_qualifiers = _safe(vocab_loaders.load_channel_qualifiers, ())
    channels = _safe(vocab_loaders.load_channels, ())
    # Genuine modifier qualifiers = qualifiers.yml, each carrying its normalized
    # category (qualifier_categories.yml). We deliberately do NOT publish the
    # parser's full acceptance union here: that union also contains operator,
    # locus, and subject tokens (added so they peel during parsing), which are
    # not qualifiers and belong to their own segments. Subjects are excluded
    # too — they have their own segment. So the Grammar composer's qualifier
    # picker offers only true generic qualifiers (external, major, absorbed, …),
    # sub-grouped by category.
    qualifier_tokens = _safe(vocab_loaders.load_qualifiers, frozenset())
    qualifier_cats = _safe(vocab_loaders.load_qualifier_categories, {})
    subject_tokens = set(_flat_vocab_tokens("subjects.yml"))

    physics_domains = (
        _safe(lambda: vocab_loaders._load_yaml("physics_domains.yml"), {}).get(
            "physics_domains", {}
        )
        or {}
    )

    def _tok_list(tokens) -> list[dict[str, Any]]:
        return [{"token": str(token)} for token in sorted(tokens)]

    def _ordered_tok_list(tokens) -> list[dict[str, Any]]:
        # Preserve the loader's emission order (canonical intra-order for zones,
        # locked file order for channels) — do NOT sort.
        return [{"token": str(token)} for token in tokens]

    return {
        "operators": [
            {"token": token, "kind": entry.kind, "returns": entry.returns}
            for token, entry in (
                sorted(operators.operators.items()) if operators else ()
            )
        ],
        "components": [{"token": t} for t in _flat_vocab_tokens("components.yml")],
        "coordinate_axes": [
            {"token": token, "aliases": list(entry.aliases)}
            for token, entry in (sorted(axes.axes.items()) if axes else ())
        ],
        "aggregations": _tok_list(aggregations),
        "orbits": _tok_list(orbits),
        "populations": _tok_list(populations),
        "subjects": [{"token": t} for t in _flat_vocab_tokens("subjects.yml")],
        # Zone follows subject/device in canonical order; channel follows zone
        # (and any residual qualifier) immediately before the base. Both keep
        # their loader order (see _ordered_tok_list).
        "zones": _ordered_tok_list(zones),
        "channel_qualifiers": _ordered_tok_list(channel_qualifiers),
        "channels": _ordered_tok_list(channels),
        "qualifiers": [
            {"token": t, "category": qualifier_cats.get(t, "other")}
            for t in sorted(qualifier_tokens - subject_tokens)
        ],
        "physical_bases": [
            {
                "token": token,
                "kind": entry.kind,
                "aliases": list(entry.aliases),
                "dimensional": entry.inherently_dimensional,
            }
            for token, entry in (sorted(bases.bases.items()) if bases else ())
        ],
        "geometry_carriers": [
            {"token": token, "aliases": list(entry.aliases)}
            for token, entry in (sorted(carriers.carriers.items()) if carriers else ())
        ],
        "locus_registry": [
            {
                "token": token,
                "type": entry.type,
                "relations": list(entry.allowed_relations),
                "definition": entry.definition,
                "abbreviations": list(entry.abbreviations),
            }
            for token, entry in (sorted(loci.loci.items()) if loci else ())
        ],
        "regions": [{"token": t} for t in _flat_vocab_tokens("regions.yml")],
        "processes": [{"token": t} for t in _flat_vocab_tokens("processes.yml")],
        "physics_domains": [
            {
                "token": token,
                "note": (meta or {}).get("description", ""),
                "category": (meta or {}).get("category", ""),
                "ids": list((meta or {}).get("ids", []) or []),
            }
            for token, meta in sorted(physics_domains.items())
        ],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _enrich_with_reverse_links(
    records: list[dict[str, Any]],
    entries: list[dict[str, Any]],
) -> None:
    """Add ``components``, ``magnitude``, and ``children`` reverse-edges.

    Each record gets:

    - ``components`` (list of ``{name, axis}``): for ``algebra='vector'``
      entries only — the axis-projection children that point at this
      vector via ``arguments[*].operator_kind='projection'``. Empty if
      no components are present in this catalog snapshot.
    - ``magnitude`` (str | None): for ``algebra='vector'`` entries only
      — the corresponding ``magnitude_of_<name>`` SN if it exists in
      this catalog. Captures the algebraic vector ⇄ magnitude link
      without requiring graph access (source-driven; only fires when
      the magnitude SN was already composed from DD).
    - ``children`` (list of ``{name, operator_kind}``): all direct
      children regardless of algebra — anything whose first argument
      points at this entry. Used by the SPA's detail panel.

    Mutates *records* in place.
    """
    # Index records by name for fast lookups.
    by_name: dict[str, dict[str, Any]] = {r["name"]: r for r in records}

    # Build a child-by-parent index from the raw YAML entries (we want
    # the operator_kind / axis on each edge, which the normalised
    # ``arguments`` field on records flattens to just names).
    child_index: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        child_name = str(entry.get("name") or "")
        if not child_name:
            continue
        for arg in entry.get("arguments") or []:
            if not isinstance(arg, dict):
                continue
            parent_name = arg.get("name")
            if not isinstance(parent_name, str) or not parent_name:
                continue
            child_index.setdefault(parent_name, []).append(
                {
                    "name": child_name,
                    "operator_kind": arg.get("operator_kind"),
                    "axis": arg.get("axis"),
                }
            )

    for record in records:
        name = record["name"]
        children = sorted(
            child_index.get(name, []),
            key=lambda c: c["name"],
        )
        record["children"] = [
            {
                "name": c["name"],
                "operator_kind": c.get("operator_kind"),
            }
            for c in children
        ]

        if record.get("algebra") == "vector":
            # Components: projection children with axis.
            record["components"] = [
                {"name": c["name"], "axis": c.get("axis")}
                for c in children
                if c.get("operator_kind") == "projection" and c.get("axis")
            ]
            # Magnitude: the magnitude_of_<name> SN if it exists.
            magnitude_id = f"magnitude_of_{name}"
            record["magnitude"] = magnitude_id if magnitude_id in by_name else None
        else:
            # Mirror the vector branch with empty values so the SPA can
            # rely on the keys being present without per-kind branching.
            record["components"] = []
            record["magnitude"] = None


def build_site_dataset(
    catalog_path: Path,
    *,
    review_base_ref: str | None = None,
    review_head_ref: str | None = None,
) -> dict[str, Any]:
    """Build the SPA dataset from a directory of standard-name YAMLs.

    Parameters
    ----------
    catalog_path : Path
        Directory containing one YAML file per physics_domain. The
        catalog manifest is read from ``catalog_path.parent/catalog.yml``
        when present (the published layout); a missing manifest is not
        an error.
    review_base_ref, review_head_ref : str, optional
        Paired Git refs for a catalog review preview. The head-side entries
        added or semantically edited between the refs become the complete
        emitted dataset. Deleted and unchanged entries are not emitted.

    Returns
    -------
    dict
        Keys: ``CATALOG_VERSION`` (str), ``CATEGORIES`` (list),
        ``GRAMMAR_VOCAB`` (dict), ``NAMES`` (list of records).

    Notes
    -----
    All entries whose normalised status is one of the four canonical
    values (``active``, ``draft``, ``deprecated``, ``superseded``) are
    emitted.  Entries with unknown status values are logged and dropped.
    """
    if (review_base_ref is None) != (review_head_ref is None):
        raise ValueError(
            "review_base_ref and review_head_ref must be provided together"
        )

    catalog_path = Path(catalog_path)
    preview_scope: dict[str, Any] | None = None
    preview_names: set[str] | None = None
    if review_base_ref is not None and review_head_ref is not None:
        repo_root, catalog_relative_path = _resolve_catalog_git_context(catalog_path)
        base_commit = _resolve_commit(repo_root, review_base_ref)
        head_commit = _resolve_commit(repo_root, review_head_ref)
        base_entries = _load_entries_at_commit(
            repo_root, catalog_relative_path, base_commit
        )
        raw_entries = _load_entries_at_commit(
            repo_root, catalog_relative_path, head_commit
        )
        base_by_name = {str(entry["name"]): entry for entry in base_entries}
        head_by_name = {str(entry["name"]): entry for entry in raw_entries}
        added = set(head_by_name) - set(base_by_name)
        deleted = set(base_by_name) - set(head_by_name)
        edited = {
            name
            for name in set(base_by_name) & set(head_by_name)
            if base_by_name[name] != head_by_name[name]
        }
        preview_names = added | edited
        preview_scope = {
            "base_ref": base_commit,
            "head_ref": head_commit,
            "added": len(added),
            "edited": len(edited),
            "deleted": len(deleted),
            "visible": len(preview_names),
        }
        manifest = _load_manifest_at_commit(
            repo_root, catalog_relative_path, head_commit
        )
    else:
        raw_entries = _load_entries(catalog_path)
        manifest = _load_manifest(catalog_path)

    # Normalise status — emit every entry with a known canonical status.
    entries: list[dict[str, Any]] = []
    for raw in raw_entries:
        if preview_names is not None and str(raw.get("name")) not in preview_names:
            continue
        entry = dict(raw)
        normalised = _normalise_status(entry.get("status"))
        if normalised is None:
            # Unknown status — already logged; drop silently.
            continue
        entry["status"] = normalised
        entries.append(entry)

    names = [_build_record(entry) for entry in entries]

    # Post-pass: enrich each record with reverse-edge lookups. Vectors
    # get their ``components`` and ``magnitude``; every entry gets
    # ``children`` for the detail panel.
    _enrich_with_reverse_links(names, entries)

    if manifest is not None:
        # Use the ACTUAL number of records emitted, not the manifest's
        # ``published_count`` — manifest counts can lag if the export
        # filter excluded entries after the manifest was written.
        version = (
            f"{manifest.catalog_name} {manifest.grammar_version} · {len(names)} names"
        )
    else:
        version = f"{len(names)} names"

    dataset: dict[str, Any] = {
        "CATALOG_VERSION": version,
        "CATEGORIES": _build_categories(names),
        "GRAMMAR_VOCAB": _build_grammar_vocab(),
        "STANDARD_TERMS": [term.model_dump(mode="json") for term in standard_terms()],
        "NAMES": names,
    }

    if preview_scope is not None:
        visible_names = sorted(record["name"] for record in names)
        dataset["review_batch"] = visible_names
        dataset["preview_scope"] = preview_scope
        if manifest is not None and manifest.review_batch:
            dataset["review_batch_provenance"] = sorted(manifest.review_batch)
    elif manifest is not None and manifest.review_batch:
        dataset["review_batch"] = sorted(manifest.review_batch)

    return dataset


def write_site_dataset(
    catalog_path: Path,
    out_path: Path,
    *,
    review_base_ref: str | None = None,
    review_head_ref: str | None = None,
) -> int:
    """Build and write the SPA dataset to ``out_path`` as JSON.

    Parameters
    ----------
    catalog_path : Path
        Directory of per-domain YAML files.
    out_path : Path
        Destination JSON file.

    Returns the number of NAMES records emitted.
    """
    catalog_path = Path(catalog_path)
    out_path = Path(out_path)
    dataset = build_site_dataset(
        catalog_path,
        review_base_ref=review_base_ref,
        review_head_ref=review_head_ref,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return len(dataset.get("NAMES", []))
