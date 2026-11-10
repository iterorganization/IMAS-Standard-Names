"""Semantic validation checks for provenance/operator logic."""

import re
from functools import lru_cache

from imas_standard_names.grammar.model import parse_standard_name
from imas_standard_names.grammar.model_types import GeometricBase
from imas_standard_names.grammar.parser import ParseError, parse
from imas_standard_names.grammar.vocab_loaders import load_operators
from imas_standard_names.models import StandardNameEntry, StandardNameMetadataEntry
from imas_standard_names.provenance import OperatorProvenance

__all__ = ["run_semantic_checks"]

# Severity applied to unresolvable *inline* documentation references (the
# markdown "[label](name:target)" form). Locked decision: ships as a
# warning now; promote to "ERROR" here when the catalog is ready to fail
# the build on dangling prose references. Structured references (links,
# deprecates, superseded_by, arguments, error_variants) always error —
# export already prunes those, so a dangling structured ref means a broken
# export, not an editorial slip.
INLINE_REFERENCE_SEVERITY = "WARNING"

# Matches the markdown-link form used for inline standard-name references in
# documentation prose, e.g. "[current at target](name:current_at_divertor_target)".
_INLINE_NAME_REF_RE = re.compile(r"\]\(name:([A-Za-z][A-Za-z0-9_]*)\)")

# Geometric bases that describe orientations (require object qualification)
ORIENTATION_BASES = {
    GeometricBase.SURFACE_NORMAL.value,
    GeometricBase.SENSOR_NORMAL.value,
    GeometricBase.TANGENT_VECTOR.value,
}

# Unit-vector carriers are device orientation properties and need the owning
# object too (a locus-less direction unit vector cannot say WHOSE direction it
# is, and lets distinct vectors of one device collapse onto one name). Error
# severity: the catalog's legacy locus-less generics are gone.
UNIT_VECTOR_BASES = {
    GeometricBase.UNIT_VECTOR.value,
    GeometricBase.FIRST_LOCAL_TANGENTIAL_UNIT_VECTOR.value,
    GeometricBase.SECOND_LOCAL_TANGENTIAL_UNIT_VECTOR.value,
    GeometricBase.DIRECTION_UNIT_VECTOR.value,
    GeometricBase.IMAGE_UP_UNIT_VECTOR.value,
    GeometricBase.MAJOR_AXIS_UNIT_VECTOR.value,
    GeometricBase.MINOR_AXIS_UNIT_VECTOR.value,
}

# Geometric bases that describe paths/boundaries (require object qualification)
PATH_BASES = {
    GeometricBase.TRAJECTORY.value,
    GeometricBase.OUTLINE.value,
    GeometricBase.CONTOUR.value,
}

# Geometric base requiring dimension specification
EXTENT_BASE = GeometricBase.EXTENT.value

# Intrinsic plasma coordinate carriers: the coordinate IS the quantity and its
# reference (the plasma equilibrium / machine frame) is universal, so a bare
# name is complete — no object/geometry qualification required.
INTRINSIC_COORDINATE_BASES = {
    GeometricBase.NORMALIZED_POLOIDAL_FLUX_COORDINATE.value,
    GeometricBase.NORMALIZED_TOROIDAL_FLUX_COORDINATE.value,
    GeometricBase.TOROIDAL_FLUX_COORDINATE.value,
    GeometricBase.POLOIDAL_ANGLE.value,
    GeometricBase.TOROIDAL_ANGLE.value,
}


def run_semantic_checks(entries: dict[str, StandardNameEntry]) -> list[str]:
    issues: list[str] = []
    known_names = set(entries)
    for name, entry in entries.items():
        # Operator-provenance checks
        prov = getattr(entry, "provenance", None)
        if isinstance(prov, OperatorProvenance):
            # Example rule: if gradient operator present, unit should include division by length
            if "gradient" in list(prov.operators):
                unit = getattr(entry, "unit", "")
                # Accept units with .m or negative exponent (e.g., m^-4, m^-1)
                if unit and ".m" not in unit and "^-" not in unit and unit != "":
                    issues.append(
                        f"{name}: gradient operator present but unit '{unit}' does not look like derivative (heuristic)."
                    )

        # Grammar-based semantic checks
        issues.extend(_check_geometric_qualifier_requirement(name, entry))
        issues.extend(_check_component_with_base_type(name, entry))
        issues.extend(_check_coordinate_with_base_type(name, entry))
        issues.extend(_check_orientation_vector_completeness(name, entry))
        issues.extend(_check_trajectory_path_qualification(name, entry))
        issues.extend(_check_extent_dimensionality(name, entry))
        issues.extend(_check_physical_base_with_object(name, entry))
        issues.extend(_check_dimensionless_physical_quantity(name, entry))
        issues.extend(_check_none_unit_with_quantitative_kind(name, entry))
        issues.extend(_check_referential_integrity(name, entry, known_names))
        issues.extend(_check_index_dimension_unit(name, entry, entries))

    issues.extend(_check_source_flux_unit_collision(entries))

    return issues


# Tokens whose presence in a name asserts a dimensional derivative of the
# bare quantity: a flux is quantity per area per time, a source is quantity
# created per volume (or area) per time. If the derived name and the bare
# name carry the SAME unit, one of the pair is dimensionally misnamed —
# the defect class where bare "*_momentum" entries were momentum source
# densities sharing units with the differently-defined "*_momentum_flux"
# family.
_DIMENSIONAL_DERIVATIVE_TOKENS = ("flux", "source")


def _check_named_pair_units(
    name: str,
    entry: StandardNameEntry,
    target: str,
    target_entry: StandardNameEntry,
    *,
    must_match: bool,
    relationship: str,
) -> list[str]:
    """Compare stored units for a named pair with a dimensional relationship."""
    unit = getattr(entry, "unit", None)
    target_unit = getattr(target_entry, "unit", None)
    if not unit or not target_unit or (unit == target_unit) == must_match:
        return []
    return [
        f"{name}: WARNING - '{name}' has unit '{unit}' and '{target}' has "
        f"unit '{target_unit}'; {relationship}."
    ]


def _registry_operator_token(token: str) -> str:
    """Resolve the registry key of a coordinate-fused operator token."""
    operator, separator, _ = token.partition("_with_respect_to_")
    return f"{operator}_with_respect_to" if separator else token


def _check_index_dimension_unit(
    name: str, entry: StandardNameEntry, entries: dict[str, StandardNameEntry]
) -> list[str]:
    """Check a result whose outer operator takes its dimension from its index."""
    if "_with_respect_to_" not in name:
        return []
    try:
        ir = parse(name, strict=True).ir
    except ParseError:
        # Structural validation reports names that cannot resolve to an IR.
        return []
    token = ir.operators[0].op
    definition = load_operators().operators[_registry_operator_token(token)]
    if definition.dimension_transforming != "index":
        return []
    coordinate = token.partition("_with_respect_to_")[2]
    target_entry = entries.get(coordinate)
    if target_entry is None:
        return [
            f"{name}: WARNING - operator index references non-existent standard "
            f"name '{coordinate}'"
        ]
    return _check_named_pair_units(
        name,
        entry,
        coordinate,
        target_entry,
        must_match=True,
        relationship="the result must carry its index coordinate's unit",
    )


def _check_source_flux_unit_collision(
    entries: dict[str, StandardNameEntry],
) -> list[str]:
    """Flag ``X`` / ``X_flux`` (or ``X_source``) pairs sharing identical units.

    Rule: for every entry whose name contains a ``flux`` or ``source`` token,
    look up the sibling name with that token removed. If the sibling is also
    published and both carry the same non-empty unit, the pair collides
    dimensionally under near-identical names — a flux/source of a quantity
    cannot share the quantity's own unit, so one of the two is misnamed.
    Severity: Warning
    """
    issues: list[str] = []
    for name, entry in entries.items():
        unit = getattr(entry, "unit", None)
        if not unit:
            continue
        tokens = name.split("_")
        for index, token in enumerate(tokens):
            if token not in _DIMENSIONAL_DERIVATIVE_TOKENS:
                continue
            sibling = "_".join(tokens[:index] + tokens[index + 1 :])
            sibling_entry = entries.get(sibling)
            if sibling_entry is None:
                continue
            issues.extend(
                _check_named_pair_units(
                    name,
                    entry,
                    sibling,
                    sibling_entry,
                    must_match=False,
                    relationship=(
                        f"the names differ only by '{token}' but share unit '{unit}'. "
                        f"A {token} of a quantity has different dimensions from the "
                        "quantity itself, so one of the pair is likely misnamed"
                    ),
                )
            )
    return issues


def _check_referential_integrity(
    name: str, entry: StandardNameEntry, known_names: set[str]
) -> list[str]:
    """Resolve every structured and inline name reference an entry carries.

    ``links`` (internal ``name:`` entries), ``deprecates``, and
    ``superseded_by`` are governance/documentation edges — a dangling
    reference here means export produced or preserved a broken edge, so
    these are reported at error severity.

    ``arguments`` (operator decomposition edges) and ``error_variants``
    (upper/lower/index siblings) are computed fields re-derived on export
    (see :class:`StandardNameEntryBase`); ``yaml_store.YamlStore.load``
    already carries a dedicated warning for these two (they may legitimately
    reference a sibling not yet composed), so this check reports them at
    warning severity too rather than duplicating that mechanism at a
    stricter level.

    Inline references embedded in documentation prose as markdown links
    (``[label](name:target)``) are authored free text and drift more
    easily; they are reported at :data:`INLINE_REFERENCE_SEVERITY`
    (warning by default — see the module docstring for promotion).

    A resolvable ``superseded_by``/``deprecates`` target is valid regardless
    of the target's own status — deprecated stub entries pointing forward to
    an active successor are exactly what this check must accept, not flag.
    Severity: Error (links / deprecates / superseded_by) / Warning
    (arguments, error_variants, inline — the last promotable to error).
    """
    issues: list[str] = []

    for link in getattr(entry, "links", None) or []:
        if isinstance(link, str) and link.startswith("name:"):
            target = link[len("name:") :].strip()
            if target and target not in known_names:
                issues.append(
                    f"{name}: ERROR - links references non-existent standard "
                    f"name '{target}'"
                )

    for field in ("deprecates", "superseded_by"):
        target = getattr(entry, field, None)
        if target and target not in known_names:
            issues.append(
                f"{name}: ERROR - {field} references non-existent standard "
                f"name '{target}'"
            )

    for arg in getattr(entry, "arguments", None) or []:
        target = getattr(arg, "name", None)
        if target and target not in known_names:
            issues.append(
                f"{name}: WARNING - arguments references non-existent standard "
                f"name '{target}'"
            )

    error_variants = getattr(entry, "error_variants", None) or {}
    for role, target in error_variants.items():
        if target and target not in known_names:
            issues.append(
                f"{name}: WARNING - error_variants[{role}] references "
                f"non-existent standard name '{target}'"
            )

    documentation = getattr(entry, "documentation", None)
    if documentation:
        for match in _INLINE_NAME_REF_RE.finditer(documentation):
            target = match.group(1)
            if target not in known_names:
                issues.append(
                    f"{name}: {INLINE_REFERENCE_SEVERITY} - documentation "
                    f"references non-existent standard name '{target}'"
                )

    return issues


def _check_geometric_qualifier_requirement(
    name: str, entry: StandardNameEntry
) -> list[str]:
    """Geometric bases MUST have object, geometry, OR position qualification.

    Rule: Names with geometric_base require an object, geometry, or position
    segment. Intrinsic plasma coordinates are exempt (their reference is
    universal); orientation/unit-vector/path bases are owned by their
    dedicated checks below.
    Rationale: Geometric quantities describe spatial properties *of something*.
    Severity: Error
    """
    issues: list[str] = []
    try:
        parsed = parse_standard_name(name)
        geometric_base = getattr(parsed, "geometric_base", None)

        if geometric_base and (
            geometric_base not in INTRINSIC_COORDINATE_BASES
            and geometric_base not in ORIENTATION_BASES
            and geometric_base not in UNIT_VECTOR_BASES
            and geometric_base not in PATH_BASES
        ):
            obj = getattr(parsed, "object", None)
            geom = getattr(parsed, "geometry", None)
            pos = getattr(parsed, "position", None)

            if not obj and not geom and not pos:
                issues.append(
                    f"{name}: ERROR - geometric quantity '{geometric_base}' requires "
                    "object, geometry, or position qualifier to specify what is "
                    "being described. "
                    f"Example: {geometric_base}_of_flux_loop or {geometric_base}_at_midplane"
                )
    except Exception:
        # If parsing fails, skip this check (structural validation will catch it)
        pass

    return issues


def _check_component_with_base_type(name: str, entry: StandardNameEntry) -> list[str]:
    """Component segment should primarily be used with physical bases, not geometric.

    Rule: component (radial/toroidal/etc.) describes directional projections of
    physical vector fields, not geometric positions (use coordinate for those).
    Severity: Warning
    """
    issues: list[str] = []
    try:
        parsed = parse_standard_name(name)
        component = getattr(parsed, "component", None)
        geometric_base = getattr(parsed, "geometric_base", None)

        if component and geometric_base:
            issues.append(
                f"{name}: WARNING - 'component' is typically for physical vectors "
                f"(magnetic_field, velocity), not geometric quantities like '{geometric_base}'. "
                f"Consider using 'coordinate' instead: {component}_coordinate_{geometric_base}_of_..."
            )
    except Exception:
        pass

    return issues


def _check_coordinate_with_base_type(name: str, entry: StandardNameEntry) -> list[str]:
    """Coordinate segment should be used with geometric bases, not physical bases.

    Rule: coordinate defines spatial reference frames for geometric quantities.
    Severity: Warning
    """
    issues: list[str] = []
    try:
        parsed = parse_standard_name(name)
        coordinate = getattr(parsed, "coordinate", None)
        physical_base = getattr(parsed, "physical_base", None)

        if coordinate and physical_base:
            issues.append(
                f"{name}: WARNING - 'coordinate' is for geometric/spatial quantities, "
                f"not physical fields like '{physical_base}'. Consider using 'component' "
                f"for directional decomposition: {coordinate}_{physical_base}"
            )
    except Exception:
        pass

    return issues


def _check_orientation_vector_completeness(
    name: str, entry: StandardNameEntry
) -> list[str]:
    """Orientation vectors (normals, tangents) must have object qualification.

    Rule: Normals and tangents are properties of surfaces/sensors/curves.
    Severity: Error
    """
    issues: list[str] = []
    try:
        parsed = parse_standard_name(name)
        geometric_base = getattr(parsed, "geometric_base", None)

        if geometric_base in ORIENTATION_BASES:
            obj = getattr(parsed, "object", None)
            if not obj:
                base_type = "normal" if "normal" in geometric_base else "tangent"
                issues.append(
                    f"{name}: ERROR - {base_type} vector '{geometric_base}' must specify "
                    f"the object/surface it belongs to. Example: {geometric_base}_of_divertor_tile"
                )
        elif geometric_base in UNIT_VECTOR_BASES:
            obj = getattr(parsed, "object", None)
            if not obj:
                issues.append(
                    f"{name}: ERROR - unit vector '{geometric_base}' must specify "
                    f"the device/object it orients. Example: {geometric_base}_of_camera"
                )
    except Exception:
        pass

    return issues


def _check_trajectory_path_qualification(
    name: str, entry: StandardNameEntry
) -> list[str]:
    """Trajectories and paths must specify what is moving/bounded.

    Rule: Trajectories describe motion; outlines/contours describe boundaries.
    Severity: Error
    """
    issues: list[str] = []
    try:
        parsed = parse_standard_name(name)
        geometric_base = getattr(parsed, "geometric_base", None)

        if geometric_base in PATH_BASES:
            entity = getattr(parsed, "object", None) or getattr(
                parsed, "geometry", None
            )
            if not entity:
                example_obj = (
                    "neutral_beam" if geometric_base == "trajectory" else "limiter_tile"
                )
                issues.append(
                    f"{name}: ERROR - '{geometric_base}' must specify what entity's path/boundary "
                    f"is described. Example: {geometric_base}_of_{example_obj}"
                )
    except Exception:
        pass

    return issues


def _check_extent_dimensionality(name: str, entry: StandardNameEntry) -> list[str]:
    """Extent should typically specify dimension via component.

    Rule: 'extent' (size/span) often needs directional qualification.
    Severity: Info
    """
    issues: list[str] = []
    try:
        parsed = parse_standard_name(name)
        geometric_base = getattr(parsed, "geometric_base", None)
        component = getattr(parsed, "component", None)

        if geometric_base == EXTENT_BASE and not component:
            obj = getattr(parsed, "object", None)
            obj_str = f"_of_{obj}" if obj else ""
            issues.append(
                f"{name}: INFO - 'extent' typically specifies dimension. "
                f"Consider: radial_extent{obj_str} or vertical_extent{obj_str}"
            )
    except Exception:
        pass

    return issues


@lru_cache(maxsize=1)
def _entity_typed_loci() -> frozenset[str]:
    """Locus tokens typed ``entity`` in the registry (hardware/geometry carriers).

    For an entity locus the ``_of_<entity>`` relation names an intrinsic
    property of the entity (the locus matrix maps ``entity -> of``), so
    ``<base>_of_<entity>`` is the canonical authoring form, not a
    measurement-location smell.
    """
    from ..grammar.vocab_loaders import load_locus_registry  # noqa: PLC0415

    registry = load_locus_registry()
    return frozenset(
        token for token, entry in registry.loci.items() if entry.type == "entity"
    )


def _check_physical_base_with_object(name: str, entry: StandardNameEntry) -> list[str]:
    """Physical bases with object qualification may indicate confusion.

    Rule: Physical quantities (temperature, density) are properties of subjects,
    not objects. Object qualification suggests measuring location rather than
    intrinsic property.

    Suppressed for entity-typed loci: ``<base>_of_<entity>`` is the canonical
    intrinsic-property form for a hardware/geometry carrier (the locus matrix
    maps ``entity -> of``), so flagging it contradicts the authoring
    convention and is pure noise.
    Severity: Info
    """
    issues: list[str] = []
    try:
        parsed = parse_standard_name(name)
        physical_base = getattr(parsed, "physical_base", None)
        obj = getattr(parsed, "object", None)

        if physical_base and obj:
            # Check if subject is also present (which would be more appropriate)
            subject = getattr(parsed, "subject", None)
            obj_token = getattr(obj, "value", obj)
            if not subject and obj_token not in _entity_typed_loci():
                issues.append(
                    f"{name}: INFO - physical quantity '{physical_base}' with object '{obj}' "
                    "may indicate measurement location. Consider using subject (electron, ion) "
                    f"for intrinsic properties, or _at_{obj} for field evaluation location."
                )
    except Exception:
        pass

    return issues


# ---------------------------------------------------------------------------
# Dimensionless physical quantity check — vocabulary-driven
# ---------------------------------------------------------------------------


def _load_inherently_dimensional_bases() -> frozenset[str]:
    """Load the set of physical bases marked ``inherently_dimensional`` in vocab."""
    from ..grammar.vocab_loaders import load_physical_bases  # noqa: PLC0415

    registry = load_physical_bases()
    return frozenset(
        token for token, defn in registry.bases.items() if defn.inherently_dimensional
    )


def _load_dimensionless_operators() -> frozenset[str]:
    """Load operators marked ``dimensionless: true`` in the operator vocabulary."""
    from ..grammar.vocab_loaders import load_operators  # noqa: PLC0415

    registry = load_operators()
    return frozenset(
        token for token, defn in registry.operators.items() if defn.dimensionless
    )


def _load_dimension_transforming_operators() -> frozenset[str]:
    """Load operators declaring a dimension transformation in the vocabulary.

    These change the dimensions of their argument (integrals, derivatives,
    inverse, square), so the base-implies-unit heuristic does not apply —
    ``volume_integrated_<density>`` is a dimensionless count.
    """
    registry = load_operators()
    return frozenset(
        token
        for token, defn in registry.operators.items()
        if defn.dimension_transforming is not None
    )


def _load_normalizing_qualifiers() -> frozenset[str]:
    """Load qualifier tokens that imply dimensionless output from vocab."""
    from ..grammar.vocab_loaders import load_normalizing_qualifiers  # noqa: PLC0415

    return load_normalizing_qualifiers()


def _check_dimensionless_physical_quantity(
    name: str, entry: StandardNameEntry
) -> list[str]:
    """Flag dimensionless unit on physical bases that inherently carry units.

    Rule: Scalar/vector entries with ``unit="1"`` whose physical base is a
    quantity that normally carries SI units are likely misconfigured. The
    following constructs are excluded because they can legitimately be
    dimensionless:

    - binary operators (ratio, product, difference);
    - operators marked ``dimensionless: true`` in the operator vocabulary
      (e.g. normalized, perturbed, logarithm);
    - any qualifier listed in ``normalizing_qualifiers.yml``
      (e.g. ``gyrocenter_pressure`` is gyrokinetic-normalized by convention).

    Severity: Warning
    """
    issues: list[str] = []

    # Only applies to scalar/vector entries that have a unit field
    if isinstance(entry, StandardNameMetadataEntry):
        return issues

    unit = getattr(entry, "unit", None)
    if unit != "1":
        return issues

    try:
        parsed = parse_standard_name(entry.name)
        binary_operator = getattr(parsed, "binary_operator", None)
        transformation = getattr(parsed, "transformation", None)

        # Binary operators (ratio, product, difference) can legitimately yield
        # dimensionless results, so skip the check for those.
        if binary_operator:
            return issues

        # Operators marked dimensionless in the vocabulary always produce
        # dimensionless output (normalized, perturbed, logarithm, etc.);
        # dimension-transforming operators (integrals, derivatives, inverse)
        # change the unit of the base, so the base-implies-unit inference is
        # invalid for them (volume_integrated density is a count).
        dimensionless_ops = _load_dimensionless_operators()
        transforming_ops = _load_dimension_transforming_operators()
        exempt_ops = dimensionless_ops | transforming_ops
        tx_value = _registry_operator_token(
            getattr(transformation, "value", transformation) or ""
        )
        if tx_value in exempt_ops:
            return issues

        # Qualifiers that imply normalization (from normalizing_qualifiers.yml)
        # or that correspond to dimensionless operator tokens produce
        # dimensionless output even when parsed as qualifiers.
        normalizing_quals = _load_normalizing_qualifiers()
        # Operator tokens can surface in the IR qualifier list (the parser's
        # acceptance union), so exempt the transforming set there too.
        exempt_qualifiers = dimensionless_ops | normalizing_quals | transforming_ops
        try:
            from ..grammar.parser import parse as ir_parse  # noqa: PLC0415

            ir = ir_parse(entry.name).ir
            if ir is not None:
                physical_base = ir.base.token
                qualifier_tokens = {q.token for q in (ir.qualifiers or [])}
                if qualifier_tokens & exempt_qualifiers:
                    return issues
                operator_tokens = {
                    _registry_operator_token(op.op) for op in (ir.operators or [])
                }
                if operator_tokens & exempt_ops:
                    return issues
        except Exception:
            physical_base = getattr(parsed, "physical_base", None)

        inherently_dimensional = _load_inherently_dimensional_bases()
        if physical_base and physical_base in inherently_dimensional:
            issues.append(
                f"{name}: WARNING - dimensionless unit '1' on physical quantity "
                f"'{physical_base}' is unexpected. Quantities like '{physical_base}' "
                "normally carry SI units. Use '1' only for true dimensionless "
                "quantities (ratios, coefficients, counts)."
            )
    except Exception:
        pass

    return issues


def _check_none_unit_with_quantitative_kind(
    name: str, entry: StandardNameEntry
) -> list[str]:
    """Flag None unit on scalar/vector entries that should have units.

    Rule: Scalar and vector entries represent physical quantities that should
    have explicit units. ``unit=None`` is reserved for metadata entries
    (identifiers, labels, enumerations). Scalar/vector entries with no unit
    likely need either a specific SI unit or ``'1'`` for dimensionless quantities.
    Severity: Warning
    """
    issues: list[str] = []

    # Only metadata entries are exempt from unit requirement
    if isinstance(entry, StandardNameMetadataEntry):
        return issues

    unit = getattr(entry, "unit", None)
    if unit is not None:
        return issues

    kind = getattr(entry, "kind", None)
    issues.append(
        f"{name}: WARNING - {kind} entry has no unit (unit=None). "
        "Scalar and vector entries should have explicit units. "
        "Use a specific SI unit, or '1' for dimensionless quantities "
        "(ratios, coefficients, counts). "
        "Reserve unit=None for metadata entries only."
    )

    return issues
