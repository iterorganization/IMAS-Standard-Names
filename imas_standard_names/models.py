"""Pydantic models for IMAS Standard Names catalog entries.

This module defines the complete data model for standard name catalog entries,
including all metadata, provenance, governance, and validation rules.

The StandardNameEntry union type represents full catalog entries with:
- Core identification (name, kind, description)
- Physical properties (unit)
- Governance (status, deprecation, supersession)
- Provenance (operators, reductions, expressions)
- Metadata (links, documentation)

Example scalar entry:
  name: ion_temperature
  kind: scalar
  physics_domain: core_plasma_physics
  status: active
  unit: eV
  description: Core ion temperature.

Example vector entry:
  name: plasma_velocity
  kind: vector
  physics_domain: core_plasma_physics
  status: active
  unit: m/s
  description: Plasma velocity vector.

Example metadata entry:
  name: plasma_boundary
  kind: metadata
  physics_domain: equilibrium
  status: draft
  description: Definition of plasma boundary.
  documentation: |
    Defines what constitutes the plasma boundary for different configurations.

Example with operator provenance:
  name: gradient_of_electron_temperature
  kind: vector
  status: active
  unit: eV/m
  description: Spatial gradient of electron temperature.
  provenance:
    mode: operator
    operators: [gradient]
    base: electron_temperature
    operator_id: gradient

"""

import re
from collections.abc import Iterable
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, get_args

import yaml as _yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from imas_standard_names import canonical_unit, pint
from imas_standard_names.field_types import (
    STANDARD_NAME_PATTERN,
    Description,
    Documentation,
    Links,
    Name,
    Unit,
)
from imas_standard_names.grammar import vocab_loaders as _vocab_loaders
from imas_standard_names.grammar.field_schemas import FIELD_DESCRIPTIONS

# Component, Position, Process remain importable from model_types for
# downstream compatibility but are NO LONGER used in the validators below.
from imas_standard_names.grammar.model_types import (  # noqa: F401
    Component,
    Position,
    Process,
)
from imas_standard_names.grammar.tag_types import PhysicsDomain
from imas_standard_names.operators import (
    enforce_operator_naming as _enforce_operator_naming,
    normalize_operator_chain as _normalize_operator_chain,
)
from imas_standard_names.provenance import (
    ExpressionProvenance,
    OperatorProvenance,
    Provenance,
)

Status = Literal["draft", "active", "deprecated", "superseded"]

# Version of the catalog manifest's edge/metadata contract. It advances
# whenever a consumer must read the manifest differently: this value carries
# the per-name metadata sidecar, which the previous one did not have.
CATALOG_EDGE_MODEL_VERSION = "v2"

# ---------------------------------------------------------------------------
# vocabulary caches (lazy-loaded once per process)
# ---------------------------------------------------------------------------

_COMPONENT_VOCAB_CACHE: frozenset[str] | None = None
_COORDINATE_AXES_CACHE: frozenset[str] | None = None
_LOCUS_REGISTRY_CACHE: dict | None = None

_VOCAB_DIR = Path(__file__).parent / "grammar" / "vocabularies"


def _get_component_vocab() -> frozenset[str]:
    """Return the component token set (from components.yml)."""
    global _COMPONENT_VOCAB_CACHE
    if _COMPONENT_VOCAB_CACHE is None:
        with (_VOCAB_DIR / "components.yml").open(encoding="utf-8") as _fh:
            _data = _yaml.safe_load(_fh) or []
        _COMPONENT_VOCAB_CACHE = frozenset(
            item for item in _data if isinstance(item, str)
        )
    return _COMPONENT_VOCAB_CACHE


def _get_coordinate_axes() -> frozenset[str]:
    """Return the coordinate axis token set (from coordinate_axes.yml)."""
    global _COORDINATE_AXES_CACHE
    if _COORDINATE_AXES_CACHE is None:
        _reg = _vocab_loaders.load_coordinate_axes()
        _COORDINATE_AXES_CACHE = frozenset(_reg.axes)
    return _COORDINATE_AXES_CACHE


def _get_locus_registry() -> dict:
    """Return the locus registry dict token -> LocusEntry."""
    global _LOCUS_REGISTRY_CACHE
    if _LOCUS_REGISTRY_CACHE is None:
        _reg = _vocab_loaders.load_locus_registry()
        _LOCUS_REGISTRY_CACHE = dict(_reg.loci)
    return _LOCUS_REGISTRY_CACHE


def _check_grammar_vocabulary_consistency(name: str) -> list[str]:
    """Check if a standard name uses vocabulary tokens that don't exist in grammar.

    Only flags cases where clear template patterns indicate missing vocabulary tokens.
    Does NOT flag compound base names like 'electron_temperature' or 'plasma_velocity'.

    Validators check against the vocabulary loaders
    (``grammar/vocab_loaders.py``) rather than the legacy Component/Position/
    Process enums in ``grammar/model_types``.
    """
    errors = []

    # ------------------------------------------------------------------
    # 1. 'component_of' pattern -> check against components.yml tokens
    #
    # Example valid:   "radial_component_of_magnetic_field"
    # Example invalid: "nonexistent_component_of_magnetic_field"
    # Skip check when the token before '_component_of_' contains '_of_' --
    # this indicates operator nesting (e.g.
    # "normalized_of_parallel_component_of_...") that the parser resolves
    # via operator peeling; the leading segment captured by the regex is
    # compound, not a bare component token.
    # ------------------------------------------------------------------
    component_match = re.search(r"^([a-z_]+)_component_of_", name)
    if component_match:
        token = component_match.group(1)
        if "_of_" not in token and token not in _get_component_vocab():
            errors.append(
                f"Token '{token}' used with 'component_of' template is missing"
                " from component vocabulary"
            )

    # ------------------------------------------------------------------
    # 2. Coordinate prefix pattern -> check against coordinate_axes.yml
    #
    # Example valid:   "radial_outline_of_plasma_boundary"
    # Skip check when the captured prefix contains '_of_' -- a multi-word
    # prefix means the regex over-matched a longer compound token (e.g.
    # "vertical_coordinate_of_plasma_boundary_outline_point" captures
    # "vertical_coordinate_of_plasma_boundary" before "_outline_").
    # ------------------------------------------------------------------
    coordinate_match = re.search(
        r"^([a-z_]+)_(?:position|vertex|centroid|outline|contour|displacement"
        r"|offset|trajectory|extent|surface_normal|sensor_normal|tangent_vector)_",
        name,
    )
    if coordinate_match:
        token = coordinate_match.group(1)
        if "_of_" not in token and token not in _get_coordinate_axes():
            errors.append(
                f"Token '{token}' used as coordinate prefix is missing from"
                " coordinate_axes vocabulary"
            )

    # ------------------------------------------------------------------
    # 3. 'at_' pattern -> check against locus_registry.yml
    #
    # Only raise an error when the token IS found in the locus registry but
    # that locus type does NOT permit the 'at' relation.  Tokens absent from
    # the registry are silently accepted -- the parser emits a 'vocab_gap'
    # info diagnostic for those, and this validator must not second-guess it.
    #
    # Example valid (in registry, allows at):
    #   "pressure_at_plasma_boundary"  (plasma_boundary: type=position,
    #                                    allowed=[at,of])
    # Example accepted via VocabGap (not in registry):
    #   "normalized_pressure_gradient_at_gyrokinetic_flux_surface"
    # ------------------------------------------------------------------
    at_match = re.search(r"_at_([a-z_]+)(?:_|$)", name)
    if at_match:
        token = at_match.group(1)
        locus_reg = _get_locus_registry()
        if token in locus_reg:
            entry = locus_reg[token]
            if "at" not in entry.allowed_relations:
                allowed_str = sorted(entry.allowed_relations)
                errors.append(
                    f"Token '{token}' used with 'at_' template is not permitted"
                    f" for locus type '{entry.type}' (allowed: {allowed_str})"
                )
        # else: token not in locus_registry -> VocabGap (parser info diagnostic),
        # no ValidationError raised here.

    # ------------------------------------------------------------------
    # 4. 'due_to_' pattern check INTENTIONALLY OMITTED.
    #
    # The parser's _strip_mechanism stage (grammar/parser.py) accepts
    # any token after '_due_to_' without vocabulary enforcement.  Unknown
    # process tokens produce no error in the parser, so this validator must
    # not raise a false-positive either.  Process vocabulary coverage is
    # tracked via the VocabGap mechanism in the parser, not here.
    # ------------------------------------------------------------------

    return errors


class Kind(StrEnum):
    """Runtime enum for standard name kinds.

    Physical rank hierarchy:

    * ``scalar``  – rank-0, real-valued point quantity
      (e.g. ``electron_temperature``, ``safety_factor``).
    * ``vector``  – rank-1 quantity, including named single components.
      A field like ``magnetic_field_r`` carries an implicit radial
      covariant index and is therefore *vector*-natured even though
      only one component is stored.
    * ``tensor``  – rank-2 or higher quantity (stress tensor, metric
      tensor, conductivity tensor — full or individual component).
    * ``complex`` – complex-valued quantity (real + imaginary parts,
      or magnitude + phase).  Orthogonal to tensor rank in principle;
      kept as its own kind to preserve a flat discriminator.  Revisit
      if rank-aware complex names accumulate.
    * ``metadata`` – non-physical bookkeeping / annotation entries
      (e.g. ``plasma_boundary``, ``scrape_off_layer``).  No unit or
      provenance required.
    """

    scalar = "scalar"
    vector = "vector"
    tensor = "tensor"
    complex = "complex"
    metadata = "metadata"


class StandardNameBase(BaseModel):
    """Core identity + governance fields valid without documentation.

    This is the shared base for both full catalog entries (with description,
    documentation, etc.) and lightweight name-only entries used during
    pipeline stages that generate names before descriptions exist. It defines
    only the fields and validators that are meaningful in both modes:

    - name, kind (set by subclass), status
    - deprecation/supersession governance
    - COCOS transformation type

    Subclasses either add full-entry documentation fields (see
    :class:`StandardNameEntryBase`) or remain minimal (name-only variants).
    """

    model_config = ConfigDict(extra="forbid")

    # Core identification
    name: Name
    status: Status = Field(
        "draft",
        description=FIELD_DESCRIPTIONS["status"],
    )

    # Governance
    deprecates: Name | None = None
    superseded_by: Name | None = None
    cocos_transformation_type: str | None = None

    # Supplemental validator for double underscore rule not expressible in pattern.
    @field_validator("name", "deprecates", "superseded_by")
    @classmethod
    def _no_double_underscore(cls, v: str | None) -> str | None:
        if v and "__" in v:
            raise ValueError("Name tokens must not contain double underscores")
        return v

    @field_validator("name")
    @classmethod
    def _check_grammar_vocabulary_consistency(cls, v: str) -> str:
        """Validate that template tokens in the name exist in the grammar vocabulary."""
        if not v:
            return v

        errors = _check_grammar_vocabulary_consistency(v)
        if errors:
            error_msg = "Grammar vocabulary consistency errors:\n" + "\n".join(
                f"  - {error}" for error in errors
            )
            raise ValueError(error_msg)
        return v

    @model_validator(mode="after")
    def _governance_rules(self):  # type: ignore[override]
        if self.status == "deprecated" and not self.superseded_by:
            raise ValueError(
                "Deprecated entries must set superseded_by referencing an active name"
            )
        return self

    # Base unit helpers (shared by scalar/vector subclasses, full and name-only).
    @staticmethod
    def _canonicalize_unit_order(v: str | None) -> str:
        """Auto-correct unit token order to canonical dot-exponent form.

        This helps LLMs and human authors by accepting units in any order
        and automatically reordering to canonical form. For example:
        's^-2.m' -> 'm.s^-2'
        'keV.m^-1' -> 'keV.m^-1' (already canonical)

        Authoring conventions (no whitespace, no '/' or '*', '1' for
        dimensionless) are enforced here; token ordering and symbol spelling
        are delegated to :func:`canonical_unit`, the single pint-based
        authority, so there is no second ordering implementation to drift.

        Empty strings and units pint cannot parse fail validation.
        """
        if v is None:
            raise ValueError(
                "Unit is required for quantitative entries; use '1' for "
                "dimensionless quantities"
            )
        if v == "1":
            return "1"
        if v == "":
            raise ValueError(
                "Empty string not allowed for unit; use '1' for dimensionless quantities"
            )
        if " " in v:
            raise ValueError("Unit must not contain whitespace")
        if "/" in v or "*" in v:
            raise ValueError(
                "Use dot-exponent style (e.g. m.s^-2); '/' and '*' are forbidden"
            )
        if not pint:
            return v
        return canonical_unit(v)

    @staticmethod
    def _validate_unit_with_pint(v: str) -> str:
        """Validate unit semantics using pint (if available)."""
        if v == "1":
            return v
        if pint:
            try:
                pint.Unit(v)
            except Exception as e:  # pragma: no cover - defensive
                raise ValueError(f"Invalid unit '{v}': {e}") from e
        return v


class ArgumentRef(BaseModel):
    """Reference to another standard name as an argument of a structural operator.

    Captures one layer of the ISN grammar's operator decomposition: the outer
    operator that wraps this entry around a *base* argument.  Entries emitted
    by the codex export pipeline carry ``arguments`` lists whose elements are
    ``ArgumentRef`` instances.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    operator: str
    operator_kind: Literal[
        "unary_prefix",
        "unary_postfix",
        "binary",
        "projection",
        "qualifier",
        "locus",
    ]
    role: Literal["a", "b"] | None = None
    separator: Literal["and", "to"] | None = None
    axis: str | None = None
    shape: Literal["component", "coordinate"] | None = None

    @model_validator(mode="after")
    def _check_operator_kind_fields(self):
        """Enforce field requirements based on operator_kind.

        The structural derivation in the imas-codex pipeline peels one
        layer of the ISN grammar per ``COMPONENT_OF`` edge and stamps
        the layer kind here. Supported kinds:

        * ``unary_prefix`` / ``unary_postfix`` — operator layer
          (e.g. ``maximum_of_x``, ``x_magnitude``). Only ``operator``
          carries data; role/separator/axis/shape MUST be absent.
        * ``binary`` — binary operator layer (e.g. ``ratio_of_x_to_y``).
          Requires ``role`` and ``separator``; ``axis`` and ``shape``
          MUST be absent.
        * ``projection`` — axis projection layer
          (e.g. ``toroidal_magnetic_field``). Requires ``axis`` and
          ``shape``; ``role`` and ``separator`` MUST be absent.
        * ``qualifier`` — outermost qualifier layer
          (e.g. ``upper_elongation_of_plasma_boundary`` peels ``upper``).
          Only ``operator`` carries the qualifier token; all other
          fields MUST be absent.
        * ``locus`` — locus suffix layer
          (e.g. ``elongation_of_plasma_boundary`` peels
          ``plasma_boundary``). Only ``operator`` carries the locus
          token; the relation (``of`` vs ``at``) is recoverable from
          the source name's parse so is not carried on the edge.
        """
        if self.operator_kind == "binary":
            if self.role is None or self.separator is None:
                raise ValueError(
                    "role and separator are required when operator_kind is 'binary'"
                )
            if self.axis is not None or self.shape is not None:
                raise ValueError(
                    "axis and shape are forbidden when operator_kind is 'binary'"
                )
        elif self.operator_kind == "projection":
            if self.axis is None or self.shape is None:
                raise ValueError(
                    "axis and shape are required when operator_kind is 'projection'"
                )
            if self.role is not None or self.separator is not None:
                raise ValueError(
                    "role and separator are forbidden when operator_kind is 'projection'"
                )
        else:
            # unary_prefix / unary_postfix / qualifier / locus — bare
            # operator, no other fields.
            if self.role is not None or self.separator is not None:
                raise ValueError(
                    f"role and separator are forbidden when operator_kind is '{self.operator_kind}'"
                )
            if self.axis is not None or self.shape is not None:
                raise ValueError(
                    f"axis and shape are forbidden when operator_kind is '{self.operator_kind}'"
                )
        return self


class CatalogSourceBinding(BaseModel):
    """Immutable reference to a source-system record used by a catalog entry."""

    model_config = ConfigDict(extra="ignore")

    kind: str = Field(min_length=1)
    ref: str = Field(min_length=1)
    version: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_binding(cls, value: Any) -> Any:
        """Translate the retired DD and facility identifier keys on load."""
        if not isinstance(value, dict):
            return value

        migrated = dict(value)
        dd_path = migrated.get("dd_path")
        signal_id = migrated.get("signal_id")
        if dd_path not in (None, ""):
            migrated.setdefault("kind", "imas-dd")
            migrated.setdefault("ref", dd_path)
            migrated.setdefault("version", migrated.get("dd_version"))
        elif signal_id not in (None, ""):
            migrated.setdefault("ref", signal_id)
        return migrated


class StandardNameEntryBase(StandardNameBase):
    """Full catalog entry definition (fields common to scalar and vector kinds).

    Extends :class:`StandardNameBase` with the documentation and metadata fields
    required of a published standard name: physics domain, description,
    documentation, and links. This remains the class used for the full
    catalog (serialization, JSON schema, rendering).
    """

    model_config = ConfigDict(extra="forbid")

    physics_domain: PhysicsDomain = Field(
        description=(
            "Required physics-domain classification from the PhysicsDomain enum."
        )
    )

    @field_validator("physics_domain", mode="before")
    @classmethod
    def validate_physics_domain(cls, value: Any) -> PhysicsDomain:
        """Require a value declared by the PhysicsDomain enum."""
        try:
            return PhysicsDomain(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "physics_domain must be a valid PhysicsDomain enum value"
            ) from exc

    # Documentation & description
    description: Description
    documentation: Documentation  # Required: valuable standalone content

    # Governance / metadata (documentation-adjacent)
    links: Links = Field(default_factory=list)

    # Structural graph edges (computed fields, re-derived on export)
    arguments: list[ArgumentRef] | None = None
    error_variants: dict[Literal["upper", "lower", "index"], str] | None = None

    # Source-system bindings included by provenance-aware catalog exports.
    sources: list["CatalogSourceBinding"] | None = None

    @field_validator("links")
    @classmethod
    def validate_links(cls, v: Iterable[str]) -> list[str]:  # type: ignore[override]
        """Validate links: normalize whitespace and validate format.

        Supports two formats:
        1. External URLs: must start with http:// or https://
        2. Internal standard name references: must start with 'name:' followed by valid name token
        """
        if v is None:
            return []

        result = []
        for item in v:
            link = str(item).strip()
            if not link:
                continue

            # Check if it's an internal name reference
            if link.startswith("name:"):
                # Extract the name part and validate it
                name_part = link[5:].strip()  # Remove 'name:' prefix
                if not name_part:
                    raise ValueError(
                        f"Invalid internal link '{link}': name cannot be empty after 'name:' prefix"
                    )
                # Validate name format
                if not re.match(STANDARD_NAME_PATTERN, name_part):
                    raise ValueError(
                        f"Invalid internal link '{link}': '{name_part}' is not a valid standard name token. "
                        f"Must match pattern {STANDARD_NAME_PATTERN}"
                    )
                result.append(link)
            # Check if it's an external URL
            elif link.startswith(("http://", "https://")):
                result.append(link)
            else:
                raise ValueError(
                    f"Invalid link '{link}': must be either an external URL (starting with http:// or https://) "
                    f"or an internal standard name reference (starting with 'name:')"
                )

        return result

    @field_validator("documentation")
    @classmethod
    def validate_sign_convention_format(cls, v: str) -> str:  # type: ignore[override]
        """Validate sign convention format if present in documentation.

        Enforces consistent formatting:
        - Must use 'Sign convention:' (not '**Sign convention:**' or variations)
        - Must start with 'Positive' followed by a qualifier:
          - 'Sign convention: Positive when <condition>.'
          - 'Sign convention: Positive <quantity-noun-phrase>.'
          - 'Sign convention: Positive for <subject>.'
        - Must follow the main documentation content (cannot be at start)
        - Must be a standalone paragraph (blank line before and after)
        """
        if not v:
            return v

        # Check if sign convention is mentioned
        if re.search(r"\bsign\s+convention\b", v, re.IGNORECASE):
            # Check for bold markdown formatting (not allowed) - check this FIRST
            # Match both **Sign convention:** and **Sign convention**
            if re.search(r"\*\*[Ss]ign\s+[Cc]onvention:?\*\*", v):
                raise ValueError(
                    "Sign convention must use plain text 'Sign convention:', not bold '**Sign convention:**'"
                )

            # Check for lowercase/uppercase issues
            if re.search(r"sign convention:", v):  # lowercase 'sign'
                raise ValueError(
                    "Sign convention format must use title case: 'Sign convention: Positive when [condition].' "
                    "(found lowercase 'sign convention:', should be 'Sign convention:')"
                )
            elif re.search(r"SIGN CONVENTION:", v):  # all caps
                raise ValueError(
                    "Sign convention format must use title case: 'Sign convention: Positive when [condition].' "
                    "(found all caps 'SIGN CONVENTION:', should be 'Sign convention:')"
                )

            # Check for exact format: "Sign convention:" (title case with colon)
            # Must be followed by "Positive" and then a qualifier word
            correct_format = re.search(r"Sign convention:\s+Positive\s+", v)

            if not correct_format:
                # Missing "Positive" keyword
                raise ValueError(
                    "Sign convention must use 'Sign convention: Positive ...' format. "
                    "Accepted forms: 'Positive when <condition>.', "
                    "'Positive <quantity-noun-phrase>.', "
                    "'Positive for <subject>.'."
                )

            # Check for standalone paragraph (must have \n\n before and after)
            # Sign convention must NOT be at document start - must follow main content
            # Find the actual "Sign convention:" text position
            sign_match = re.search(r"Sign convention:[^\n]+", v)
            if sign_match:
                start_pos = sign_match.start()
                end_pos = sign_match.end()

                # Must have content before sign convention (cannot be at document start)
                if start_pos < 2:
                    raise ValueError(
                        "Sign convention must follow the main documentation content. "
                        "It cannot be at the start of the documentation field."
                    )

                # Check if preceded by \n\n
                preceding_text = v[start_pos - 2 : start_pos]
                if preceding_text != "\n\n":
                    raise ValueError(
                        "Sign convention must be a standalone paragraph with a blank line before it. "
                        "Add '\\n\\n' before 'Sign convention:' to separate it from preceding text."
                    )

                # Check if followed by \n\n (or is at end of string)
                if end_pos < len(v):
                    # Look at the 2 characters after the sign convention sentence
                    following_text = v[end_pos : min(len(v), end_pos + 2)]
                    if not following_text.startswith("\n\n") and following_text.strip():
                        raise ValueError(
                            "Sign convention must be a standalone paragraph with a blank line after it. "
                            "Add '\\n\\n' after the sign convention sentence to separate it from following text."
                        )

        return v

    @model_validator(mode="after")
    def _entry_governance_rules(self):  # type: ignore[override]
        # Base class already runs deprecated/superseded_by check; kept for future doc-level rules.
        return self

    @property
    def is_dimensionless(self) -> bool:
        return self.unit == "1"

    def formatted_unit(self, style: str = "plain") -> str:
        if self.is_dimensionless:
            return "1"
        if not pint:
            return self.unit
        u = pint.Unit(self.unit)
        match style:
            case "plain":
                # Plain style now uses pint's pretty (~P) for human readability
                # (may produce Unicode superscripts or ASCII fallbacks)
                s = f"{u:~P}"
                return s.replace("·", "/")  # preserve legacy visual slash style
            case "dotexp":
                # Canonical fused dot-exponent short-symbol format
                return f"{u:~F}"
            case "latex":
                return f"$`{u:L}`$"
            case _:
                raise ValueError(f"Unknown unit style: {style}")


class StandardNameScalarEntry(StandardNameEntryBase):
    """Scalar standard name catalog entry."""

    kind: Literal["scalar"] = "scalar"
    unit: Unit  # Required for scalar (use "1" for dimensionless)
    provenance: Provenance | None = None

    @field_validator("unit", mode="before")
    @classmethod
    def canonicalize_unit_order(cls, v: str) -> str:
        """Auto-correct unit token order to canonical lexicographic form."""
        return cls._canonicalize_unit_order(v)

    @field_validator("unit")
    @classmethod
    def validate_unit_with_pint(cls, v: str) -> str:
        """Validate unit semantics using pint (if available)."""
        return cls._validate_unit_with_pint(v)

    @model_validator(mode="after")
    def _provenance_rules(self):  # type: ignore[override]
        if self.provenance:
            if isinstance(self.provenance, OperatorProvenance):
                self.provenance.operators = _normalize_operator_chain(
                    self.provenance.operators
                )
                _enforce_operator_naming(
                    name=self.name,
                    operators=self.provenance.operators,
                    base=self.provenance.base,
                    operator_id=self.provenance.operator_id,
                    kind=self.kind,
                )
        return self


class StandardNameVectorEntry(StandardNameEntryBase):
    """Vector standard name catalog entry.

    Individual named components (e.g. radial_magnetic_field) are also
    classified as vector because they carry implicit directional structure.
    """

    kind: Literal["vector"] = "vector"
    unit: Unit  # Required for vector (use "1" for dimensionless)
    provenance: Provenance | None = None

    @field_validator("unit", mode="before")
    @classmethod
    def canonicalize_unit_order(cls, v: str) -> str:
        """Auto-correct unit token order to canonical lexicographic form."""
        return cls._canonicalize_unit_order(v)

    @field_validator("unit")
    @classmethod
    def validate_unit_with_pint(cls, v: str) -> str:
        """Validate unit semantics using pint (if available)."""
        return cls._validate_unit_with_pint(v)

    @model_validator(mode="after")
    def _provenance_rules(self):  # type: ignore[override]
        if self.provenance:
            if isinstance(self.provenance, OperatorProvenance):
                self.provenance.operators = _normalize_operator_chain(
                    self.provenance.operators
                )
                _enforce_operator_naming(
                    name=self.name,
                    operators=self.provenance.operators,
                    base=self.provenance.base,
                    operator_id=self.provenance.operator_id,
                    kind=self.kind,
                )
        return self

    @property
    def magnitude(self) -> str:
        """Derived magnitude standard name.

        Conventionally 'magnitude_of_<vector_name>'.
        """
        return f"magnitude_of_{self.name}"


class StandardNameTensorEntry(StandardNameEntryBase):
    """Tensor (rank-2+) standard name catalog entry.

    Used for quantities that represent tensor fields — metric tensors,
    stress tensors, conductivity tensors, etc.  Individual named
    components (e.g. ``g11_covariant_metric_tensor_component``) are
    also classified as *tensor* because they carry implicit index
    structure.
    """

    kind: Literal["tensor"] = "tensor"
    unit: Unit  # Required (use "1" for dimensionless)
    provenance: Provenance | None = None

    @field_validator("unit", mode="before")
    @classmethod
    def canonicalize_unit_order(cls, v: str) -> str:
        """Auto-correct unit token order to canonical lexicographic form."""
        return cls._canonicalize_unit_order(v)

    @field_validator("unit")
    @classmethod
    def validate_unit_with_pint(cls, v: str) -> str:
        """Validate unit semantics using pint (if available)."""
        return cls._validate_unit_with_pint(v)

    @model_validator(mode="after")
    def _provenance_rules(self):  # type: ignore[override]
        if self.provenance:
            if isinstance(self.provenance, OperatorProvenance):
                self.provenance.operators = _normalize_operator_chain(
                    self.provenance.operators
                )
                _enforce_operator_naming(
                    name=self.name,
                    operators=self.provenance.operators,
                    base=self.provenance.base,
                    operator_id=self.provenance.operator_id,
                    kind=self.kind,
                )
        return self


class StandardNameComplexEntry(StandardNameEntryBase):
    """Complex-valued standard name catalog entry.

    Used for quantities that have real and imaginary parts (or
    equivalently, magnitude and phase).  Examples include perturbed
    MHD quantities, wave amplitudes, and impedance components.
    """

    kind: Literal["complex"] = "complex"
    unit: Unit  # Required (use "1" for dimensionless)
    provenance: Provenance | None = None

    @field_validator("unit", mode="before")
    @classmethod
    def canonicalize_unit_order(cls, v: str) -> str:
        """Auto-correct unit token order to canonical lexicographic form."""
        return cls._canonicalize_unit_order(v)

    @field_validator("unit")
    @classmethod
    def validate_unit_with_pint(cls, v: str) -> str:
        """Validate unit semantics using pint (if available)."""
        return cls._validate_unit_with_pint(v)

    @model_validator(mode="after")
    def _provenance_rules(self):  # type: ignore[override]
        if self.provenance:
            if isinstance(self.provenance, OperatorProvenance):
                self.provenance.operators = _normalize_operator_chain(
                    self.provenance.operators
                )
                _enforce_operator_naming(
                    name=self.name,
                    operators=self.provenance.operators,
                    base=self.provenance.base,
                    operator_id=self.provenance.operator_id,
                    kind=self.kind,
                )
        return self


class StandardNameMetadataEntry(StandardNameEntryBase):
    """Metadata standard name catalog entry.

    Used for definitional entries that document concepts, boundaries,
    or reference anchors rather than measurable physical quantities.
    These entries provide documentation and cross-references but do not
    represent data that can be measured or calculated.

    Examples: plasma_boundary, scrape_off_layer, confined_region

    Metadata entries have relaxed validation:
    - No unit field required (these are definitional, not measurable)
    - No provenance required (these are definitional, not derived)
    - Focus on documentation and links fields
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["metadata"] = "metadata"
    # No unit field - metadata entries are definitional, not measurable

    def model_dump(self, **kwargs):
        """Override to ensure consistent serialization for metadata entries."""
        return super().model_dump(**kwargs)


StandardNameEntry = Annotated[
    StandardNameScalarEntry
    | StandardNameVectorEntry
    | StandardNameTensorEntry
    | StandardNameComplexEntry
    | StandardNameMetadataEntry,
    Field(discriminator="kind"),
]

_STANDARD_NAME_ENTRY_ADAPTER = TypeAdapter(StandardNameEntry)


# ---------------------------------------------------------------------------
# Name-only entry classes
# ---------------------------------------------------------------------------
# Support partial validation during LLM-driven generation where only the
# identity+unit portion of a standard name has been composed and the
# description/documentation are filled in by a later enrichment pass.
# These classes inherit governance + grammar + unit validation from
# ``StandardNameBase`` but deliberately omit the documentation fields.


class StandardNameScalarNameOnly(StandardNameBase):
    """Scalar standard name — name + unit only (no documentation)."""

    kind: Literal["scalar"] = "scalar"
    unit: Unit  # Required for scalar (use "1" for dimensionless)

    @field_validator("unit", mode="before")
    @classmethod
    def canonicalize_unit_order(cls, v: str) -> str:
        return cls._canonicalize_unit_order(v)

    @field_validator("unit")
    @classmethod
    def validate_unit_with_pint(cls, v: str) -> str:
        return cls._validate_unit_with_pint(v)


class StandardNameVectorNameOnly(StandardNameBase):
    """Vector standard name — name + unit only (no documentation)."""

    kind: Literal["vector"] = "vector"
    unit: Unit  # Required for vector (use "1" for dimensionless)

    @field_validator("unit", mode="before")
    @classmethod
    def canonicalize_unit_order(cls, v: str) -> str:
        return cls._canonicalize_unit_order(v)

    @field_validator("unit")
    @classmethod
    def validate_unit_with_pint(cls, v: str) -> str:
        return cls._validate_unit_with_pint(v)


class StandardNameTensorNameOnly(StandardNameBase):
    """Tensor standard name — name + unit only (no documentation)."""

    kind: Literal["tensor"] = "tensor"
    unit: Unit  # Required (use "1" for dimensionless)

    @field_validator("unit", mode="before")
    @classmethod
    def canonicalize_unit_order(cls, v: str) -> str:
        return cls._canonicalize_unit_order(v)

    @field_validator("unit")
    @classmethod
    def validate_unit_with_pint(cls, v: str) -> str:
        return cls._validate_unit_with_pint(v)


class StandardNameComplexNameOnly(StandardNameBase):
    """Complex standard name — name + unit only (no documentation)."""

    kind: Literal["complex"] = "complex"
    unit: Unit  # Required (use "1" for dimensionless)

    @field_validator("unit", mode="before")
    @classmethod
    def canonicalize_unit_order(cls, v: str) -> str:
        return cls._canonicalize_unit_order(v)

    @field_validator("unit")
    @classmethod
    def validate_unit_with_pint(cls, v: str) -> str:
        return cls._validate_unit_with_pint(v)


class StandardNameMetadataNameOnly(StandardNameBase):
    """Metadata standard name — name only (no unit, no documentation)."""

    kind: Literal["metadata"] = "metadata"


StandardNameNameOnly = Annotated[
    StandardNameScalarNameOnly
    | StandardNameVectorNameOnly
    | StandardNameTensorNameOnly
    | StandardNameComplexNameOnly
    | StandardNameMetadataNameOnly,
    Field(discriminator="kind"),
]

_NAME_ONLY_ADAPTER = TypeAdapter(StandardNameNameOnly)


def _build_standard_name_models() -> dict[str, type[StandardNameEntry]]:
    """Build mapping of kind string to model class from StandardNameEntry union.

    Extracts model classes from the discriminated union and indexes them by
    their kind literal value. This provides O(1) lookup for loading entries.

    Returns:
        Dictionary mapping kind strings ('scalar', 'vector', 'metadata') to
        their corresponding model classes.
    """
    union_type = get_args(StandardNameEntry)[0]
    model_classes = get_args(union_type)
    return {
        model_class.model_fields["kind"].default: model_class
        for model_class in model_classes
    }


STANDARD_NAME_MODELS = _build_standard_name_models()


class CatalogNameMetadata(BaseModel):
    """Machine-owned metadata the manifest sidecar holds for one entry.

    Everything a physicist does not hand-edit during catalog review lives
    here rather than beside the prose: the algebraic ``kind`` the entry
    union discriminates on, the lifecycle ``status``, the physics domain,
    the source-system bindings with their pinned versions, the reference
    links, and the structural argument edges. The reviewable entry is then
    just ``name``, ``description``, ``documentation`` and ``unit``.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Kind | None = None
    status: Status | None = None
    physics_domain: PhysicsDomain | None = None
    links: Links | None = None
    arguments: list[ArgumentRef] | None = None
    sources: list["CatalogSourceBinding"] | None = None

    def declared(self) -> dict[str, Any]:
        """Return only the fields this block actually carries, JSON-shaped."""
        return self.model_dump(mode="json", exclude_none=True)


class StandardNameCatalogManifest(BaseModel):
    """Catalog-level manifest describing a published standard names catalog.

    Placed at the repository root (``catalog.yml``) and records run-level
    metadata about the export that produced the catalog. Individual entries
    carry only editorial fields; generation provenance and every
    machine-owned per-entry field live here, the latter under ``names``.
    """

    model_config = ConfigDict(extra="forbid")

    catalog_name: str
    cocos_convention: int
    grammar_version: str
    isn_model_version: str
    dd_version_lineage: list[str]
    generated_by: str
    generated_at: datetime
    min_score_applied: float | None = None
    min_description_score_applied: float | None = None
    include_unreviewed: bool = False
    candidate_count: int
    published_count: int
    excluded_below_score_count: int = 0
    excluded_unreviewed_count: int = 0
    source_repo: str | None = None
    source_commit_sha: str | None = None
    export_scope: Literal["full", "domain", "scoped", "review"] | None = None
    domains_included: list[str] = Field(default_factory=list)
    # Review-batch id-set: on a ``review`` export this holds the exact
    # standard-name ids in the PR'd batch; absent on normal builds.
    review_batch: list[str] | None = None
    catalog_commit_sha: str | None = None
    exported_at: datetime | None = None
    edge_model_version: str = CATALOG_EDGE_MODEL_VERSION
    # Machine-owned per-entry metadata, keyed by standard name. A catalog
    # whose entries still carry these fields inline leaves this empty.
    names: dict[str, CatalogNameMetadata] = Field(default_factory=dict)

    def resolve_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Return ``entry`` overlaid with the sidecar block for its name.

        The sidecar is authoritative for the fields it declares, so a
        reviewable entry that carries only prose still gains its ``kind``
        before the entry union discriminates on it.
        """
        resolved = dict(entry)
        block = self.names.get(str(entry.get("name") or ""))
        if block is not None:
            resolved.update(block.declared())
        return resolved


def create_standard_name_entry(
    data: dict,
    *,
    name_only: bool = False,
    manifest: StandardNameCatalogManifest | None = None,
) -> StandardNameEntry | StandardNameNameOnly:
    """Validate data into a StandardName entry instance.

    Args:
        data: Entry dictionary. Must include ``kind`` for discrimination,
            either inline or through ``manifest``.
        name_only: When ``True``, validate against the lightweight name-only
            union (identity + unit) that omits description/documentation/tags.
            Use this during early LLM generation passes. Defaults to ``False``
            for full catalog-entry validation.
        manifest: Catalog manifest whose per-name sidecar block supplies the
            machine-owned fields — including ``kind`` — that the reviewable
            entry no longer carries. Applied before discrimination.
    """
    if manifest is not None:
        data = manifest.resolve_entry(data)
    if name_only:
        return _NAME_ONLY_ADAPTER.validate_python(data)
    return _STANDARD_NAME_ENTRY_ADAPTER.validate_python(data)


def load_standard_name_entry(data: dict) -> StandardNameEntry:
    """Load a StandardNameEntry instance without validation (bypasses validators)."""
    kind = data.get("kind")
    if not kind:
        raise ValueError("Missing required field 'kind' in data dictionary")

    kind_str = kind.value if isinstance(kind, Kind) else kind
    model_class = STANDARD_NAME_MODELS.get(kind_str)

    if not model_class:
        valid_kinds = ", ".join(STANDARD_NAME_MODELS.keys())
        raise ValueError(f"Unknown kind: {kind_str}. Valid kinds: {valid_kinds}")

    return model_class.model_construct(**data)


__all__ = [
    "ArgumentRef",
    "Kind",
    "Status",
    "OperatorProvenance",
    "ExpressionProvenance",
    "StandardNameBase",
    "StandardNameEntryBase",
    "StandardNameScalarEntry",
    "StandardNameVectorEntry",
    "StandardNameTensorEntry",
    "StandardNameComplexEntry",
    "StandardNameMetadataEntry",
    "StandardNameScalarNameOnly",
    "StandardNameVectorNameOnly",
    "StandardNameTensorNameOnly",
    "StandardNameComplexNameOnly",
    "StandardNameMetadataNameOnly",
    "StandardNameNameOnly",
    "Name",
    "StandardNameEntry",
    "CatalogNameMetadata",
    "CatalogSourceBinding",
    "StandardNameCatalogManifest",
    "CATALOG_EDGE_MODEL_VERSION",
    "STANDARD_NAME_MODELS",
    "create_standard_name_entry",
    "load_standard_name_entry",
]
