"""YAML persistence utilities (authoritative storage)."""

import logging
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from .models import (
    StandardNameEntry,
    StandardNameScalarEntry,
    create_standard_name_entry,
)
from .services import validate_models

logger = logging.getLogger(__name__)

# Fields that are no longer part of the catalog entry model.
# They are stripped from loaded YAML data to support clean schema migration.
_STRIPPED_FIELDS = {"dd_paths"}
_PROSE_FIELDS = {"description", "documentation"}

# Reviewers read the catalog diff in a side-by-side view. Prose is emitted as a
# literal block whose paragraphs are hard-wrapped at this column, so a paragraph
# edit shows as a few changed lines instead of one, and a paragraph break costs
# one blank line rather than the two a folded block needs to encode it.
_PROSE_WRAP_COLUMN = 80
# Entries are dumped as items of a sequence, so block scalar content sits two
# indentation levels in.
_PROSE_CONTENT_INDENT = 4
_PROSE_CONTENT_WIDTH = _PROSE_WRAP_COLUMN - _PROSE_CONTENT_INDENT


class _LiteralBlockString(str):
    """String rendered with YAML literal-block style."""


class _CatalogDumper(yaml.SafeDumper):
    """Safe dumper carrying the catalog's review-oriented scalar styles."""


def _represent_literal_block(
    dumper: yaml.SafeDumper, value: _LiteralBlockString
) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style="|")


_CatalogDumper.add_representer(_LiteralBlockString, _represent_literal_block)


def _is_list_item(text: str) -> bool:
    if text.startswith(("- ", "* ", "+ ")):
        return True
    marker, separator, _ = text.partition(". ")
    return bool(separator and marker.isdigit())


def _break_positions(line: str) -> list[int]:
    """Single spaces that rejoin to exactly one space when reloaded.

    A break beside another space would put a space at the start or end of an
    emitted line, and rejoining inserts one space of its own, so the reloaded
    run of spaces would not match the authored one.
    """
    return [
        index
        for index, char in enumerate(line)
        if char == " "
        and index > 0
        and line[index - 1] != " "
        and index + 1 < len(line)
        and line[index + 1] != " "
    ]


def _wrap_prose_line(line: str, width: int) -> list[str]:
    """Split one paragraph line into soft-wrapped lines of at most ``width``."""
    wrapped: list[str] = []
    remainder = line
    while len(remainder) > width:
        positions = _break_positions(remainder)
        if not positions:
            break
        within_width = [index for index in positions if index <= width]
        # An unbreakable run longer than the column overflows by one word
        # rather than being split inside the word.
        break_at = within_width[-1] if within_width else positions[0]
        wrapped.append(remainder[:break_at])
        remainder = remainder[break_at + 1 :]
    wrapped.append(remainder)
    return wrapped


def _keeps_authored_layout(line: str, in_display_equation: bool) -> bool:
    """Whether ``line`` is structure that must survive the round trip verbatim.

    Blank lines, display-equation fences and their contents, list items, and
    indented lines carry meaning in their own layout, so they are neither
    wrapped on write nor rejoined on read.
    """
    return (
        in_display_equation
        or not line.strip()
        or line.startswith(" ")
        or line.strip().startswith("$$")
        or _is_list_item(line)
    )


def _is_display_equation_fence(line: str) -> bool:
    return line.strip() == "$$"


def wrap_catalog_prose(text: str, width: int = _PROSE_CONTENT_WIDTH) -> str:
    """Soft-wrap paragraph lines of catalog prose, leaving structure alone."""
    output: list[str] = []
    in_display_equation = False

    for line in text.split("\n"):
        if _keeps_authored_layout(line, in_display_equation):
            output.append(line)
        else:
            output.extend(_wrap_prose_line(line, width))
        if _is_display_equation_fence(line):
            in_display_equation = not in_display_equation

    return "\n".join(output)


def unwrap_catalog_prose(text: str) -> str:
    """Rejoin the soft wrap breaks :func:`wrap_catalog_prose` introduced.

    Consecutive paragraph lines rejoin with the single space the break
    replaced; structure recognised by :func:`_keeps_authored_layout` is left
    exactly as written, so an authored string survives a write and read
    unchanged.
    """
    output: list[str] = []
    in_display_equation = False
    joins_with_previous = False

    for line in text.split("\n"):
        if _keeps_authored_layout(line, in_display_equation):
            output.append(line)
            joins_with_previous = False
        elif joins_with_previous:
            output[-1] = f"{output[-1]} {line}"
        else:
            output.append(line)
            joins_with_previous = True
        if _is_display_equation_fence(line):
            in_display_equation = not in_display_equation

    return "\n".join(output)


def _review_friendly_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    rendered = dict(entry)
    for field in _PROSE_FIELDS:
        value = rendered.get(field)
        if isinstance(value, str) and value:
            rendered[field] = _LiteralBlockString(wrap_catalog_prose(value))
    return rendered


def dump_catalog_yaml(entries: Sequence[Mapping[str, Any]]) -> str:
    """Serialize catalog entries as review-friendly YAML.

    Each entry is emitted as one item in the same YAML sequence and separated
    from the next by one blank line. Unicode remains literal. Description and
    documentation prose is emitted as a literal block carrying hard-wrapped
    paragraphs; equations, list items and blank lines keep their authored
    layout, and :class:`YamlStore` rejoins the wrap breaks on load so the
    string a reader gets back is the one that was written.
    """
    if not entries:
        return "[]\n"

    rendered_entries = [
        yaml.dump(
            [_review_friendly_entry(entry)],
            Dumper=_CatalogDumper,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
            # Block scalars are wrapped before the dump, and no other field
            # carries prose, so the emitter never breaks a line itself.
            width=10**9,
        ).rstrip("\n")
        for entry in entries
    ]
    return "\n\n".join(rendered_entries) + "\n"


def write_catalog_yaml(path: str | Path, entries: Sequence[Mapping[str, Any]]) -> None:
    """Write catalog entries to ``path`` using the canonical YAML rendering."""
    Path(path).write_text(dump_catalog_yaml(entries), encoding="utf-8")


class CatalogMigrationError(Exception):
    """Raised when a legacy catalog layout is detected."""


class YamlStore:
    def __init__(self, root: str | Path, permissive: bool = False):
        self.root = Path(root).expanduser().resolve()
        self.permissive = permissive
        self.validation_warnings: list[str] = []

    # Discovery ---------------------------------------------------------------
    def yaml_files(self):
        return sorted(list(self.root.rglob("*.yml")) + list(self.root.rglob("*.yaml")))

    # Load --------------------------------------------------------------------
    def load(self) -> list[StandardNameEntry]:
        models: list[StandardNameEntry] = []
        for f in self.yaml_files():
            # Detect nested paths (legacy per-file layout)
            relative = f.relative_to(self.root)
            if len(relative.parts) > 1:
                if not self.permissive:
                    raise CatalogMigrationError(
                        f"Legacy per-file YAML detected at {f}; catalog has migrated "
                        "to per-domain list format. Re-run `sn publish` "
                        "from imas-codex to regenerate."
                    )
                # In permissive mode, fall through and process as single-entry dict

            with open(f, encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}

            # Determine entries to process from this file
            if isinstance(data, list):
                entries = data
            elif isinstance(data, dict):
                if "name" in data:
                    entries = [data]
                elif not self.permissive:
                    raise CatalogMigrationError(
                        f"Legacy per-file YAML detected at {f}; catalog has migrated "
                        "to per-domain list format. Re-run `sn publish` "
                        "from imas-codex to regenerate."
                    )
                else:
                    continue
            else:
                continue

            for entry_data in entries:
                if not isinstance(entry_data, dict) or "name" not in entry_data:
                    continue

                unit_val = entry_data.get("unit")
                if isinstance(unit_val, int | float):
                    entry_data["unit"] = str(unit_val)

                # Strip fields no longer in the catalog entry model
                for field in _STRIPPED_FIELDS:
                    entry_data.pop(field, None)

                # Rejoin the soft wrap breaks the writer introduced, so the
                # model carries the prose as it was authored.
                for field in _PROSE_FIELDS:
                    prose = entry_data.get(field)
                    if isinstance(prose, str):
                        entry_data[field] = unwrap_catalog_prose(prose)

                # Handle Pydantic validation errors in permissive mode
                try:
                    m = create_standard_name_entry(entry_data)
                    models.append(m)
                except Exception as e:
                    if self.permissive:
                        # Load invalid entry anyway by creating object without validation
                        # Use object.__new__ to bypass __init__ and all validators
                        m = object.__new__(StandardNameScalarEntry)
                        # Manually set fields from data
                        for key, value in entry_data.items():
                            object.__setattr__(m, key, value)
                        # Set defaults for missing required fields
                        for attr, default in [
                            ("kind", "scalar"),
                            ("status", "draft"),
                            ("unit", ""),
                            ("tags", []),
                            ("links", []),
                            ("documentation", ""),
                            ("deprecates", ""),
                            ("superseded_by", ""),
                            ("provenance", None),
                            ("arguments", None),
                            ("error_variants", None),
                        ]:
                            if not hasattr(m, attr):
                                object.__setattr__(m, attr, default)
                        models.append(m)
                        warning = f"Validation error in {f.name}: {e}"
                        self.validation_warnings.append(warning)
                    else:
                        raise  # Re-raise in strict mode

        # Cross-reference warnings for arguments and error_variants
        all_names = {m.name for m in models}
        for m in models:
            args = getattr(m, "arguments", None)
            if args:
                for arg in args:
                    arg_name = (
                        getattr(arg, "name", None) if not isinstance(arg, str) else arg
                    )
                    if arg_name and arg_name not in all_names:
                        w = (
                            f"Entry '{m.name}': argument reference "
                            f"'{arg_name}' not found in catalog"
                        )
                        self.validation_warnings.append(w)
                        warnings.warn(w, stacklevel=1)
            evars = getattr(m, "error_variants", None)
            if evars and isinstance(evars, dict):
                for error_key, target in evars.items():
                    if target not in all_names:
                        w = (
                            f"Entry '{m.name}': error_variant '{error_key}' "
                            f"target '{target}' not found in catalog"
                        )
                        self.validation_warnings.append(w)
                        warnings.warn(w, stacklevel=1)

        # Separate warning/info-severity issues from genuine errors. The
        # semantic checks tag their messages with " WARNING - " or " INFO -
        # " prefixes; those should not abort loading in strict mode.
        # Structural checks emit untagged messages, which are treated as
        # errors.
        issues = validate_models({m.name: m for m in models})
        if issues:
            errors = [
                i for i in issues if " WARNING - " not in i and " INFO - " not in i
            ]
            advisory = [i for i in issues if i not in errors]

            for note in advisory:
                self.validation_warnings.append(f"Structural: {note}")
                warnings.warn(note, stacklevel=1)

            if errors:
                if self.permissive:
                    self.validation_warnings.extend(
                        [f"Structural: {issue}" for issue in errors]
                    )
                else:
                    raise ValueError(
                        "Structural validation failed on load:\n" + "\n".join(errors)
                    )
        return models


__all__ = [
    "CatalogMigrationError",
    "YamlStore",
    "dump_catalog_yaml",
    "unwrap_catalog_prose",
    "wrap_catalog_prose",
    "write_catalog_yaml",
]
