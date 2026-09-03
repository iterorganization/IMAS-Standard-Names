"""Validated loaders for grammar guidance resources."""

from functools import lru_cache
from importlib import resources
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from imas_standard_names.grammar.constants import (
    SEGMENT_TOKEN_MAP,
    TRANSFORMATION_TOKENS,
)
from imas_standard_names.grammar.support import TOKEN_PATTERN


class FrozenDict[K, V](dict[K, V]):
    """JSON-serializable mapping that rejects mutation."""

    @staticmethod
    def _immutable(*args: object, **kwargs: object) -> None:
        raise TypeError("frozen mapping does not support mutation")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __copy__(self) -> "FrozenDict[K, V]":
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> "FrozenDict[K, V]":
        return self


# Aliases are keyed by grammar segment. Prefix operators are not a segment —
# they wrap a whole expression — so the transformation vocabulary is offered
# under its own key for retired operator spellings.
ALIAS_VOCABULARIES: dict[str, tuple[str, ...]] = {
    **SEGMENT_TOKEN_MAP,
    "transformation": tuple(TRANSFORMATION_TOKENS),
}


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            msg = f"duplicate advisory alias key: {key!r}"
            raise ValueError(msg)
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


class _AdvisoryAliasDefinition(BaseModel):
    """One guidance-only source-to-canonical mapping."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    canonical: str = Field(pattern=TOKEN_PATTERN.pattern)
    reason: str = Field(min_length=20)


class _AdvisoryAliasResource(BaseModel):
    """Segment-scoped advisory alias registry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    segments: dict[str, dict[str, _AdvisoryAliasDefinition]]


def _freeze_mapping(value: dict[str, Any]) -> FrozenDict[str, Any]:
    return FrozenDict(
        {
            key: _freeze_mapping(item) if isinstance(item, dict) else item
            for key, item in value.items()
        }
    )


def freeze_advisory_aliases(
    aliases: dict[str, Any],
) -> FrozenDict[str, Any]:
    """Return an immutable JSON-compatible advisory alias mapping."""
    return _freeze_mapping(aliases)


@lru_cache(maxsize=1)
def load_advisory_aliases() -> FrozenDict[str, Any]:
    """Load guidance aliases and validate every target against its segment."""
    path = (
        resources.files("imas_standard_names.grammar")
        / "vocabularies"
        / "advisory_aliases.yml"
    )
    with path.open(encoding="utf-8") as handle:
        raw = yaml.load(handle, Loader=_UniqueKeyLoader) or {}
    resource = _AdvisoryAliasResource.model_validate(raw)

    seen_sources: set[str] = set()
    for segment, definitions in resource.segments.items():
        if segment not in ALIAS_VOCABULARIES:
            msg = f"unknown advisory alias segment: {segment!r}"
            raise ValueError(msg)
        canonical_tokens = set(ALIAS_VOCABULARIES[segment])
        for source, definition in definitions.items():
            if not TOKEN_PATTERN.fullmatch(source):
                msg = f"invalid advisory alias token: {source!r}"
                raise ValueError(msg)
            if source in seen_sources:
                msg = f"advisory alias source appears in multiple segments: {source!r}"
                raise ValueError(msg)
            if source in canonical_tokens:
                msg = (
                    f"advisory alias source {source!r} collides with the "
                    f"{segment!r} vocabulary"
                )
                raise ValueError(msg)
            if definition.canonical not in canonical_tokens:
                msg = (
                    f"advisory alias target {definition.canonical!r} is not in "
                    f"the {segment!r} vocabulary"
                )
                raise ValueError(msg)
            seen_sources.add(source)

    serializable = {
        segment: {
            source: definition.model_dump(mode="json")
            for source, definition in definitions.items()
        }
        for segment, definitions in resource.segments.items()
    }
    return freeze_advisory_aliases(serializable)


__all__ = ["ALIAS_VOCABULARIES", "freeze_advisory_aliases", "load_advisory_aliases"]
