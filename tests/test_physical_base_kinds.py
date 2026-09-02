import hashlib

from imas_standard_names.grammar.vocab_loaders import load_physical_bases

_VECTOR_CORRECTIONS = {"momentum", "torque", "torque_density"}
_BASELINE_UNCHANGED_KIND_DIGEST = (
    "704ef5143e7a8e8f399cf38e3df0c53e09ca6febe1874f1dca746b89cf2d49a9"
)


def _kind_map_digest(kind_map: dict[str, str]) -> str:
    payload = "".join(
        f"{name}={kind}\n"
        for name, kind in sorted(kind_map.items())
        if name not in _VECTOR_CORRECTIONS
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def test_only_momentum_and_torque_bases_change_to_vector() -> None:
    kinds = {
        name: definition.kind
        for name, definition in load_physical_bases().bases.items()
    }

    assert {name: kinds[name] for name in _VECTOR_CORRECTIONS} == dict.fromkeys(
        _VECTOR_CORRECTIONS, "vector"
    )
    assert _kind_map_digest(kinds) == _BASELINE_UNCHANGED_KIND_DIGEST

    baseline_kinds = kinds | dict.fromkeys(_VECTOR_CORRECTIONS, "scalar")
    changed = {name for name, kind in kinds.items() if kind != baseline_kinds.get(name)}
    assert changed == _VECTOR_CORRECTIONS
