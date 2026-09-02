# Catalog entry schema

Every full catalog entry is validated by `StandardNameEntryBase` and one of its
kind-specific models. The following fields are required for scalar, vector,
tensor, complex, and metadata entries:

| Field | Type | Meaning |
| --- | --- | --- |
| `name` | standard-name token | Grammar-valid identity of the quantity. |
| `kind` | `scalar`, `vector`, `tensor`, `complex`, or `metadata` | Algebraic entry kind. |
| `physics_domain` | `PhysicsDomain` | Primary physics classification and catalog Domain grouping. |
| `description` | string | Concise description of the quantity. |
| `documentation` | string | Standalone documentation for the entry. |

Scalar, vector, tensor, and complex entries also require `unit`. Metadata
entries are definitional and do not carry a unit.

`physics_domain` is a first-class model field, not a secondary tag. It must be
one of the values declared by `PhysicsDomain` in
`imas_standard_names.grammar.tag_types`; unknown values and omitted fields are
validation errors. The catalog site uses this field directly to build its
Domain groups.

```yaml
name: electron_temperature
kind: scalar
physics_domain: core_plasma_physics
description: Electron temperature.
documentation: Kinetic temperature of the electron population.
unit: eV
status: active
```

Optional fields include `status`, `tags`, `links`, `sources`, graph edges, and
operator provenance. `tags` remains available for secondary, cross-cutting
classification and does not replace `physics_domain`.
