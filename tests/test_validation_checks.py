from pathlib import Path

import yaml

from imas_standard_names.models import create_standard_name_entry
from imas_standard_names.validation.semantic import run_semantic_checks
from imas_standard_names.validation.structural import run_structural_checks


def test_structural_and_semantic_checks(tmp_path: Path):
    # base vector - components no longer in schema, specified via metadata
    (tmp_path / "gradient.yml").write_text(
        """name: gradient_of_temperature
kind: vector
physics_domain: core_plasma_physics
status: draft
unit: K.m^-1
description: Gradient of temperature.
documentation: |
  Gradient of temperature for structural and semantic validation testing.
provenance:
  mode: operator
  operators: [gradient]
  base: temperature
  operator_id: gradient
""",
        encoding="utf-8",
    )
    (tmp_path / "radial_component_of_gradient_of_temperature.yml").write_text(
        """name: radial_component_of_gradient_of_temperature
kind: scalar
physics_domain: core_plasma_physics
status: draft
unit: K.m^-1
description: Radial component.
documentation: |
  Radial component of gradient of temperature for structural and semantic validation testing.
provenance:
  mode: operator
  operators: [gradient]
  base: temperature
  operator_id: gradient
""",
        encoding="utf-8",
    )
    (tmp_path / "toroidal_component_of_gradient_of_temperature.yml").write_text(
        """name: toroidal_component_of_gradient_of_temperature
kind: scalar
physics_domain: core_plasma_physics
status: draft
unit: K.m^-1
description: Toroidal component.
documentation: |
  Toroidal component of gradient of temperature for structural and semantic validation testing.
provenance:
  mode: operator
  operators: [gradient]
  base: temperature
  operator_id: gradient
""",
        encoding="utf-8",
    )
    entries = {}
    for p in tmp_path.rglob("*.yml"):
        d = yaml.safe_load(p.read_text(encoding="utf-8"))
        m = create_standard_name_entry(d)
        entries[m.name] = m
    structural_issues = run_structural_checks(entries)
    semantic_issues = run_semantic_checks(entries)
    # Semantic heuristic: gradient expects derivative-like units (contains '/' or .m)
    assert not structural_issues
    assert not semantic_issues  # units include /m via K/m
