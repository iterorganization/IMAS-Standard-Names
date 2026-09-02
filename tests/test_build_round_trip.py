from pathlib import Path

import yaml

from imas_standard_names.database.build import build_catalog
from imas_standard_names.database.read import CatalogRead
from imas_standard_names.repository import StandardNameCatalog


def _write_yaml(root: Path, model):
    """Write a standard name model to a per-domain list YAML file."""
    domain = getattr(model, "physics_domain", "general") or "general"
    domain_file = root / f"{domain}.yml"
    data = {
        k: v
        for k, v in model.model_dump(mode="json").items()
        if v not in (None, [], "")
    }
    data["name"] = model.name
    data["physics_domain"] = str(domain)
    existing: list[dict] = []
    if domain_file.exists():
        with open(domain_file, encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
        if isinstance(loaded, list):
            existing = loaded
    existing.append(data)
    with open(domain_file, "w", encoding="utf-8") as fh:
        yaml.safe_dump(existing, fh, sort_keys=False, allow_unicode=True, width=80)


def test_build_catalog_round_trip(tmp_path: Path, example_scalars):
    # Use examples from catalog
    yaml_root = tmp_path / "standard_names"
    yaml_root.mkdir()

    for example in example_scalars[:2]:
        _write_yaml(yaml_root, example)

    # Load via repository (in-memory) baseline
    repo = StandardNameCatalog(yaml_root)
    baseline = {m.name: m.description for m in repo.list()}
    # Build file-backed catalog
    db_path = tmp_path / "artifacts" / "catalog.db"
    build_catalog(yaml_root, db_path, overwrite=True)
    assert db_path.exists()
    # Open read-only
    ro = CatalogRead(db_path)
    rebuilt = {m.name: m.description for m in ro.list()}
    assert baseline == rebuilt
    # Ensure FTS search works identically
    first_name = example_scalars[0].name
    results = ro.search(first_name)
    assert first_name in results
