import yaml

from imas_standard_names.models import create_standard_name_entry


def test_save_and_load_roundtrip(tmp_path, scalar_data):
    entry = create_standard_name_entry(scalar_data)
    path = tmp_path / f"{entry.name}.yml"
    data = {
        k: v
        for k, v in entry.model_dump(mode="json").items()
        if v not in (None, [], "")
    }
    data["name"] = entry.name
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    loaded = create_standard_name_entry(
        yaml.safe_load(path.read_text(encoding="utf-8"))
    )
    assert loaded == entry


def test_catalog_duplicate_detection(tmp_path, scalar_data):
    entry = create_standard_name_entry(scalar_data)
    base_path = tmp_path / f"{entry.name}.yml"
    base_path.write_text(
        yaml.safe_dump(entry.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    # Duplicate yaml with different extension
    (tmp_path / f"{entry.name}.yaml").write_text(base_path.read_text(encoding="utf-8"))
    # Manual duplicate detection pass
    seen = set()
    duplicate = False
    for p in tmp_path.rglob("*.yml"):
        d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if d.get("name") in seen:
            duplicate = True
            break
        seen.add(d.get("name"))
    for p in tmp_path.rglob("*.yaml"):
        d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if d.get("name") in seen:
            duplicate = True
            break
        seen.add(d.get("name"))
    assert duplicate, "Expected duplicate detection"
