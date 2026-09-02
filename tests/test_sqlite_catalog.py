from imas_standard_names.database.readwrite import CatalogReadWrite
from imas_standard_names.models import create_standard_name_entry
from imas_standard_names.services import row_to_model


def test_sqlite_catalog_insert_search_get():
    cat = CatalogReadWrite()
    m = create_standard_name_entry(
        {
            "name": "electron_temperature",
            "kind": "scalar",
            "physics_domain": "core_plasma_physics",
            "description": "Electron temperature.",
            "documentation": "Temperature of electrons in the plasma.",
            "unit": "eV",
            "status": "active",
        }
    )
    cat.insert(m)
    assert cat.get("electron_temperature") is not None
    results = cat.search("electron_temperature")
    assert "electron_temperature" in results
    # round-trip conversion
    raw_row = cat.conn.execute(
        "SELECT * FROM standard_name WHERE name=?", ("electron_temperature",)
    ).fetchone()
    model2 = row_to_model(cat.conn, raw_row)
    assert model2.name == m.name
