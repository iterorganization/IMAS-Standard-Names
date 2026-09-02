from imas_standard_names.database.readwrite import CatalogReadWrite
from imas_standard_names.models import create_standard_name_entry
from imas_standard_names.services import row_to_model, validate_models


def test_services_validate_and_row_to_model():
    cat = CatalogReadWrite()
    model = create_standard_name_entry(
        {
            "name": "electron_density",
            "kind": "scalar",
            "physics_domain": "core_plasma_physics",
            "description": "Electron density.",
            "documentation": "Number density of electrons in the plasma.",
            "unit": "m^-3",  # canonical single token style for test purposes
        }
    )
    issues = validate_models({model.name: model})
    assert issues == []
    cat.insert(model)
    raw_row = cat.conn.execute(
        "SELECT * FROM standard_name WHERE name=?", ("electron_density",)
    ).fetchone()
    round_trip = row_to_model(cat.conn, raw_row)
    assert round_trip.name == model.name
