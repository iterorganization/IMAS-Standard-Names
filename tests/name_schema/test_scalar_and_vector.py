import pytest

from imas_standard_names.models import (
    StandardNameScalarEntry,
    StandardNameVectorEntry,
    create_standard_name_entry,
)


def test_scalar_creation(scalar_data):
    sn = create_standard_name_entry(scalar_data)
    assert isinstance(sn, StandardNameScalarEntry)
    assert sn.name == scalar_data["name"]
    assert sn.formatted_unit() != ""  # eV formatted


def test_vector_creation(vector_data):
    sn = create_standard_name_entry(vector_data)
    assert isinstance(sn, StandardNameVectorEntry)
    # Vector no longer has components field in schema
    # Components are specified at runtime via vector_axes metadata
    assert sn.magnitude == "magnitude_of_plasma_velocity"


@pytest.mark.parametrize("bad_name", ["ElectronTemp", "1temp", "temp__double"])
def test_invalid_name_rejected(scalar_data, bad_name):
    data = scalar_data | {"name": bad_name}
    with pytest.raises((ValueError, KeyError, TypeError)):
        create_standard_name_entry(data)


def test_dimensionless_unit_blank():
    sn = create_standard_name_entry(
        {
            "kind": "scalar",
            "physics_domain": "general",
            "name": "beta_pol",
            "description": "Dimensionless plasma beta",
            "documentation": "Poloidal beta - dimensionless plasma parameter.",
            "unit": "1",
            "status": "active",
        }
    )
    assert sn.unit == "1"  # canonical form for dimensionless
    assert sn.is_dimensionless
