from pathlib import Path

import pytest
import yaml

from imas_standard_names.models import create_standard_name_entry


@pytest.fixture
def scalar_data():
    return {
        "kind": "scalar",
        "name": "electron_temperature",
        "physics_domain": "core_plasma_physics",
        "description": "Core electron temperature",
        "documentation": "Temperature of electrons in the plasma core.",
        "unit": "eV",
        "status": "active",
    }


@pytest.fixture
def vector_data():
    return {
        "kind": "vector",
        "name": "plasma_velocity",
        "physics_domain": "core_plasma_physics",
        "description": "Bulk plasma velocity",
        "documentation": "Velocity vector of the bulk plasma flow.",
        "unit": "m.s^-1",
        "status": "active",
    }


@pytest.fixture
def operator_scalar_data():
    return {
        "kind": "scalar",
        "name": "divergence_of_plasma_velocity",
        "physics_domain": "core_plasma_physics",
        "description": "Divergence of velocity",
        "documentation": "Divergence operator applied to plasma velocity field.",
        "unit": "s^-1",
        "status": "active",
        "provenance": {
            "mode": "operator",
            "operators": ["divergence"],
            "base": "plasma_velocity",
            "operator_id": "divergence",
        },
    }


@pytest.fixture
def gradient_vector_data():
    return {
        "kind": "vector",
        "name": "gradient_of_electron_temperature",
        "physics_domain": "core_plasma_physics",
        "description": "Spatial gradient of Te",
        "documentation": "Gradient operator applied to electron temperature field.",
        "unit": "eV.m^-1",
        "status": "active",
        "provenance": {
            "mode": "operator",
            "operators": ["gradient"],
            "base": "electron_temperature",
            "operator_id": "gradient",
        },
    }


@pytest.fixture
def expression_scalar_data():
    return {
        "kind": "scalar",
        "name": "pressure_balance_indicator",
        "physics_domain": "core_plasma_physics",
        "description": "Derived scalar from multiple quantities",
        "documentation": "Indicator derived from electron and ion temperatures via expression.",
        "unit": "1",
        "status": "draft",
        "provenance": {
            "mode": "expression",
            "expression": "electron_temperature * ion_temperature",
            "dependencies": ["electron_temperature", "ion_temperature"],
        },
    }


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def save_and_load_scalar(temp_dir, scalar_data):
    entry = create_standard_name_entry(scalar_data)
    path = Path(temp_dir) / f"{entry.name}.yml"
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
    return path, loaded
