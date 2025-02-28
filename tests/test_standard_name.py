import json
import pydantic
import pytest
import strictyaml as syaml

from imas_standard_names.standard_name import (
    StandardName,
    ParseYaml,
    ParseJson,
    StandardNameFile,
    StandardInput,
)

standard_name_data = {
    "name": "ion_temperature",
    "units": "A",
    "documentation": "multi-line\ndoc string",
    "tags": "",
    "alias": "",
    "overwrite": False,
}

yaml_single = syaml.as_document(
    {
        standard_name_data["name"]: {
            key: value for key, value in standard_name_data.items() if key != "name"
        }
    }
)

yaml_multi = syaml.as_document(
    {
        name: {"units": units, "documentation": "docs"}
        for name, units in zip(
            ["plasma_current", "plasma_current_density", "electron_temperature"],
            ["A", "A/m^2", "eV"],
        )
    }
)


def test_standard_name():
    standard_name = StandardName(**standard_name_data)
    for key, value in standard_name_data.items():
        assert getattr(standard_name, key) == value


@pytest.mark.parametrize(
    "name",
    ["1st_plasma", "Main_ion_density", "_private", "plasma current", "plasmaCurrent"],
)
def test_name_validator(name):
    with pytest.raises(NameError):
        StandardName(name=name, units="A", documentation="docs")


def test_units_validator():
    with pytest.raises(pydantic.ValidationError):
        StandardName(name="plasma_current", units=1, documentation="docs")


def test_docs_validator():
    with pytest.raises(pydantic.ValidationError):
        StandardName(name="plasma_current", units="A", documentation=None)


def test_alias_validator():
    with pytest.raises(pydantic.ValidationError):
        StandardName(name="plasma_current", units="A", documentation="docs", alias=1)


def test_yaml_input():
    standard_name = ParseYaml(yaml_single.as_yaml())[standard_name_data["name"]]
    for key, value in standard_name_data.items():
        if key == "name":
            continue
        assert getattr(standard_name, key) == value


def test_json_input():
    github_response = json.dumps(standard_name_data)
    standard_name = ParseJson(github_response)[standard_name_data["name"]]
    for key, value in standard_name_data.items():
        if key == "name":
            continue
        assert getattr(standard_name, key) == value


def test_json_name():
    github_response = json.dumps(standard_name_data)
    assert ParseJson(github_response).name == standard_name_data["name"]


def test_yaml_json_equality():
    yaml_standard_name = ParseYaml(yaml_single.as_yaml())[standard_name_data["name"]]
    json_standard_name = ParseJson(json.dumps(standard_name_data)).standard_name
    assert yaml_standard_name == json_standard_name


def test_yaml_roundtrip():
    standard_name = ParseYaml(yaml_multi.as_yaml())
    for key in yaml_multi.as_marked_up():
        assert (
            ParseYaml(standard_name[key].as_yaml())[key]
            == ParseYaml(yaml_multi.as_yaml())[key]
        )


@pytest.fixture()
def standardnames(tmp_path_factory):
    filepath = tmp_path_factory.mktemp("data") / "standardnames.yaml"
    with open(filepath, "w") as f:
        f.write(yaml_multi.as_yaml())
    return filepath


def test_standard_name_file_without_suffix(standardnames):
    StandardNameFile(standardnames.with_suffix(""))


def test_standard_name_file(standardnames):
    standard_input = ParseYaml(yaml_multi.as_yaml())
    standard_names = StandardNameFile(standardnames)
    for key in yaml_multi.as_marked_up():
        assert standard_names[key] == standard_input[key]


def test_file_update(standardnames):
    standard_names = StandardNameFile(standardnames)
    assert standard_name_data["name"] not in standard_names.data
    standard_names_document = standard_names.data.whole_document()
    github_response = json.dumps(standard_name_data)
    standard_name = ParseJson(github_response).standard_name
    standard_names.update(standard_name)
    assert standard_name_data["name"] in standard_names.data
    assert (
        standard_names[standard_name_data["name"]]
        == ParseJson(github_response).standard_name
    )
    new_standard_names = StandardNameFile(standardnames)
    assert standard_names.data.whole_document() != standard_names_document
    assert (
        new_standard_names[standard_name_data["name"]]
        == ParseJson(github_response).standard_name
    )


def test_alias_update(standardnames):
    standard_names = StandardNameFile(standardnames)
    github_response = json.dumps(
        standard_name_data
        | {"name": "another_plasma_current", "alias": "plasma_current"}
    )
    standard_name = ParseJson(github_response).standard_name
    standard_names.update(standard_name)
    assert "another_plasma_current" in standard_names.data


def test_alias_update_error(standardnames):
    standard_names = StandardNameFile(standardnames)
    github_response = json.dumps(standard_name_data | {"alias": "undefined"})
    standard_name = ParseJson(github_response).standard_name
    with pytest.raises(KeyError):
        standard_names.update(standard_name)


def test_file_update_overwrite_error(standardnames):
    standard_names = StandardNameFile(standardnames)
    github_response = json.dumps(
        standard_name_data | {"name": "plasma_current", "overwrite": False}
    )
    standard_name = ParseJson(github_response).standard_name
    with pytest.raises(KeyError):
        standard_names.update(standard_name)


@pytest.mark.parametrize(
    "options", ["overwrite", "high priority", "overwrite,high priority"]
)
def test_json_overwrite(tmp_path, options):
    filename = tmp_path / "test.json"
    with open(filename, "w") as f:
        json.dump(standard_name_data | {"options": options}, f)
    standard_input = StandardInput(filename)
    assert standard_input.standard_name.overwrite is ("overwrite" in options)


def test_json_extra_priority_attr(tmp_path):
    filename = tmp_path / "test.json"
    with open(filename, "w") as f:
        json.dump(standard_name_data | {"options": "high priority"}, f)
    StandardInput(filename)


def test_json_roundtrip():
    assert (
        ParseJson(StandardName(**standard_name_data).as_json()).standard_name
        == ParseJson(json.dumps(standard_name_data)).standard_name
    )


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
