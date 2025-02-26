from click.testing import CliRunner
import json
from pathlib import Path
import pytest
import strictyaml as syaml

from imas_standard_names.scripts import update_standardnames
from tests.test_standard_name import standard_name_data as github_input
from tests.test_standard_name import yaml_multi as standardnames


def run_update(
    standardnames: syaml.representation.YAML,
    github_input: dict[str, str],
    path: str | Path,
):
    """Lanuch CLI to update a temporary standard names file with input data."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=path) as temp_dir:
        standardnames_file = Path(temp_dir) / "standardnames.yml"
        submission_file = Path(temp_dir) / "submission.json"

        with open(standardnames_file, "w") as f:
            f.write(standardnames.as_yaml())

        with open(submission_file, "w") as f:
            f.write(json.dumps(github_input))

        result = runner.invoke(
            update_standardnames,
            [standardnames_file.as_posix(), submission_file.as_posix()],
        )
        return result


def test_add_standard_name(tmp_path):
    assert github_input["overwrite"] is False
    result = run_update(standardnames, github_input, tmp_path)
    assert result.exit_code == 0
    assert result.output.split(" ")[0] == f"{github_input['name']}"

def test_overwrite(tmp_path):
    _github_input = github_input.copy()
    _github_input["name"] = "plasma_current"
    _github_input["overwrite"] = True
    result = run_update(standardnames, _github_input, tmp_path)
    assert result.exit_code == 0
    assert result.output.split(" ")[0] == f"{github_input['name']}"

def test_overwrite_error(tmp_path):
    _github_input = github_input.copy()
    _github_input["name"] = "plasma_current"
    result = run_update(standardnames, _github_input, tmp_path)
    assert result.exit_code == 1


if __name__ == "__main__":
    pytest.main([__file__])
