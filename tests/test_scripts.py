from contextlib import contextmanager

from click.testing import CliRunner
import json
from pathlib import Path
import pytest
import strictyaml as syaml

from imas_standard_names.scripts import has_standardname, update_standardnames
from tests.test_standard_name import standard_name_data as github_input
from tests.test_standard_name import yaml_multi as standardnames


@contextmanager
def launch_cli(
    standardnames: syaml.representation.YAML,
    github_input: dict[str, str],
    path: str | Path,
):
    """Lanuch CLI to update a temporary standard names file with input data."""
    with (
        click_runner(path) as (runner, temp_dir),
        write_standardnames(standardnames, temp_dir) as standardnames_file,
        write_submission(github_input, temp_dir) as submission_file,
    ):
        yield runner, (standardnames_file, submission_file)


@contextmanager
def click_runner(path: str | Path):
    """Lanuch click runner within isolated filesystem."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=path) as temp_dir:
        yield runner, temp_dir


@contextmanager
def write_standardnames(standardnames: syaml.representation.YAML, temp_dir):
    """Write yaml standardnames to a temporary file."""
    standardnames_file = Path(temp_dir) / "standardnames.yml"
    with open(standardnames_file, "w") as f:
        f.write(standardnames.as_yaml())
    yield standardnames_file.as_posix()


@contextmanager
def write_submission(github_input: dict[str, str], temp_dir):
    """Write json submission to a temporary file."""
    submission_file = Path(temp_dir) / "submission.json"
    with open(submission_file, "w") as f:
        f.write(json.dumps(github_input))
    yield submission_file.as_posix()


def test_add_standard_name(tmp_path):
    assert github_input["overwrite"] is False
    with launch_cli(standardnames, github_input, tmp_path) as (runner, args):
        result = runner.invoke(update_standardnames, args)
    assert result.exit_code == 0
    assert result.output.split(" ")[0] == f"{github_input['name']}"


def test_overwrite(tmp_path):
    _github_input = github_input.copy()
    _github_input["name"] = "plasma_current"
    _github_input["overwrite"] = True
    with launch_cli(standardnames, _github_input, tmp_path) as (runner, args):
        result = runner.invoke(update_standardnames, args)
    assert result.exit_code == 0
    assert result.output.split(" ")[0] == "plasma_current"


def test_overwrite_error(tmp_path):
    _github_input = github_input.copy()
    _github_input["name"] = "plasma_current"
    with launch_cli(standardnames, _github_input, tmp_path) as (runner, args):
        result = runner.invoke(update_standardnames, args)
    assert result.exit_code == 1


def test_is_standardname(tmp_path):
    with (
        click_runner(tmp_path) as (runner, temp_dir),
        write_standardnames(standardnames, temp_dir) as standardnames_file,
    ):
        result = runner.invoke(has_standardname, (standardnames_file, "plasma_current"))
    assert result.exit_code == 0
    assert result.output == "True\n"


def test_is_not_standardname(tmp_path):
    with (
        click_runner(tmp_path) as (runner, temp_dir),
        write_standardnames(standardnames, temp_dir) as standardnames_file,
    ):
        result = runner.invoke(has_standardname, (standardnames_file, "PlasmaCurrent"))
    assert result.exit_code == 0
    assert result.output == "False\n"

if __name__ == "__main__":
    pytest.main([__file__])
