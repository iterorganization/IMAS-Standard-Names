from contextlib import contextmanager

from click.testing import CliRunner
import json
from pathlib import Path
import pytest
import strictyaml as syaml

from imas_standard_names.scripts import has_standardname, update_standardnames


github_input = {
    "name": "ion_temperature",
    "units": "A",
    "documentation": "multi-line\ndoc string",
    "tags": "",
    "alias": "",
    "overwrite": False,
}

standardnames = syaml.as_document(
    {
        name: {"units": units, "documentation": "docs"}
        for name, units in zip(
            ["plasma_current", "plasma_current_density", "electron_temperature"],
            ["A", "A/m^2", "eV"],
        )
    }
)


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
    assert f"**{github_input['name']}** appended to" in result.output


def test_overwrite(tmp_path):
    _github_input = github_input.copy()
    _github_input["name"] = "plasma_current"
    _github_input["options"] = "overwrite"
    with launch_cli(standardnames, _github_input, tmp_path) as (runner, args):
        result = runner.invoke(update_standardnames, args)
    assert "**plasma_current** appended to" in result.output


def test_overwrite_error(tmp_path):
    _github_input = github_input.copy()
    _github_input["name"] = "plasma_current"
    with launch_cli(standardnames, _github_input, tmp_path) as (runner, args):
        result = runner.invoke(update_standardnames, args)
    assert "Error" in result.output
    assert "**plasma_current** is already present" in result.output


def test_standard_name_error(tmp_path):
    _github_input = github_input.copy()
    _github_input["name"] = "1st_plasma_current"
    with launch_cli(standardnames, _github_input, tmp_path) as (runner, args):
        result = runner.invoke(update_standardnames, args)
    assert "Error" in result.output
    assert f"**{_github_input['name']}** is not a valid" in result.output


def test_standard_name_alias(tmp_path):
    _github_input = github_input.copy()
    _github_input |= {"name": "second_plasma_current", "alias": "plasma_current"}
    with launch_cli(standardnames, _github_input, tmp_path) as (runner, args):
        result = runner.invoke(update_standardnames, args)
    assert "Success" in result.output
    assert f"**{_github_input['name']}** appended to" in result.output in result.output


def test_standard_name_alias_error(tmp_path):
    _github_input = github_input.copy()
    _github_input |= {"name": "second_plasma_current", "alias": "1st_plasma_current"}
    with launch_cli(standardnames, _github_input, tmp_path) as (runner, args):
        result = runner.invoke(update_standardnames, args)
    assert "Error" in result.output
    assert f"**{_github_input['alias']}** is not present" in result.output


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


def test_standardname_whitespace(tmp_path):
    with (
        click_runner(tmp_path) as (runner, temp_dir),
        write_standardnames(standardnames, temp_dir) as standardnames_file,
    ):
        result = runner.invoke(has_standardname, (standardnames_file, "Plasma Current"))
    assert result.exit_code == 0
    assert result.output == "False\n"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
