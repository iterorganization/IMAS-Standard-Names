import click

from imas_standard_names.standard_name import StandardInput, StandardNameFile


@click.command()
@click.argument("standardnames_file")
@click.argument("submission_file")
def update_standardnames(standardnames_file: str, submission_file: str):
    """Add a standard name to the project's standard name file."""
    standardnames = StandardNameFile(standardnames_file)
    try:
        standard_name = StandardInput(submission_file).standard_name
        standardnames.update(standard_name)
    except (NameError, KeyError, Exception) as error:
        click.echo(error)
    else:
        click.echo(
            f"Success: **{standard_name.name}** appended to {standardnames.filename}."
        )


@click.command()
@click.argument("standardnames_file")
@click.argument("standard_name", nargs=-1)  # handle whitespace in standard name
def has_standardname(standardnames_file: str, standard_name: str):
    """Check if a standard name exists in the project's standard name file."""
    standardnames = StandardNameFile(standardnames_file)
    standard_name = " ".join(standard_name)
    click.echo(f"{standard_name in standardnames.data}")
