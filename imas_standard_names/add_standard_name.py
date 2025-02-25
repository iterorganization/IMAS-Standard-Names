import click 

from imas_standard_names.parse import StandardInput, StandardNameFile

@click.command()
def add_standard_name(filename: str = 'submission.json'):
    """Add a standard name to the project's standard name file."""
    standard_name = StandardInput(filename).standard_name
    standard_names = StandardNameFile('../_standardnames.yml')
    standard_names.update(standard_name)

    print(standard_name)

if __name__ == '__main__':

    add_standard_name('../imas_standard_names/test.json')