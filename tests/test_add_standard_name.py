
import pytest
from click.testing import CliRunner

from imas_standard_names.add_standard_name import add_standard_name

def test_add_standard_name():
    runner = CliRunner()
    result = runner.invoke(add_standard_name, ['Peter'])
    assert result.exit_code == 0
    assert result.output == 'Hello Peter!\n'

if __name__ == '__main__':
    test_add_standard_name()