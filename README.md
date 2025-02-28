[![image](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://github.com/iterorganization/IMAS-Standard-Names)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/charliermarsh/ruff)
[![pytest](https://github.com/iterorganization/IMAS-Standard-Names/actions/workflows/python-package.yml/badge.svg)](https://github.com/iterorganization/IMAS-Standard-Names/actions)
[![cov](https://iterorganization.github.io/IMAS-Standard-Names/badges/coverage.svg)](https://github.com/iterorganization/IMAS-Standard-Names/actions) 

# IMAS Standard Names

This repository hosts a collection of Standard Names used in the Fusion
Conventions and logic for creating a static website for documentation.

## Generate documentation

```
python -m venv venv
. venv/bin/activate
pip install .[docs]
# Build the documentation
mkdocs build
# Or alternatively, serve the documentation:
# The site will automatically refresh when making changes
mkdocs serve
```
