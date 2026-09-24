"""Smoke test: the package is installed and importable."""

import yieldlite


def test_package_is_installed():
    assert yieldlite.__version__ != "0.0.0", "not installed: run pip install -e ."
