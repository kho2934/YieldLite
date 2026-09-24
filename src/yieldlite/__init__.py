"""yieldlite: crop-yield prediction for small, low-data farm datasets."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("yieldlite")
except PackageNotFoundError:  # source checkout that was never installed
    __version__ = "0.0.0"

__all__ = ["__version__"]
