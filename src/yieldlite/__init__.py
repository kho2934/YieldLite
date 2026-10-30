"""Small-data rice-yield prediction with clear uncertainty and refusals."""

from importlib.metadata import (
    PackageNotFoundError,
    version,
)

from yieldlite.result import PredictionResult
from yieldlite.yieldmodel import YieldModel

try:
    __version__ = version("yieldlite")
except PackageNotFoundError:
    # This happens when the source folder is used without installing the
    # package first.
    __version__ = "0.0.0"

__all__ = [
    "PredictionResult",
    "YieldModel",
    "__version__",
]
