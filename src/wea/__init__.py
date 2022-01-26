"""
Wrapped Exchange Array package
"""
from importlib.metadata import PackageNotFoundError, version

from .buffered_memory import (
    BufferedExchangeArray,
    create_buffered_array,
    load_buffered_array,
)
from .shared_memory import SharedExchangeArray, attach_shared_array, create_shared_array

__all__ = [
    "__version__",
    "SharedExchangeArray",
    "BufferedExchangeArray",
    "create_shared_array",
    "attach_shared_array",
    "create_buffered_array",
    "load_buffered_array",
]

try:
    __version__ = version("wea")
except PackageNotFoundError:
    # package is not installed
    pass
