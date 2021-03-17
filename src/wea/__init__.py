"""
Wrapped Exchange Array package
"""
from .shared_memory import WrappedExchangeArray,\
    create_shared_array, attach_shared_array

__all__ = ['__version__', 'WrappedExchangeArray',
           'create_shared_array', 'attach_shared_array']

try:
    from ._version import version as __version__
except ImportError:
    # broken installation, we don't even try
    # unknown only works because I do poor mans version compare
    __version__ = "unknown"
