"""
Wrapped Exchange Array package
"""
from .shared_memory import SharedExchangeArray,\
    create_shared_array, attach_shared_array
from .buffered_memory import BufferedExchangeArray, \
    create_buffered_array, load_buffered_array

__all__ = ['__version__', 'SharedExchangeArray',
           'create_shared_array', 'attach_shared_array',
           'create_buffered_array', 'load_buffered_array']

try:
    from ._version import version as __version__
except ImportError:
    # broken installation, we don't even try
    # unknown only works because I do poor mans version compare
    __version__ = "unknown"
