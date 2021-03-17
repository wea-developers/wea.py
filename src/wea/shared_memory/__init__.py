"""
Shared Memory Wrapped Exchange Array
"""
from .wrapped_exchange_array import WrappedExchangeArray,\
    create_shared_array, attach_shared_array

__all__ = ['WrappedExchangeArray',
           'create_shared_array', 'attach_shared_array']
