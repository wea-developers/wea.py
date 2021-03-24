"""
Shared Memory Wrapped Exchange Array
"""
from .shared_exchange_array import SharedExchangeArray,\
    create_shared_array, attach_shared_array

__all__ = ['SharedExchangeArray',
           'create_shared_array', 'attach_shared_array']
