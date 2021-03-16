from multiprocessing import shared_memory
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import logging
from ..meta_data import _calculate_size, _write_header, _read_header, \
    _julia_wa_header_sizeof, _julia_wa_magic, _julia_wa_etypes

_logger = logging.getLogger(__name__)


class WrappedExchangeArray(np.ndarray):
    def __new__(cls, name: str, create: bool, **kwargs):
        if create is True:
            kwarg = ['dtype', 'shape']
            for x in kwarg:
                if x not in kwargs:
                    raise TypeError(f'Missing {x} for creating wrapped array')
            shm, off = _create_shared_array(
                name, kwargs['dtype'], kwargs['shape'])
        else:
            kwarg = ['dtype', 'shape']
            for x in kwarg:
                if x in kwargs:
                    raise TypeError(
                        f'Ignoring {x}. Is not necessary for attaching to'
                        f'wrapped array')
            shm, off, type, dims = _attach_shared_array(name)
            for x, y in zip(kwarg, [type, dims]):
                kwargs[x] = y
        kwargs['buffer'] = shm.buf[off:]
        kwargs['order'] = 'F'
        obj = super(WrappedExchangeArray, cls).__new__(cls, **kwargs)
        obj._mem = shm
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._mem: SharedMemory = getattr(obj, '_mem', None)

    def __enter__(self):
        return self

    def __exit__(self):
        self._close('close')

    @property
    def mem(self):
        return self._mem

    def reopen(self):
        if self._mem is not None:
            shm, off, _, _ = _attach_shared_array(self.mem.name)
            self._mem, self.data = shm, shm.buf[off:]
        else:
            raise FileNotFoundError(
                'No shared memory element set for connecting')

    def close(self) -> None:
        self._close('close')

    def unlink(self) -> None:
        self._close('unlink')

    def _close(self, f: str) -> None:
        self._arr = None
        func = getattr(self._mem, f)
        func()


def create_shared_array(name: str, dtype: np.dtype,
                        shape: tuple):
    """
    Create a new WrappedExchangeArray in shared memory

    :param name: Shared memory location
    :type name: str
    :param dtype: Data format
    :type dtype: np.dtype
    :param shape: Array dimension
    :type shape: tuple
    :return: Returns a WrappedArray instance
    :rtype: WrappedArray
    """
    return WrappedExchangeArray(name, True, dtype=dtype, shape=shape)


def attach_shared_array(name: str):
    """
    Attach to an existing WrappedExchangeArray in shared memory

    :param name: Shared memory location
    :type name: str
    :return: Returns a WrappedArray instance
    :rtype: WrappedArray
    """
    return WrappedExchangeArray(name, False)


def _create_shared_array(name: str, dtype: np.dtype,
                         shape: tuple):
    """
    Create a new WrappedArray in shared memory

    :param name: Shared memory location
    :type name: str
    :param dtype: Data format
    :type type: np.dtype
    :param shape: Array dimension
    :type shape: tuple
    :return: Shared memory segment and buffer offset
    :rtype: Tuple
    """
    size, _, _ = _calculate_size(shape, dtype)
    _logger.info(f'Creating shared memory segment: {name}')
    shm = shared_memory.SharedMemory(name=name, create=True,
                                     size=size)
    off = _write_header(shm.buf, dtype, shape)
    return shm, off


def _attach_shared_array(name: str):
    """
    Attach to an existing WrappedExchangeArray

    :param name: Shared memory location
    :type name: str
    :return: Shared memory segment, buffer offset, dtype and shape
    :rtype: Tuple
    """
    shm = shared_memory.SharedMemory(name=name, create=False)
    if shm.size < _julia_wa_header_sizeof:
        raise MemoryError("Shared memory is smaller than header size")
    magic, eltype, _, off, dims = _read_header(shm.buf)
    if magic != _julia_wa_magic:
        raise TypeError(f'WrappedArray version {magic} not supported')
    if eltype > len(_julia_wa_etypes):
        raise TypeError("Provided eltype not found in supported list")
    if eltype == 11:
        raise TypeError("Complex32 is not supported by numpy")
    type = _julia_wa_etypes[eltype-1]
    return shm, off, type, dims
