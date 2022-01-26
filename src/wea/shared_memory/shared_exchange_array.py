"""
Wrapped Exchange Array implementation for shared memory
"""
import logging
import typing

# pylint: disable=W0201,W1202,W1203
from multiprocessing import shared_memory
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from ..interface import WrappedExchangeArray
from ..meta_data import _calculate_size, _write_header, check_buffer_array

LOGGER = logging.getLogger(__name__)


class SharedExchangeArray(WrappedExchangeArray):
    """
    Shared memory Wrapped Exchange Array

    :param np: WrappedExchangeArray type
    :type np: WrappedExchangeArray
    """

    def __new__(cls, name: str, create: bool, **kwargs):
        if create is True:
            kwarg = ["dtype", "shape"]
            for x_val in kwarg:
                if x_val not in kwargs:
                    raise TypeError(f"Missing {x_val} for creating wrapped array")
            shm, off = _create_shared_array(name, kwargs["dtype"], kwargs["shape"])
        else:
            kwarg = ["dtype", "shape"]
            for x_val in kwarg:
                if x_val in kwargs:
                    raise TypeError(
                        f"Ignoring {x_val}. Is not necessary for attaching to"
                        f"wrapped array"
                    )
            shm, off, pytype, dims = _attach_shared_array(name)
            for x_val, y_val in zip(kwarg, [pytype, dims]):
                kwargs[x_val] = y_val
        kwargs["buffer"] = shm.buf[off:]
        kwargs["order"] = "F"
        obj = super(SharedExchangeArray, cls).__new__(cls, **kwargs)
        obj._mem = shm
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._mem: SharedMemory = getattr(obj, "_mem", None)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, trace):
        self._close("close")

    @property
    def mem(self):
        """
        Return shared memory handle

        :return: shared memory handle
        :rtype: multiprocessing.shared_memory.SharedMemory
        """
        return self._mem

    def reopen(self):
        """
        Reopen a shared memory segement

        :raises FileNotFoundError: If shared memory segment was deleted
        """
        if self._mem is not None:
            shm, off, _, _ = _attach_shared_array(self.mem.name)
            self._mem, self.data = shm, shm.buf[off:]
        else:
            raise FileNotFoundError("No shared memory element set for connecting")

    def close(self) -> None:
        """
        Close shared memory segment
        """
        self._close("close")

    def unlink(self) -> None:
        """
        Unlink shared memory segment
        """
        self._close("unlink")

    def _close(self, action: str) -> None:
        """
        Protected wrapper function for closing or unlinking shared memory

        :param action: attribute function
        :type action: str
        """
        func = getattr(self._mem, action)
        func()


def create_shared_array(name: str, dtype: np.dtype, shape: tuple):
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
    return SharedExchangeArray(name, True, dtype=dtype, shape=shape)


def attach_shared_array(name: str):
    """
    Attach to an existing WrappedExchangeArray in shared memory

    :param name: Shared memory location
    :type name: str
    :return: Returns a WrappedArray instance
    :rtype: WrappedArray
    """
    return SharedExchangeArray(name, False)


def _create_shared_array(name: str, dtype: np.dtype, shape: tuple) -> typing.Tuple:
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
    LOGGER.debug(f"Creating shared memory segment: {name}")
    shm = shared_memory.SharedMemory(name=name, create=True, size=size)
    off = _write_header(shm.buf, dtype, shape)
    return shm, off


def _attach_shared_array(name: str) -> typing.Tuple:
    """
    Attach to an existing WrappedExchangeArray

    :param name: Shared memory location
    :type name: str
    :return: Shared memory segment, buffer offset, dtype and shape
    :rtype: Tuple
    """
    shm = shared_memory.SharedMemory(name=name, create=False)
    return shm, *check_buffer_array(shm.buf)
