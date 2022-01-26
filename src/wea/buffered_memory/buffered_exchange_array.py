"""
Wrapped Exchange Array implementation with
buffer memory
"""
# pylint: disable=W0201,W1202,W1203
import logging
import typing

import numpy as np

from ..interface import WrappedExchangeArray
from ..meta_data import _calculate_size, _write_header, check_buffer_array

LOGGER = logging.getLogger(__name__)


class BufferedExchangeArray(WrappedExchangeArray):
    """
    Buffered memory Wrapped Exchange Array

    :param WrappedExchangeArray: WrappedExchangeArray type
    :type WrappedExchangeArray: WrappedExchangeArray
    """

    def __new__(cls, **kwargs):
        kwarg = ["dtype", "shape"]
        if "exchange_buffer" in kwargs:
            buffer = kwargs["exchange_buffer"]
            size = len(buffer)
            off, pytype, dims = _load_buffered_array(buffer)
            for x_val, y_val in zip(kwarg, [pytype, dims]):
                kwargs[x_val] = y_val
            del kwargs["exchange_buffer"]
        else:
            for x_val in kwarg:
                if x_val not in kwargs:
                    raise TypeError(f"Missing {x_val} for creating wrapped array")
                buffer, off, size = _create_buffered_array(
                    kwargs["dtype"], kwargs["shape"]
                )
        kwargs["buffer"] = buffer[off:]
        kwargs["order"] = "F"
        obj = super(BufferedExchangeArray, cls).__new__(cls, **kwargs)
        obj._exchange_buffer_offset = off
        obj._exchange_buffer_size = size
        obj._exchange_buffer_header = buffer[:off]
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._exchange_buffer_header: bytearray = getattr(
            obj, "_exchange_buffer_header", None
        )
        self._exchange_buffer_offset: int = getattr(
            obj, "_exchange_buffer_offset", None
        )
        self._exchange_buffer_size: int = getattr(obj, "_exchange_buffer_size", None)

    @property
    def exchange_buffer(self) -> bytearray:
        """
        Exchange bytearray which contains also the header information

        :return: Array data with meta information
        :rtype: bytearray
        """
        buf = bytearray(self._exchange_buffer_size)
        buf[: self._exchange_buffer_offset] = self._exchange_buffer_header
        buf[self._exchange_buffer_offset :] = self.tobytes(order="F")
        return buf


def create_buffered_array(dtype: np.dtype, shape: tuple) -> BufferedExchangeArray:
    """
    Create a new BufferedExchangeArray

    :return: WrappedExchangeArray instance
    :rtype: BufferedExchangeArray
    """
    return BufferedExchangeArray(dtype=dtype, shape=shape)


def load_buffered_array(
    buf: typing.Union[memoryview, bytearray]
) -> BufferedExchangeArray:
    """
    Load a BufferedExchangeArray from a exchange bytes buffer

    :return: WrappedExchangeArray instance
    :rtype: BufferedExchangeArray
    """
    return BufferedExchangeArray(exchange_buffer=buf)


def _create_buffered_array(dtype: np.dtype, shape: tuple):
    """
    Create a new exchange buffer for the BufferedExchangeArray

    :param dtype: Data format
    :type type: np.dtype
    :param shape: Array dimension
    :type shape: tuple
    :return: exchange buffer header, buffer offset and size
    :rtype: Tuple
    """
    size, _, _ = _calculate_size(shape, dtype)
    LOGGER.debug(f"Creating bytes buffer with size {size}")
    buf = bytearray(size)
    off = _write_header(buf, dtype, shape)
    return buf, off, size


def _load_buffered_array(buf: typing.Union[memoryview, bytearray]):
    """
    Extract meta data from an exchange buffer

    :param buf: Exchange buffer
    :type buf: typing.Union[memoryview, bytearray]
    :raises MemoryError: If buffer is smaller than expected
    :raises TypeError: If Julia magic number is not inside
    :raises TypeError: The dtype does not fit
    :raises TypeError: If Complex32 is provided by Julia
    :return: Offset, dtype and dimesions
    :rtype: Tuple
    """
    return check_buffer_array(buf)
