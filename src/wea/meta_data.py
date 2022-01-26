"""
Copyright(c) 2016: Éric Thiébaut https: // github.com/emmt

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files(the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and / or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
"""

import struct
from typing import Tuple, Union

import numpy as np

from .utils import checkdims, roundup

_JULIA_WA_MAGIC = np.uint32(0x57412D31)
_JULIA_WA_AGLIGN = 64
_JULIA_WA_TYPES = (
    (1, np.dtype("int8"), "signed 8-bit integer"),
    (2, np.dtype("uint8"), "unsigned 8-bit integer"),
    (3, np.dtype("int16"), "signed 16-bit integer"),
    (4, np.dtype("uint16"), "unsigned 16-bit integer"),
    (5, np.dtype("int32"), "signed 32-bit integer"),
    (6, np.dtype("uint32"), "unsigned 32-bit integer"),
    (7, np.dtype("int64"), "signed 64-bit integer"),
    (8, np.dtype("uint64"), "unsigned 64-bit integer"),
    (9, np.dtype("float32"), "32-bit floating-point"),
    (10, np.dtype("float64"), "64-bit floating-point"),
    (11, np.dtype("complex64"), "64-bit complex"),
    (12, np.dtype("complex128"), "128-bit complex"),
)
_JULIA_WA_IDENTS = {T: i for (i, T, _) in _JULIA_WA_TYPES}
_JULIA_WA_ELTYPES = [T for (i, T, str) in _JULIA_WA_TYPES]
_JULIA_WA_HEADER_FORMAT = "I2Hq"
_JULIA_WA_HEADER_SIZEOF = struct.calcsize(_JULIA_WA_HEADER_FORMAT)


def _write_header(buf: Union[memoryview, bytearray], dtype: np.dtype, shape: tuple):
    """
    Write the header data into the shared memory

    :param buf: Shared memory buffer
    :type buf: bytes
    :param dtype: Data format
    :type dtype: np.dtype
    :param shape: Array dimension
    :type shape: tuple
    :return: Offset to the start of the array
    :rtype: int
    """
    size, off, n_count = _calculate_size(shape, dtype)
    if dtype not in _JULIA_WA_IDENTS:
        raise TypeError(f"Type {dtype} is not supported for WrappedArray")
    eltype = _JULIA_WA_IDENTS[dtype]
    if len(buf) < size:
        raise MemoryError("Shared memory buffer is too small for wrapped array")
    struct.pack_into(
        _JULIA_WA_HEADER_FORMAT,
        buf,
        0,
        np.uint32(_JULIA_WA_MAGIC),
        np.uint16(eltype),
        np.uint16(n_count),
        np.int64(off),
    )
    for idx, val in enumerate(shape):
        struct.pack_into(
            "q",
            buf,
            int(_JULIA_WA_HEADER_SIZEOF + idx * np.dtype("int64").itemsize),
            np.int64(val),
        )
    return int(off)


def _read_header(buf: Union[memoryview, bytearray]):
    """
    Read the header data from the shared memory

    :param buf: Shared memory buffer
    :type buf: bytes
    :return: WrappedArray version, data type index, dimensions, Offset
     to the start of the array, Array shape
    :rtype: Tuple[int, int, int, int, tuple]
    """
    magic, eltype, n_count, off = struct.unpack_from(_JULIA_WA_HEADER_FORMAT, buf)
    dims = struct.unpack_from(
        "".join(["q" for _ in range(0, n_count, 1)]), buf, int(_JULIA_WA_HEADER_SIZEOF)
    )
    return magic, eltype, n_count, off, tuple(dims)


def _wrapped_exchange_array_header_size(n_count: int):
    """
    Calculate the header size

    :param n_count: Dimensions
    :type n_count: int
    :return: Up-rounded size
    :rtype: int
    """
    add = _JULIA_WA_HEADER_SIZEOF + n_count * np.dtype("int64").itemsize
    cld = roundup(add, _JULIA_WA_AGLIGN)
    return int(cld)


def _calculate_size(shape: tuple, dtype: np.dtype):
    """
    Calculate the overall shared memory size

    :param shape: Array dimensions
    :type shape: tuple
    :param dtype: Array data format
    :type dtype: np.dtype
    :return: Segment size, Start offset, Dimensions
    :rtype: Tuple[int, int, int]
    """
    n_count = len(shape)
    num = checkdims(shape)
    off = _wrapped_exchange_array_header_size(n_count)
    size = off + dtype.itemsize * num
    return int(size), int(off), int(n_count)


def check_buffer_array(buf: Union[memoryview, bytearray]) -> Tuple:
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
    if len(buf) < _JULIA_WA_HEADER_SIZEOF:
        raise MemoryError("Shared memory is smaller than header size")
    magic, eltype, _, off, dims = _read_header(buf)
    if magic != _JULIA_WA_MAGIC:
        raise TypeError(f"WrappedArray version {magic} not supported")
    if eltype > len(_JULIA_WA_ELTYPES):
        raise TypeError("Provided eltype not found in supported list")
    pytype = _JULIA_WA_ELTYPES[eltype - 1]
    return off, pytype, dims
