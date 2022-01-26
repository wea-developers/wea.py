import struct
from typing import Tuple

import numpy as np

import wea.meta_data as meta
from wea.utils import checkdims


def create_buffer() -> Tuple[bytearray, tuple, int, int, np.dtype, bytes]:
    """
    Create default data

    :return: Returns a buffer, a dimension tuple, dimension elements, \
        header offset, numpy type and a default header
    :rtype: Tuple[bytearray, tuple, int, int, np.dtype, bytes]
    """
    type = np.dtype("float64")
    dims = (5, 2)
    N = len(dims)
    num = checkdims(dims)
    off = meta._wrapped_exchange_array_header_size(len(dims))
    buf = bytearray(int(off) + int(num * type.itemsize))
    header = struct.pack(
        f"{meta._JULIA_WA_HEADER_FORMAT}qq",
        meta._JULIA_WA_MAGIC,
        np.uint16(10),
        np.uint16(N),
        np.int64(128),
        np.int64(dims[0]),
        np.int64(dims[1]),
    )
    return buf, dims, N, off, type, header
