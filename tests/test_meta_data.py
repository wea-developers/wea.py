import numpy as np
import pytest

import wea.meta_data as meta

from .helper import create_buffer


def test_calculate_size():
    dims = (5, 2)
    type = np.dtype("float64")
    size, off, N = meta._calculate_size(dims, type)
    assert size == 208
    assert off == 128
    assert N == len(dims)


def test_wrapped_array_header_size():
    dims = (5, 2)
    N = len(dims)
    off = meta._wrapped_exchange_array_header_size(N)
    assert off == 2 * meta._JULIA_WA_AGLIGN


def test_write_header():
    buf, dims, _, _, type, exp_header = create_buffer()
    meta._write_header(buf, type, dims)
    header = bytes(buf[: meta._JULIA_WA_HEADER_SIZEOF])
    assert header == exp_header[: meta._JULIA_WA_HEADER_SIZEOF]


def test_read_header():
    buf, exp_dims, exp_N, exp_off, _, exp_header = create_buffer()
    buf[: meta._JULIA_WA_HEADER_SIZEOF] = exp_header
    magic, eltype, N, off, dims = meta._read_header(buf)
    assert magic == meta._JULIA_WA_MAGIC
    assert eltype == 10
    assert N == exp_N
    assert off == exp_off
    assert dims == exp_dims


if __name__ == "__main__":
    pytest.main()
