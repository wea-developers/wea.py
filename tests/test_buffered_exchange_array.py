from src.wea import create_buffered_array, load_buffered_array
import src.wea.meta_data as meta
import numpy as np
import pytest
import struct


@pytest.mark.parametrize('shape', [
    (10, 2),
    (10, 1)
])
def test_create_buffered_array(shape):
    data = np.random.random_sample(shape)
    wa = create_buffered_array(data.dtype, data.shape)
    wa[:] = data[:]
    magic, eltype, N, off, dims = meta._read_header(wa.exchange_buffer)
    compare = wa[:] == data[:]
    assert compare.all()
    assert magic == meta._JULIA_WA_MAGIC
    assert eltype == 10
    assert N == len(data.shape)
    assert dims == data.shape
    assert off == 128
    assert wa.exchange_buffer[off:] == bytearray(data.data)


@pytest.mark.parametrize('shape', [
    (10, 2),
    (10, 1)
])
def test_load_buffered_array(shape):
    data = np.asfortranarray(np.random.random_sample(shape))
    size, _, _ = meta._calculate_size(data.shape, data.dtype)
    buf = bytearray(size)
    off = meta._write_header(buf, data.dtype, data.shape)
    buf[off:] = data.data
    wa = load_buffered_array(buf)
    compare = wa[:] == data[:]
    assert compare.all()
    assert off == 128
    assert wa.exchange_buffer[off:] == bytearray(data.data)


def test_BufferedExchangeArray_attributes():
    data = np.random.randn(10, 2)
    wa = create_buffered_array(data.dtype, data.shape)
    wa[:] = data[:]
    compare = wa[:] == data[:]
    assert compare.all()
    assert isinstance(wa.exchange_buffer, bytearray)
    assert wa.exchange_buffer[128:] == bytearray(data.tobytes())
