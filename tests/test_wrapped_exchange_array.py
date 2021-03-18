from multiprocessing import shared_memory
from typing import Tuple
import unittest
import struct
import numpy as np
import logging
import pytest
import src.wea.meta_data as meta
from parameterized import parameterized
from src.wea.utils import checkdims
from src.wea.shared_memory import SharedExchangeArray, create_shared_array, \
    attach_shared_array


logger = logging.getLogger(__name__)


def create_buffer() -> Tuple[bytearray, tuple, int, int, np.dtype, bytes]:
    """
    Create default data

    :return: Returns a buffer, a dimension tuple, dimension elements, \
        header offset, numpy type and a default header
    :rtype: Tuple[bytearray, tuple, int, int, np.dtype, bytes]
    """
    type = np.dtype('float64')
    dims = (5, 2)
    N = len(dims)
    num = checkdims(dims)
    off = meta._wrapped_exchange_array_header_size(len(dims))
    buf = bytearray(int(off) + int(num * type.itemsize))
    header = struct.pack(f'{meta._JULIA_WA_HEADER_FORMAT}qq',
                         meta._JULIA_WA_MAGIC, np.uint16(10),
                         np.uint16(N), np.int64(128), np.int64(dims[0]),
                         np.int64(dims[1]))
    return buf, dims, N, off, type, header


class TestWrappedArray(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestWrappedArray, self).__init__(*args, **kwargs)
        self._wa: SharedExchangeArray = None
        self._shm_name = '/test-awesome-1'

    def setUp(self) -> None:
        super(TestWrappedArray, self).setUp()
        pass

    def tearDown(self) -> None:
        super(TestWrappedArray, self).tearDown()
        try:
            shm = shared_memory.SharedMemory(self._shm_name, create=False)
            shm.unlink()
        except FileNotFoundError:
            logger.info('Nothing to tear down')

    def test_calculate_size(self):
        dims = (5, 2)
        type = np.dtype('float64')
        size, off, N = meta._calculate_size(dims, type)
        self.assertEqual(size, 208)
        self.assertEqual(off, 128)
        self.assertEqual(N, len(dims))

    def test_wrapped_array_header_size(self):
        dims = (5, 2)
        N = len(dims)
        off = meta._wrapped_exchange_array_header_size(N)
        self.assertEqual(off, 2 * meta._JULIA_WA_AGLIGN)

    def test_write_header(self):
        buf, dims, _, _, type, exp_header = create_buffer()
        meta._write_header(buf, type, dims)
        header = bytes(buf[:meta._JULIA_WA_HEADER_SIZEOF])
        self.assertEqual(header, exp_header[:meta._JULIA_WA_HEADER_SIZEOF])

    def test_read_header(self):
        buf, exp_dims, exp_N, exp_off, _, exp_header = create_buffer()
        buf[:meta._JULIA_WA_HEADER_SIZEOF] = exp_header
        magic, eltype, N, off, dims = meta._read_header(buf)
        self.assertEqual(magic, meta._JULIA_WA_MAGIC)
        self.assertEqual(eltype, 10)
        self.assertEqual(N, exp_N)
        self.assertEqual(off, exp_off)
        self.assertEqual(dims, exp_dims)

    @parameterized.expand([
        ((10, 2),),
        ((10, 1),)
    ])
    def test_create_shared_array(self, shape):
        data = np.random.random_sample(shape)
        self._wa = create_shared_array(
            self._shm_name, data.dtype, data.shape)
        self._wa[:] = data[:]
        shm = shared_memory.SharedMemory(self._shm_name, create=False)
        magic, eltype, N, off, dims = meta._read_header(shm.buf)
        a = np.ndarray(shape=data.shape, dtype=data.dtype,
                       buffer=shm.buf[off:], order='F')
        compare = self._wa[:] == a[:]
        self.assertTrue(compare.all())
        self.assertEqual(magic, meta._JULIA_WA_MAGIC)
        self.assertEqual(eltype, 10)
        self.assertEqual(N, len(data.shape))
        self.assertEqual(dims, data.shape)
        self.assertEqual(off, 128)

    @parameterized.expand([
        ((10, 2),),
        ((10, 1),)
    ])
    def test_attach_shared_array(self, shape):
        data = np.random.random_sample(shape)
        size, _, _ = meta._calculate_size(data.shape, data.dtype)
        shm = shared_memory.SharedMemory(
            self._shm_name, create=True, size=size)
        off = meta._write_header(shm.buf, data.dtype, data.shape)
        a = np.ndarray(shape=data.shape, dtype=data.dtype,
                       buffer=shm.buf[off:], order='F')
        a[:] = data[:]
        self._wa = attach_shared_array(self._shm_name)
        compare = self._wa[:] == a[:]
        self.assertTrue(compare.all())
        self.assertEqual(off, 128)

    def test_attach_shared_array_with_complex32(self):
        data = np.random.randn(10, 2)
        size, _, _ = meta._calculate_size(data.shape, data.dtype)
        shm = shared_memory.SharedMemory(
            self._shm_name, create=True, size=size)
        off = meta._write_header(shm.buf, data.dtype, data.shape)
        struct.pack_into('H', shm.buf[4:6], 0, np.uint16(11))
        a = np.ndarray(shape=data.shape, dtype=data.dtype,
                       buffer=shm.buf[off:], order='F')
        a[:] = data[:]
        with pytest.raises(TypeError):
            self._wa = attach_shared_array(self._shm_name)

    def test_WrappedArray_attributes(self):
        data = np.random.randn(10, 2)
        self._wa = create_shared_array(
            self._shm_name, data.dtype, data.shape)
        self._wa[:] = data[:]
        compare = self._wa[:] == data[:]
        self.assertTrue(compare.all())
        self.assertEqual(type(self._wa.mem), shared_memory.SharedMemory)


if __name__ == '__main__':
    unittest.main()
