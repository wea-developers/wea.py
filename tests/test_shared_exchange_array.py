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

    @parameterized.expand([
        ((10, 2),),
        ((10, 1),)
    ])
    def test_full_loop(self, shape):
        data = np.random.random_sample(shape)
        self._wa = create_shared_array(
            self._shm_name, data.dtype, data.shape)
        self._wa[:] = data[:]
        wa = attach_shared_array(self._shm_name)
        compare = self._wa[:] == wa[:]
        self.assertTrue(compare.all())

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
