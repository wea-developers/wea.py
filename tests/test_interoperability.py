import logging
import unittest
from multiprocessing import shared_memory

import numpy as np
import pytest
from parameterized import parameterized

from wea.shared_memory import (
    SharedExchangeArray,
    attach_shared_array,
    create_shared_array,
)


@pytest.fixture(scope="class")
def julia_handle(request, julia):
    try:
        julia.using("InterProcessCommunication")
    except julia.core.JuliaError:
        julia.using("Pkg")
        julia.eval('Pkg.add("InterProcessCommunication")')
        julia.using("InterProcessCommunication")
    try:
        julia.using("Random")
    except julia.core.JuliaError:
        julia.using("Pkg")
        julia.eval('Pkg.add("Random")')
        julia.using("Random")
    request.cls.julia = julia


logger = logging.getLogger(__name__)


@pytest.mark.usefixtures("julia_handle")
class TestWrappedExchangeArray(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestWrappedExchangeArray, self).__init__(*args, **kwargs)
        self._wa: SharedExchangeArray = None
        self._shm_name = "/test-awesome-1"
        self._dims = (5, 2)

    def setUp(self) -> None:
        super(TestWrappedExchangeArray, self).setUp()
        pass

    def tearDown(self) -> None:
        super(TestWrappedExchangeArray, self).tearDown()
        try:
            shm = shared_memory.SharedMemory(self._shm_name, create=False)
            shm.unlink()
        except FileNotFoundError:
            logger.info("Nothing to tear down")

    @parameterized.expand([["Float64"]])
    def test_create_shared_array(self, T: str):
        self.skipTest(
            "Does not work at the moment. Creates StackOverflowError in Julia"
        )
        type = np.dtype(T.lower())
        self._wa = create_shared_array(self._shm_name, type, self._dims)
        self.julia.eval(f'wa = WrappedArray("{self._shm_name}"; ' f"readonly=false);")
        data = self.julia.eval(f"wa[:] = randn({T}, ({self._dims[0]},{self._dims[1]}))")
        self.julia.eval("dump(wa)")
        compare = self._wa[:] == data[:]
        self.assertTrue(compare.all())

    @parameterized.expand([["Float64"]])
    def test_attach_shared_array(self, T: str):
        self.julia.eval(
            f'wa = WrappedArray("{self._shm_name}", '
            f"{T}, ({self._dims[0]},{self._dims[1]}));"
        )
        data = self.julia.eval(f"wa[:] = randn({T}, ({self._dims[0]},{self._dims[1]}))")
        self._wa = attach_shared_array(self._shm_name)
        compare = self._wa[:] == data[:]
        self.assertTrue(compare.all())


if __name__ == "__main__":
    unittest.main()
