import unittest

from wea.utils import checkdims, roundup


class TestUtils(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestUtils, self).__init__(*args, **kwargs)
        self._server = None

    def setUp(self) -> None:
        super(TestUtils, self).setUp()
        pass

    def tearDown(self) -> None:
        super(TestUtils, self).tearDown()
        pass

    def test_checkdims(self):
        count = checkdims((5, 2))
        self.assertEqual(count, 10)
        count = checkdims((5, 2, 3))
        self.assertEqual(count, 30)

    def test_roundup(self):
        round = roundup(10, 20)
        self.assertEqual(round, 40)
        round = roundup(11, 20)
        self.assertEqual(round, 40)
        round = roundup(21, 20)
        self.assertEqual(round, 40)
        round = roundup(30, 20)
        self.assertEqual(round, 60)
        round = roundup(50, 20)
        self.assertEqual(round, 80)


if __name__ == "__main__":
    unittest.main()
