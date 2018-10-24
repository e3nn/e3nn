# pylint: disable=C,E1101,E1102
import unittest
from se3cnn.kernel import SE3Kernel

class Tests(unittest.TestCase):
    def test_kij_is_none(self):
        kernel = SE3Kernel([(1, 0)], [(1, 0), (1, 5)], 3) 
        kernel.forward()

unittest.main()
