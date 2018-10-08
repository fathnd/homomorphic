from common import TestCase, run_tests, TEST_NUMPY

import torch
import unittest
import warnings
from torch import tensor


if TEST_NUMPY:
    import numpy as np


class TestDTypeInfo(TestCase):

    def test_invalid_input(self):
        for type in [torch.float32, torch.float64]:
            with self.assertRaises(TypeError):
                xinfo = torch.iinfo(type)

        for type in [torch.int64, torch.int32, torch.int16, torch.uint8]:
            with self.assertRaises(TypeError):
                xinfo = torch.finfo(type)

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_common_info(self):
        for type in [torch.float32, torch.float64,
                     torch.int64, torch.int32, torch.int16, torch.uint8]:
            x = torch.zeros((2, 2), dtype=type)
            xinfo = torch.finfo(x.dtype) if x.dtype.is_floating_point else torch.iinfo(x.dtype)
            xn = x.cpu().numpy()
            xninfo = np.finfo(xn.dtype) if x.dtype.is_floating_point else np.iinfo(xn.dtype)
            self.assertEqual(xinfo.bits, xninfo.bits)
            self.assertEqual(xinfo.max, xninfo.max)
            if x.dtype.is_floating_point:
                self.assertEqual(xinfo.eps, xninfo.eps)

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_finfo(self):
        for type in [torch.float32, torch.float64]:
            x = torch.zeros((2, 2), dtype=type)
            xinfo = torch.finfo(x.dtype)
            xn = x.cpu().numpy()
            xninfo = np.finfo(xn.dtype)
            self.assertEqual(xinfo.eps, xninfo.eps)


if __name__ == '__main__':
    run_tests()
