import unittest

import torch

from int4 import pack_int4, quantize, unpack_int4


class Int4Tests(unittest.TestCase):
    def test_finite_float64_outside_fp32_range_is_rejected(self):
        with self.assertRaises(ValueError):
            quantize(torch.tensor([[1e100, 1.0]], dtype=torch.float64), 2)

    def test_known_byte_layout_and_round_trip(self):
        values = torch.tensor([-7, 0, 7, -1], dtype=torch.int8)
        packed = pack_int4(values)
        self.assertEqual(packed.tolist(), [0x81, 0x7F])
        torch.testing.assert_close(unpack_int4(packed), values)

    def test_tail_padding_does_not_change_shape_or_row_scale(self):
        weight = torch.tensor([[7.0, 0.0, -7.0], [1.0, 2.0, 7.0]])
        result = quantize(weight, 2)
        self.assertEqual(tuple(result.packed.shape), (2, 2))
        self.assertEqual(tuple(result.dequantize().shape), (2, 3))
        self.assertAlmostEqual(result.dequantize()[0, 2].item(), -7.0)
        self.assertAlmostEqual(result.dequantize()[1, 2].item(), 7.0)
        self.assertEqual(result.payload_bytes, 20)

    def test_zero_groups_use_unit_scale(self):
        result = quantize(torch.zeros(2, 32), 32)
        self.assertEqual(result.scales.tolist(), [[1.0], [1.0]])
        self.assertEqual(result.dequantize().count_nonzero().item(), 0)

    def test_smaller_groups_isolate_the_outlier(self):
        weight = torch.tensor([[70.0, 0.0, 1.2, 2.0]])
        grouped = quantize(weight, 2).dequantize()
        global_quant = quantize(weight, 4).dequantize()
        self.assertAlmostEqual(grouped[0, 2].item(), 8 / 7, places=6)
        self.assertEqual(global_quant[0, 2].item(), 0.0)
