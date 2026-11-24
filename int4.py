"""Grouped symmetric INT4 reference implementation for linear weights."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F


def pack_int4(values: torch.Tensor) -> torch.Tensor:
    """Use offset binary and put the even-indexed value in the low nibble."""
    if values.dtype != torch.int8 or values.ndim < 1 or values.shape[-1] % 2:
        raise ValueError('Expected int8 values with an even final dimension')
    if torch.any((values < -8) | (values > 7)):
        raise ValueError('INT4 values must be in [-8, 7]')
    unsigned = (values + 8).to(torch.uint8)
    return unsigned[..., 0::2] | (unsigned[..., 1::2] << 4)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    if packed.dtype != torch.uint8 or packed.ndim < 1:
        raise ValueError('Expected a tensor of uint8 packed bytes')
    lanes = torch.stack((packed & 0x0F, packed >> 4), dim=-1)
    return lanes.flatten(start_dim=-2).to(torch.int8) - 8


@dataclass(frozen=True)
class QuantizedWeight:
    packed: torch.Tensor
    scales: torch.Tensor
    original_width: int
    group_size: int

    @property
    def payload_bytes(self) -> int:
        return self.packed.nbytes + self.scales.nbytes

    def dequantize(self) -> torch.Tensor:
        rows = self.packed.shape[0]
        integers = unpack_int4(self.packed).reshape(rows, -1, self.group_size)
        weights = integers.to(torch.float32) * self.scales.unsqueeze(-1)
        return weights.flatten(start_dim=1)[:, :self.original_width]


def quantize(weight: torch.Tensor, group_size: int) -> QuantizedWeight:
    if weight.ndim != 2 or min(weight.shape) == 0 or not weight.is_floating_point():
        raise ValueError('Expected a non-empty 2D floating-point weight')
    if group_size <= 0 or group_size % 2:
        raise ValueError('Group size must be a positive even integer')
    weight = weight.float()
    if not torch.isfinite(weight).all():
        raise ValueError('Weights must be finite in FP32')
    rows, width = weight.shape
    padding = (-width) % group_size
    grouped = F.pad(weight, (0, padding)).reshape(rows, -1, group_size)
    maximum = grouped.abs().amax(dim=-1)
    scales = torch.where(maximum == 0, 1.0, maximum / 7)
    # The clamp also covers the smallest FP32 subnormals, where scale can round to zero.
    scales = scales.clamp_min(torch.finfo(torch.float32).smallest_normal * torch.finfo(torch.float32).eps)
    integers = (grouped / scales.unsqueeze(-1)).round().clamp(-7, 7).to(torch.int8)
    return QuantizedWeight(pack_int4(integers.flatten(start_dim=1)), scales, width, group_size)
