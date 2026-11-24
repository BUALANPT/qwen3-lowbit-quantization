"""Measure grouped INT4 reconstruction and linear output on Qwen3 weights."""

import csv
from pathlib import Path

import torch
from safetensors import safe_open

from int4 import quantize


ROOT = Path(__file__).resolve().parent
TENSORS = ('model.layers.0.self_attn.q_proj.weight', 'model.layers.0.mlp.down_proj.weight')


def relative_error(reference, approximate):
    denominator = torch.linalg.vector_norm(reference).item()
    numerator = torch.linalg.vector_norm(reference - approximate).item()
    return numerator / denominator if denominator else numerator


def measure(weight_name, source, generator):
    original = source.float()
    inputs = torch.randn((128, original.shape[1]), generator=generator)
    baseline = inputs @ original.T
    for group in (32, 64, 128):
        result = quantize(original, group)
        approximation = result.dequantize()
        yield {'tensor': weight_name, 'out_features': original.shape[0],
               'in_features': original.shape[1], 'group_size': group,
               'bf16_tensor_bytes': source.nbytes, 'packed_weight_bytes': result.packed.nbytes,
               'scale_bytes': result.scales.nbytes, 'int4_payload_bytes': result.payload_bytes,
               'relative_weight_l2': relative_error(original, approximation),
               'relative_output_l2': relative_error(baseline, inputs @ approximation.T)}


def main():
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(71)
    with safe_open(ROOT / 'models/hf/model.safetensors', framework='pt', device='cpu') as model:
        rows = [row for tensor in TENSORS for row in measure(tensor, model.get_tensor(tensor), generator)]
    destination = ROOT / 'results/int4_layer_errors.csv'
    destination.parent.mkdir(exist_ok=True)
    with destination.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'{len(rows)} INT4 layer measurements -> {destination.name}')


if __name__ == '__main__':
    main()
