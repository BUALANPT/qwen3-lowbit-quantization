import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiment import (TensorInfo, calibration_prefix, final_ppl, mixed_controls,
                        parse_benchmark, require_full_cuda, verify_override)


class ExperimentTests(unittest.TestCase):
    def test_calibration_prefix_preserves_whitespace_and_utf8(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / 'train.txt'
            target = Path(directory) / 'calibration.txt'
            source.write_bytes('a\r\n雪b'.encode('utf-8'))
            calibration_prefix(source, target, 4)
            self.assertEqual(target.read_bytes(), 'a\r\n雪'.encode('utf-8'))

    def test_benchmark_requires_vram_samples_from_a_valid_run(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            table, idle, samples = [root / f'{name}.csv' for name in ('bench', 'idle', 'vram')]
            table.write_text('n_prompt,n_gen,backends,avg_ts,stddev_ts\n512,0,CUDA,3500.0,1.0\n0,128,CUDA,200.0,2.0\n')
            idle.write_text('t0, 1000\n')
            samples.write_text('')
            with self.assertRaises(ValueError):
                parse_benchmark(table, idle, samples)
            samples.write_text('t1, 2200\nt2, 1800\n')
            result = parse_benchmark(table, idle, samples)
            self.assertEqual(result['vram_increment_mib'], 1200)
            self.assertEqual(result['decode_tps'], 200.0)

    def test_offload_requires_a_nonempty_cuda_weight_buffer(self):
        require_full_cuda('offloaded 29/29 layers to GPU\nCUDA0 model buffer size = 3281.97 MiB', 'f16')
        for sample in ('offloaded 29/29 layers to GPU\nCUDA0 model buffer size = 0.00 MiB',
                       'offloaded 5/29 layers to GPU\nCUDA0 model buffer size = 800 MiB'):
            with self.assertRaises(RuntimeError):
                require_full_cuda(sample, 'f16')

    def test_ppl_parser_ignores_running_estimates(self):
        sample = '[1]23.09,[2]10.93\nFinal estimate: PPL = 12.3750 +/- 0.30320\n'
        self.assertEqual(final_ppl(sample), (12.375, 0.3032))
        with self.assertRaises(ValueError):
            final_ppl('[1]23.09,[2]10.93')

    def test_random_allocations_are_reproducible_and_distinct(self):
        pool = [f'layer-{i}' for i in range(8)]
        top = tuple(pool[:3])
        first = mixed_controls(pool, top, seed=71)
        self.assertEqual(first, mixed_controls(pool, top, seed=71))
        self.assertEqual(len(set(first) | {top}), 4)
        self.assertTrue(all(len(set(group)) == 3 and set(group) <= set(pool) for group in first))

    def test_rejects_changes_outside_the_selected_tensors(self):
        low = TensorInfo('Q2_K', (256, 256), 21504)
        high = TensorInfo('Q4_K', (256, 256), 36864)
        original = {'a': low, 'b': low}
        self.assertEqual(verify_override(original, {'a': high, 'b': low}, ('a',)), 15360)
        for invalid in ({'a': high, 'b': high}, {'a': low, 'b': low}, {'a': high}):
            with self.assertRaises(ValueError):
                verify_override(original, invalid, ('a',))
