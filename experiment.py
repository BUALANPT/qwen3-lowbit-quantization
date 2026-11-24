"""Reproduce the GGUF experiment with the pinned Windows llama.cpp tools."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import msvcrt
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass


ROOT = Path(__file__).resolve().parent
MODELS = Path('models')
DATA = Path('data')
TEXT = DATA / 'wikitext-2-raw'
OUT = Path('results')
LOG = OUT / 'logs'
BIN = Path('tools/llama')
SOURCE = Path('tools/source')
MATRIX = DATA / 'qwen3-imatrix.gguf'
F16 = MODELS / 'f16.gguf'
BASELINES = {
    'f16': (None, False),
    'q4_k_m': ('Q4_K_M', False),
    'q4_k_m_im': ('Q4_K_M', True),
    'q3_k_m_im': ('Q3_K_M', True),
    'q2_k_im': ('Q2_K', True),
}
REVISION = 'ea980cb0a6c2ae4b936e82123acc929f1cec04c1'
WEIGHT_SHA256 = '6df85b39330e5a425ee36253d0f894e4387e4f0a15b9c53cb467d668e6b3a841'
DATA_SHA256 = {'wiki.train.raw': '6707892fa3788b5ab9ed78ab5ff37d9fe825f6011a2ad4fcd6a6d467f0e7da57',
               'wiki.valid.raw': '4cd0f6876d07a413aa911261ff6d363c72d757d47f0fdd6015702014c89cb9c7',
               'wiki.test.raw': '173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08'}
SETTINGS = {'model_revision': REVISION, 'llama_build': 'b12100', 'context': 512,
            'calibration_chunks': 32, 'validation_chunks': 32, 'gpu_layers': 99,
            'batch': 512, 'ubatch': 512, 'kv_type': 'f16', 'threads': 8,
            'benchmark_repeats': 5, 'random_seed': 71, 'candidate_count': 8,
            'upgrade_count': 3, 'random_groups': 3, 'calibration_prefix_characters': 140_000}
GPU_ARGS = ('-ngl', str(SETTINGS['gpu_layers']), '-b', str(SETTINGS['batch']),
            '-ub', str(SETTINGS['ubatch']), '-t', str(SETTINGS['threads']),
            '-ctk', SETTINGS['kv_type'], '-ctv', SETTINGS['kv_type'])
BENCH_PROMPT, BENCH_GENERATION = 512, 128


@dataclass(frozen=True)
class TensorInfo:
    kind: str
    shape: tuple[int, ...]
    nbytes: int


def final_ppl(log):
    matches = re.findall(r'Final estimate: PPL = (\d+\.\d+) \+/- (\d+\.\d+)', log)
    if not matches:
        raise ValueError('No finite final perplexity estimate in the tool output')
    return tuple(map(float, matches[-1]))


def require_full_cuda(log, name):
    buffers = re.findall(r'CUDA0 model buffer size =\s+([\d.]+) MiB', log)
    if not re.search(r'offloaded ([1-9]\d*)/\1 layers to GPU', log) or not any(float(x) > 0 for x in buffers):
        raise RuntimeError(f'Full CUDA offload not confirmed for {name}')


def mixed_controls(pool, top, seed):
    choices = [group for group in itertools.combinations(sorted(pool), len(top))
               if set(group) != set(top)]
    return random.Random(seed).sample(choices, SETTINGS['random_groups'])


def verify_override(baseline, variant, selected):
    if baseline.keys() != variant.keys():
        raise ValueError('The tensor inventories have different names')
    expected = set(selected)
    if not expected or not expected <= baseline.keys():
        raise ValueError('The selected names must exist in both GGUF files')
    changed = {name for name in baseline if baseline[name] != variant[name]}
    if changed != expected:
        raise ValueError(f'Expected to change {sorted(expected)}, changed {sorted(changed)}')
    for name in selected:
        before, after = baseline[name], variant[name]
        if after.kind != 'Q4_K' or after.shape != before.shape or after.nbytes <= before.nbytes:
            raise ValueError(f'Tensor {name} did not gain bytes as Q4_K: {before}, {after}')
    return sum(variant[name].nbytes - baseline[name].nbytes for name in selected)


def tensor_inventory(path):
    from gguf_tensors import read_tensor_headers

    return {tensor.name: TensorInfo(tensor.kind, tensor.shape, tensor.nbytes)
            for tensor in read_tensor_headers(path)}


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def log_output(name, command, *, cache=False):
    destination = LOG / f'{name}.log'
    if cache and destination.exists():
        return destination.read_text(encoding='utf-8', errors='replace')
    print(name, flush=True)
    temporary = destination.with_suffix('.partial')
    with temporary.open('wb') as log:
        subprocess.run([str(item) for item in command], stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    temporary.replace(destination)
    return destination.read_text(encoding='utf-8', errors='replace')


def command(tool, *args):
    return [BIN / f'llama-{tool}.exe', *args]


def create_gguf(name, destination, make_command):
    if destination.exists():
        return
    temporary = destination.with_suffix('.partial.gguf')
    log_output(name, make_command(temporary))
    if not temporary.exists() or temporary.stat().st_size == 0:
        raise ValueError(f'Tool did not create {temporary}')
    temporary.replace(destination)


def setup_models():
    output = log_output('cuda_devices', command('bench', '--list-devices'))
    if 'CUDA0:' not in output:
        raise RuntimeError('CUDA device missing. See results/logs/cuda_devices.log')
    version = log_output('llama_version', command('perplexity', '--version'))
    if 'build 12100' not in version:
        raise RuntimeError('Expected llama.cpp b12100')
    if not F16.exists() and sha256(MODELS / 'hf/model.safetensors') != WEIGHT_SHA256:
        raise ValueError('Qwen3 checkpoint is incomplete or from a different revision')
    create_gguf('convert_f16', F16, lambda tmp: [sys.executable,
                SOURCE / 'convert_hf_to_gguf.py', MODELS / 'hf', '--outfile', tmp, '--outtype', 'f16'])
    for name in ('f16', 'q4_k_m'):
        if name == 'q4_k_m':
            quantize_gguf(name, 'Q4_K_M')
        output = log_output(f'smoke_{name}', command('completion', '-m', MODELS / f'{name}.gguf',
                            '-ngl', '99', '-c', '512', '-n', '32', '--no-conversation',
                            '--log-verbosity', '4', '--seed', '71', '-p', 'The basic idea of quantization is'))
        require_full_cuda(output, name)


def quantize_gguf(name, recipe, calibrated=False, targets=()):
    options = ['--imatrix', MATRIX] if calibrated else []
    for target in targets:
        options.extend(['--tensor-type', f'^{re.escape(target)}$=q4_k'])
    destination = MODELS / f'{name}.gguf'
    create_gguf(f'quantize_{name}', destination, lambda tmp: command('quantize', *options,
                F16, tmp, recipe, SETTINGS['threads']))
    return destination


def measure_ppl(name, model, split, chunks=None):
    args = ('--chunks', str(chunks)) if chunks else ()
    log = log_output(f'ppl_{split}_{name}', command('perplexity', '-m', model,
                     '-f', TEXT / f'wiki.{split}.raw', '-c', SETTINGS['context'], '--log-verbosity', '4',
                     *GPU_ARGS, *args), cache=True)
    require_full_cuda(log, name)
    return final_ppl(log)


def calibration_prefix(source, destination, size):
    # This Windows imatrix build can crash tokenizing the full 11 MB train file.
    # Keep the exact prefix and provide more text than the 32 chunks consumed.
    with source.open(encoding='utf-8', newline='') as file:
        destination.write_bytes(file.read(size).encode('utf-8'))


def baselines():
    setup_models()
    prefix = DATA / 'calibration.txt'
    calibration_prefix(TEXT / 'wiki.train.raw', prefix, SETTINGS['calibration_prefix_characters'])
    chunks = SETTINGS['calibration_chunks']
    create_gguf('imatrix', MATRIX, lambda tmp: command('imatrix', '-m', F16, '-f',
                prefix, '-c', SETTINGS['context'], '--chunks', chunks, '--no-ppl', '--log-verbosity', '4',
                '-o', tmp, *GPU_ARGS))
    calibration_log = (LOG / 'imatrix.log').read_text(encoding='utf-8', errors='replace')
    if f'computing over {chunks} chunks' not in calibration_log:
        raise RuntimeError(f'Expected {chunks} chunks in the CUDA calibration')
    require_full_cuda(calibration_log, 'imatrix')
    rows = []
    for name, (recipe, calibrated) in BASELINES.items():
        model = quantize_gguf(name, recipe, calibrated) if recipe else F16
        ppl, uncertainty = measure_ppl(name, model, 'valid', SETTINGS['validation_chunks'])
        rows.append({'model': name, 'bytes': model.stat().st_size, 'valid_ppl': ppl,
                     'valid_ppl_uncertainty': uncertainty})
    write_csv(OUT / 'validation_baselines.csv', rows)


def candidate_pool(inventory):
    groups = {}
    pattern = re.compile(r'blk\.(\d+)\.ffn_down\.weight')
    for name, value in inventory.items():
        match = pattern.fullmatch(name)
        if match and value.kind in {'Q2_K', 'Q3_K'}:
            groups.setdefault(value, []).append((int(match[1]), name))
    if not groups:
        raise ValueError('No eligible FFN down tensors')
    eligible = sorted(max(groups.values(), key=len))
    count = SETTINGS['candidate_count']
    if len(eligible) < count:
        raise ValueError(f'Need {count} equally sized, identically quantized FFN down candidates')
    return [eligible[round(i * (len(eligible) - 1) / (count - 1))][1] for i in range(count)]


def scan():
    anchor = MODELS / 'q2_k_im.gguf'
    inventory = tensor_inventory(anchor)
    if (OUT / 'allocations.json').exists():
        locked = json.loads((OUT / 'allocations.json').read_text(encoding='utf-8'))['models']
        assemble_allocations(anchor, inventory, locked)
        return
    pool = candidate_pool(inventory)
    anchor_ppl, _ = measure_ppl('q2_k_im', anchor, 'valid', SETTINGS['validation_chunks'])
    rows = []
    for tensor in pool:
        name = 'scan_' + tensor.split('.')[1]
        path = MODELS / f'{name}.gguf'
        if not (LOG / f'ppl_valid_{name}.log').exists() or path.exists():
            quantize_gguf(name, 'Q2_K', True, (tensor,))
            extra = verify_override(inventory, tensor_inventory(path), (tensor,))
            write_json(LOG / f'{name}_audit.json', {'tensor': tensor, 'extra_payload_bytes': extra,
                       'base_tensor_type': inventory[tensor].kind, 'new_tensor_type': 'Q4_K',
                       'file_bytes': path.stat().st_size})
        ppl, uncertainty = measure_ppl(name, path, 'valid', SETTINGS['validation_chunks'])
        audit = json.loads((LOG / f'{name}_audit.json').read_text(encoding='utf-8'))
        rows.append({**audit, 'valid_ppl': ppl, 'valid_ppl_uncertainty': uncertainty,
                     'delta_nll_improvement': math.log(anchor_ppl / ppl)})
        path.unlink(missing_ok=True)
    rows.sort(key=lambda row: (-row['delta_nll_improvement'], row['tensor']))
    write_csv(OUT / 'sensitivity.csv', rows)
    top = tuple(row['tensor'] for row in rows[:SETTINGS['upgrade_count']])
    allocations = {'mixed_sensitive': top}
    for index, allocation in enumerate(mixed_controls(pool, top, SETTINGS['random_seed']), 1):
        allocations[f'mixed_random_{index}'] = allocation
    assemble_allocations(anchor, inventory, allocations)
    write_json(OUT / 'allocations.json', {'pool': pool, 'models': allocations,
               'validation_chunks': SETTINGS['validation_chunks'], 'seed': SETTINGS['random_seed']})


def assemble_allocations(anchor, inventory, allocations):
    expected_bytes = None
    audits = []
    for name, tensors in allocations.items():
        path = quantize_gguf(name, 'Q2_K', True, tensors)
        extra = verify_override(inventory, tensor_inventory(path), tensors)
        file_bytes = path.stat().st_size
        if expected_bytes is None:
            expected_bytes = (extra, file_bytes)
        if (extra, file_bytes) != expected_bytes:
            raise ValueError('Mixed precision variants do not have identical byte budgets')
        audits.append({'model': name, 'tensors': ', '.join(tensors), 'extra_payload_bytes': extra,
                       'extra_file_bytes': file_bytes - anchor.stat().st_size, 'file_bytes': file_bytes})
    write_csv(OUT / 'allocation_audit.csv', audits)


def measure_bench(name, model):
    idle_file = LOG / f'idle_{name}.csv'
    device_query = ['nvidia-smi', '-i', '0', '--query-gpu=timestamp,memory.used',
                    '--format=csv,noheader,nounits']
    sample_file = LOG / f'vram_{name}.csv'
    bench_file = LOG / f'bench_{name}.csv'
    if bench_file.exists():
        try:
            return parse_benchmark(bench_file, idle_file, sample_file)
        except (OSError, KeyError, ValueError):
            print(f'Remeasuring {name}: incomplete benchmark or VRAM samples', flush=True)
            bench_file.unlink()
    print(f'bench_{name}', flush=True)
    with idle_file.open('wb') as idle:
        subprocess.run(device_query, stdout=idle, check=True)
    temporary = bench_file.with_suffix('.partial')
    with (sample_file.open('wb', buffering=0) as samples,
          temporary.open('wb') as bench_out,
          (LOG / f'bench_{name}.log').open('wb') as bench_err):
        sampler = subprocess.Popen([*device_query, '-lms', '100'], stdout=samples)
        try:
            subprocess.run([str(a) for a in command('bench', '-m', model, '-p', BENCH_PROMPT,
                           '-n', BENCH_GENERATION, *GPU_ARGS, '-r', SETTINGS['benchmark_repeats'],
                           '-o', 'csv')], stdout=bench_out,
                           stderr=bench_err, check=True)
        finally:
            sampler.terminate()
            sampler.wait(timeout=5)
    metrics = parse_benchmark(temporary, idle_file, sample_file)
    temporary.replace(bench_file)
    return metrics


def parse_benchmark(bench_file, idle_file, sample_file):
    with bench_file.open(encoding='utf-8-sig', newline='') as file:
        rows = list(csv.DictReader(file))
    by_workload = {(int(row['n_prompt']), int(row['n_gen'])): row for row in rows}
    pp, tg = by_workload[BENCH_PROMPT, 0], by_workload[0, BENCH_GENERATION]
    if any('CUDA' not in row['backends'] for row in (pp, tg)):
        raise ValueError(f'Benchmark did not identify CUDA in {bench_file}')

    def memory_samples(path):
        with path.open(newline='', encoding='utf-8-sig') as file:
            return [int(row[1]) for row in csv.reader(file)
                    if len(row) == 2 and row[1].strip().isdigit()]

    samples, idle = memory_samples(sample_file), memory_samples(idle_file)
    if not samples or not idle or max(samples) <= idle[0]:
        raise ValueError(f'No usable run and idle VRAM samples for {bench_file}')
    idle_mib = idle[0]
    return {'prefill_tps': float(pp['avg_ts']), 'prefill_std': float(pp['stddev_ts']),
            'decode_tps': float(tg['avg_ts']), 'decode_std': float(tg['stddev_ts']),
            'vram_idle_mib': idle_mib, 'vram_peak_mib': max(samples),
            'vram_increment_mib': max(samples) - idle_mib, 'vram_samples': len(samples)}


def evaluate():
    locked = json.loads((OUT / 'allocations.json').read_text(encoding='utf-8'))['models']
    rows = []
    for name in [*BASELINES, *locked]:
        model = MODELS / f'{name}.gguf'
        ppl, uncertainty = measure_ppl(name, model, 'test')
        rows.append({'model': name, 'bytes': model.stat().st_size, 'test_ppl': ppl,
                     'ppl_uncertainty': uncertainty, **measure_bench(name, model)})
        write_csv(OUT / 'metrics.csv', rows)
    write_json(OUT / 'comparison.json', compare(rows))


def compare(rows):
    by_name = {row['model']: row for row in rows}
    base = by_name['q2_k_im']['test_ppl']
    full = by_name['f16']['test_ppl']
    sensitive = by_name['mixed_sensitive']['test_ppl']
    random_ppl = [by_name[f'mixed_random_{i}']['test_ppl'] for i in (1, 2, 3)]
    return {'q2_to_sensitive_ppl_reduction': base - sensitive,
            'sensitive_nll_gap_recovered_fraction': math.log(base / sensitive) / math.log(base / full)
            if base > full else None,
            'random_ppl': random_ppl, 'random_mean_ppl': statistics.mean(random_ppl),
            'sensitive_advantage_over_random_mean_ppl': statistics.mean(random_ppl) - sensitive,
            'extra_bytes_vs_q2': by_name['mixed_sensitive']['bytes'] - by_name['q2_k_im']['bytes']}


def sha256(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def record_protocol():
    protocol = OUT / 'protocol.json'
    inputs = {f'wiki.{split}.raw': sha256(TEXT / f'wiki.{split}.raw')
              for split in ('train', 'valid', 'test')}
    if inputs != DATA_SHA256:
        raise ValueError('WikiText-2 does not match the pinned split SHA-256 values')
    settings = {**SETTINGS, 'model_sha256': WEIGHT_SHA256, 'data_sha256': inputs}
    if protocol.exists() and json.loads(protocol.read_text(encoding='utf-8')) != settings:
        raise ValueError('The dataset or protocol changed. Save the previous run before starting another.')
    write_json(protocol, settings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'baselines', 'scan', 'evaluate', 'all'))
    stage = parser.parse_args().stage
    os.chdir(ROOT)
    LOG.mkdir(parents=True, exist_ok=True)
    with (OUT / '.run.lock').open('a+b') as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise SystemExit('Another experiment process owns this workspace') from exc
        record_protocol()
        if stage == 'prepare':
            setup_models()
        if stage in ('baselines', 'all'):
            baselines()
        if stage in ('scan', 'all'):
            scan()
        if stage in ('evaluate', 'all'):
            evaluate()


if __name__ == '__main__':
    main()
