"""Draw the model size / perplexity trade-off and the same-budget comparison."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
LABELS = {'f16': 'F16', 'q4_k_m': 'Q4_K_M', 'q4_k_m_im': 'Q4_K_M + imatrix',
          'q3_k_m_im': 'Q3_K_M + imatrix', 'q2_k_im': 'Q2_K + imatrix',
          'mixed_sensitive': 'Sensitive allocation', 'mixed_random_1': 'Random 1',
          'mixed_random_2': 'Random 2', 'mixed_random_3': 'Random 3'}


def main():
    with (RESULTS / 'metrics.csv').open(encoding='utf-8-sig', newline='') as file:
        table = {row['model']: (int(row['bytes']) / 2**20, float(row['test_ppl']))
                 for row in csv.DictReader(file)}
    missing = LABELS.keys() - table.keys()
    if missing:
        raise ValueError(f'Missing final measurements: {sorted(missing)}')
    navy, teal, orange, gray = '#204666', '#068478', '#ba552c', '#677080'
    figure, (overall, allocation) = plt.subplots(1, 2, figsize=(12, 5.4), layout='constrained',
                                                 gridspec_kw={'width_ratios': [1.18, 1]})
    figure.suptitle('Qwen3-1.7B-Base weight quantization', fontsize=17, fontweight='bold')
    offsets = {'f16': (-6, 9), 'q4_k_m': (11, 15), 'q4_k_m_im': (11, -14),
               'q3_k_m_im': (10, 11), 'q2_k_im': (11, 13), 'mixed_sensitive': (11, -18)}
    for name, (size, ppl) in table.items():
        if name.startswith('mixed_random'):
            continue
        color = teal if name == 'mixed_sensitive' else navy
        marker = 'D' if name == 'mixed_sensitive' else 'o'
        offset = offsets[name]
        overall.scatter(size, ppl, s=51, color=color, marker=marker, zorder=4)
        overall.annotate(LABELS[name], (size, ppl), xytext=offset, textcoords='offset points',
                         ha='right' if offset[0] < 0 else 'left', fontsize=8.5, color=color)
    overall.set(title='Size–quality trade-off', xlabel='GGUF size (MiB)', ylabel='WikiText-2 test perplexity')
    overall.margins(x=0.12, y=0.14)

    order = ['q2_k_im', 'mixed_sensitive', 'mixed_random_1', 'mixed_random_2', 'mixed_random_3']
    anchor = table['q2_k_im'][1]
    values = [table[name][1] - anchor for name in order]
    allocation.axhline(0, color=gray, linestyle='--', linewidth=1)
    allocation.scatter(range(5), values, color=[navy, teal, orange, orange, orange], s=57, zorder=4)
    for i, value in enumerate(values):
        allocation.annotate(f'{value:+.3f}', (i, value), xytext=(0, 8), textcoords='offset points',
                            ha='center', fontsize=8.5)
    allocation.set_xticks(range(5), ('Q2 base', 'Sensitive', 'Random 1', 'Random 2', 'Random 3'))
    extra_mib = table['mixed_sensitive'][0] - table['q2_k_im'][0]
    allocation.set(title=f'Mixed allocations add the same {extra_mib:.2f} MiB',
                   ylabel='Perplexity change vs. Q2_K + imatrix')
    allocation.margins(x=0.11, y=0.2)
    for axes in (overall, allocation):
        axes.spines[['top', 'right']].set_visible(False)
        axes.grid(axis='y', alpha=0.20)
    figure.savefig(RESULTS / 'size_vs_perplexity.png', dpi=185)
    plt.close(figure)
    print('results/size_vs_perplexity.png')


if __name__ == '__main__':
    main()
