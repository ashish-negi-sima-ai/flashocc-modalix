"""Summarize AFE local/accumulated errors, normalized by each FP32 tensor RMS."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def summarize(result):
    rows = []
    names = result['samples'][0]['local_feed']['mse']
    for name in names:
        row = {'name': name, **result['layers'].get(name, {})}
        local, accumulated, maes, peaks = [], [], [], []
        reference_rms = []
        for sample in result['samples']:
            stats = sample['reference_stats']
            mean, std = stats['mean'][name][0], stats['std'][name][0]
            rms_squared = mean ** 2 + std ** 2
            reference_rms.append(np.sqrt(rms_squared))
            local.append(np.sqrt(sample['local_feed']['mse'][name][0] / max(rms_squared, 1e-30)))
            accumulated.append(np.sqrt(sample['global_feed']['mse'][name][0] / max(rms_squared, 1e-30)))
            maes.append(sample['local_feed']['mae'][name][0])
            peaks.append(sample['global_feed']['psnr'][name][0])
        row.update(local_relative_rmse_mean=float(np.mean(local)),
                   local_relative_rmse_max=float(np.max(local)),
                   global_relative_rmse_mean=float(np.mean(accumulated)),
                   global_relative_rmse_max=float(np.max(accumulated)),
                   local_mae_mean=float(np.mean(maes)), global_psnr_mean=float(np.mean(peaks)),
                   reference_rms_mean=float(np.mean(reference_rms)))
        rows.append(row)
    return rows


def source(row):
    names = row.get('source_names', [])
    # Prefer the convolution source over fused activation names.
    name = next((n for n in names if n.endswith('/Conv')), names[0] if names else row['name'])
    return {'/Conv': '/occ_head/predicter/0 (pointwise Linear)',
            '/Conv_1': '/occ_head/predicter/2 (classifier)',
            '/predicter.1/Softplus': '/occ_head/predicter/1/Softplus',
            '/final_conv/conv/Conv': '/occ_head/final_conv/conv/Conv',
            '/depth_net/Conv': '/img_view_transformer/depth_net/Conv',
            '/Slice': '/depth_output/slice',
            '/Softmax': '/depth_output/Softmax'}.get(name, name)


def table(rows):
    lines = ['| Layer | Local relative RMSE | Accumulated relative RMSE |',
             '| --- | ---: | ---: |']
    for row in rows:
        lines.append(f"| `{source(row)}` (`{row['name'].split('/')[-1]}`) | {row['local_relative_rmse_mean']:.2%} | {row['global_relative_rmse_mean']:.2%} |")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path(__file__).resolve().parents[1] / 'artifacts/ptq-layerwise')
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    report = ['# FlashOCC INT8 PTQ layer analysis', '',
              'These are numerical errors relative to the saved AFE FP32 models, not ground-truth occupancy accuracy.', '',
              '**Local feed:** each operator receives FP32 reference inputs quantized to its input format. This removes upstream accumulated error but includes input quantization, weights, arithmetic and output quantization. '
              '**Global feed:** the quantized network runs from its original input, allowing error to propagate.', '',
              'Relative RMSE is `sqrt(MSE) / sqrt(reference_mean² + reference_std²)`, calculated per input and then averaged. '
              'Normalization allows comparisons across layers with different activation magnitudes. '
              'The analysis is at fused AFE operator boundaries, not every original PyTorch operation.', '',
              'A large local error identifies a candidate for a precision experiment; it does not establish the occupancy benefit of changing that operator alone. '
              'The BEV inputs come from FP32 camera/pooling, so these measurements isolate BEV quantization and omit cross-stage camera error. '
              'Plots follow analyzer operator order; residual branches mean adjacent points do not necessarily form a single sequential path.', '']
    summary = {}
    for stage in ['camera', 'bev']:
        result = json.loads((args.directory / (stage + '.json')).read_text())
        rows = summarize(result)
        directory = Path(result['model_directory'])
        model_name = 'flashocc_m0_' + stage
        summary[stage] = {'sample_count': len(result['samples']), 'layers': rows,
                          'model_sha256': hashlib.sha256((directory / (model_name + '.sima')).read_bytes()).hexdigest(),
                          'reference_sha256': hashlib.sha256((directory / (model_name + '.fp32.sima')).read_bytes()).hexdigest()}
        keys = ['name', 'source_names', 'local_relative_rmse_mean', 'local_relative_rmse_max',
                'global_relative_rmse_mean', 'global_relative_rmse_max', 'local_mae_mean', 'global_psnr_mean', 'reference_rms_mean']
        with (args.directory / (stage + '.csv')).open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: '; '.join(row[k]) if k == 'source_names' else row[k] for k in keys})
        ranked = sorted(rows, key=lambda r: r['local_relative_rmse_mean'], reverse=True)
        report.extend([f'## {stage.capitalize()}: {len(result["samples"])} holdout inputs, {len(rows)} fused operators', '',
                       'Largest local normalized errors:', '', table(ranked[:10]), '',
                       'Largest accumulated normalized errors:', '',
                       table(sorted(rows, key=lambda r: r['global_relative_rmse_mean'], reverse=True)[:5]), ''])
        zeroed = [r for r in rows if r.get('all_zero_output_channels')]
        if zeroed:
            report.extend(['### Zeroed weight filters', '',
                           'The table compares all-zero INT8 output filters with their original folded FP32 weight energy. '
                           'Near-zero energy makes the warning weak evidence for a meaningful loss; biases and downstream sensitivity still matter.', '',
                           '| Layer | Zero INT8 filters | Original weight energy in those filters |',
                           '| --- | ---: | ---: |'])
            for row in zeroed:
                report.append(f"| `{source(row)}` | {len(row['all_zero_output_channels'])}/{row['output_channels']} | {row['zeroed_channels_original_weight_energy_fraction']:.3e} |")
            report.append('')
        fig, ax = plt.subplots(figsize=(15, 5 if stage == 'bev' else 7))
        xs = np.arange(len(rows))
        ax.plot(xs, [r['local_relative_rmse_mean'] * 100 for r in rows], '.-', label='Local: quantized reference inputs')
        ax.plot(xs, [r['global_relative_rmse_mean'] * 100 for r in rows], '.-', label='Accumulated: quantized upstream inputs')
        ax.set_xticks(xs)
        ax.set_xticklabels([r['name'].split('/')[-1] for r in rows], rotation=90, fontsize=7)
        ax.set_ylabel('Mean relative RMSE (%)')
        ax.set_title(f'FlashOCC {stage}: INT8 vs FP32, {len(result["samples"])} holdout inputs')
        ax.grid(axis='y', alpha=.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.directory / (stage + '.png'), dpi=160)
        plt.close(fig)
        report.extend([f'![{stage} errors]({stage}.png)', ''])
    (args.directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.directory / 'REPORT.md').write_text('\n'.join(report) + '\n')


if __name__ == '__main__':
    main()
