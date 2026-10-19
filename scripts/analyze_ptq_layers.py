"""Analyze saved INT8 models with the installed AFE graph analyzer (SDK host)."""
import argparse
import gc
import io
import json
import zipfile
from enum import Enum
from pathlib import Path

import numpy as np
import yaml


def read_network(path):
    with zipfile.ZipFile(path) as archive:
        network = yaml.safe_load(archive.read(next(n for n in archive.namelist() if n.endswith('.yaml'))))[-1]
        arrays = np.load(io.BytesIO(archive.read(next(n for n in archive.namelist() if n.endswith('.npz')))))
    nodes = {}
    def visit(net):
        for node in net['nodes']:
            if node['ir'].get('node_type') == 'subgraph':
                visit(node['ir'])
            else:
                nodes[node['name']] = node
    visit(network)
    return nodes, arrays


def weight_stats(directory, name):
    nodes, arrays = read_network(directory / (name + '.sima'))
    reference, fp_arrays = read_network(directory / (name + '.fp32.sima'))
    result = {}
    for node_name, node in nodes.items():
        quant = node['ir'].get('quant') or {}
        key = quant.get('weight_quant_data')
        if not key:
            continue
        value = arrays[key]
        assert list(value.shape) == quant['conv']['weight_shape']
        channels = value.reshape(-1, value.shape[-1])
        zero_channels = np.flatnonzero(np.all(channels == 0, axis=0))
        ref_key = reference[node_name]['ir']['attrs']['weights']['data']
        original = fp_arrays[ref_key]
        assert original.shape == value.shape
        energy = np.sum(original.astype(np.float64) ** 2, axis=tuple(range(original.ndim - 1)))
        result[node_name] = {
            'zero_weight_fraction': float(np.mean(value == 0)),
            'all_zero_output_channels': zero_channels.tolist(),
            'output_channels': value.shape[-1],
            'original_all_zero_output_channels': np.flatnonzero(energy == 0).tolist(),
            'zeroed_channels_original_weight_energy_fraction': float(energy[zero_channels].sum() / max(energy.sum(), 1e-30)),
        }
    return result


def serial(value):
    if isinstance(value, dict):
        return {str(k.value if isinstance(k, Enum) else k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    return value


def main():
    from afe.apis.model import Model
    from afe.core.graph_analyzer.graph_analyzer import QuantizedGraphAnalyzer, IntermediatesGraphAnalyzer
    from afe.core.graph_analyzer.utils import Metric, QuantizedGraphAnalyzerMode

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stages', nargs='+', choices=['camera', 'bev'], default=['bev', 'camera'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    root = Path(__file__).resolve().parents[1]
    output = args.output or root / 'artifacts/ptq-layerwise'
    output.mkdir(parents=True, exist_ok=True)
    # SDK's high-level analysis call mutates layer stats and does not return
    # measurements. Use its graph analyzer directly to retain per-input metrics
    # for both modes. _net/_fp32_net are version-specific AFE adapter access.
    for stage in args.stages:
        directory = root / 'artifacts' / ('compile-real-int8' if stage == 'camera' else 'compile-real') / stage
        name = 'flashocc_m0_' + stage
        topology = json.loads((directory / (name + '.sima.json')).read_text())
        layers = topology['modelTopology']['model_config']['config']['layers']
        weights = weight_stats(directory, name)
        paths = sorted((root / 'artifacts/real-calibration' / stage / 'validation').glob('*.npy'))
        # All eight BEV holdouts; all six views of two holdout scenes for camera.
        if stage == 'camera':
            tokens = ['3e8750f331d7499e9b5123e9eb70f2e2', 'b4ff30109dd14c89b24789dc5713cf8c']
            paths = [p for p in paths if p.stem.split('-')[0] in tokens]
        if args.limit:
            paths = paths[:args.limit]
        if not paths:
            raise ValueError(f'No holdout inputs for {stage}')
        result = {'stage': stage, 'model_directory': str(directory),
                  'ground_truth_accuracy': False, 'executor': 'AFE QuantizedGraphAnalyzer',
                  'mode_definitions': {'local_feed': 'Each layer receives quantized FP32 reference inputs; isolates local error including input quantization.',
                                       'global_feed': 'Quantized execution from model input; includes upstream accumulated error.'},
                  'layers': {l['name']: {'source_names': l['config'].get('Source names', []),
                                        'class_name': l['class_name'], **weights.get(l['name'], {})} for l in layers},
                  'samples': []}
        report_path = output / (stage + '.json')
        previous = json.loads(report_path.read_text())['samples'] if args.resume and report_path.exists() else []
        previous = {s['input']: s for s in previous}
        print('Loading', stage, flush=True)
        model = Model.load(name, str(directory))
        for path in paths:
            inputs = {stage + '_input': np.load(path).transpose(0, 2, 3, 1)}
            sample = previous.get(str(path), {'input': str(path)})
            if 'reference_stats' not in sample:
                print(stage, path.stem, 'reference_stats', flush=True)
                analyzer = IntermediatesGraphAnalyzer()
                analyzer.analyze(model._fp32_net, [inputs], [Metric.mean, Metric.std], dequantize_intermediate=False)
                sample['reference_stats'] = serial(analyzer.analyzed_results)
                del analyzer
                gc.collect()
            for mode_name in ['local_feed', 'global_feed']:
                if mode_name in sample:
                    continue
                print(stage, path.stem, mode_name, flush=True)
                analyzer = QuantizedGraphAnalyzer(mode=getattr(QuantizedGraphAnalyzerMode, mode_name))
                analyzer.analyze(model._fp32_net, model._net, [inputs], [Metric.mae, Metric.mse, Metric.psnr])
                sample[mode_name] = serial(analyzer.analyzed_results)
                del analyzer
                gc.collect()
            result['samples'].append(sample)
            report_path.write_text(json.dumps(result, indent=2, allow_nan=True) + '\n')
        del model
        gc.collect()
        print('Completed', stage, len(paths), flush=True)


if __name__ == '__main__':
    main()
