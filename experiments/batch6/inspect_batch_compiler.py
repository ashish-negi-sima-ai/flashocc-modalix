"""Inspect installed compiler batch rewrites without generating MLA code."""

from pathlib import Path as _ProjectPath
import sys as _project_sys
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
_project_sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
import argparse
import json
import logging
import pickle
from pathlib import Path


class InspectionComplete(BaseException):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, default=6)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from afe.apis.model import Model
    import afe.backends.mla.afe_to_n2a_compiler.n2a_compiler_operations as operations
    import mlc.compiler.model_graph.large_tensor_helper as helper

    rewrites = []
    for cls in (helper.LargeTensorPool, helper.LargeTensorConv):
        original = cls._create_vertex

        def inspect(self, *pos, _original=original, **kw):
            result = _original(self, *pos, **kw)
            before = self._layer.operator._parameters
            after = result.operator._parameters
            entry = {'original': self._layer.name, 'segment': result.name,
                     'operator': type(result.operator).__name__,
                     'before_padding': before.padding, 'after_padding': after.padding,
                     'input': result.inputs[0].name,
                     'input_operator': type(result.inputs[0].operator).__name__,
                     'input_segment_axis': str(result.inputs[0].operator.segment_axis),
                     'before_shape': before.convolution_ifm_shape(),
                     'after_shape': after.convolution_ifm_shape()}
            rewrites.append(entry)
            if before.padding != after.padding:
                print('PADDING_CHANGED ' + json.dumps(entry), flush=True)
            return result

        cls._create_vertex = inspect

    original_properties = operations.set_model_compilation_properties

    def properties(config, model, **kw):
        with (args.output / 'original-graph.pkl').open('wb') as stream:
            pickle.dump((config, model), stream)
        original_properties(config, model, **kw)
        with (args.output / 'compiled-graph.pkl').open('wb') as stream:
            pickle.dump((config, model), stream)
        report = {'batch': args.batch, 'rewrites': rewrites,
                  'changed_padding': [r for r in rewrites
                                      if r['before_padding'] != r['after_padding']],
                  'compiled_pools': [
                      {'name': v.name, 'padding': v.operator._parameters.padding,
                       'batch': v.batch_size}
                      for v in model.compile_properties.layers
                      if type(v.operator).__name__ == 'PoolOperator'],
                  'expected_diagnostic_stop_before_codegen': True}
        (args.output / 'rewrites.json').write_text(json.dumps(report, indent=2) + '\n')
        print('INSPECTION_COMPLETE ' + str(args.output), flush=True)
        raise InspectionComplete()

    operations.set_model_compilation_properties = properties
    root = _PROJECT_ROOT
    model = Model.load('flashocc_m0_camera', str(root / 'artifacts/compile-real/camera'))
    try:
        model.compile(output_path=str(args.output), batch_size=args.batch, log_level=logging.INFO)
    except InspectionComplete:
        pass


if __name__ == '__main__':
    main()
