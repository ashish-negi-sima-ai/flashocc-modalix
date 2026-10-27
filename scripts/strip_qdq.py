"""Strip QDQ nodes from a QAT QDQ ONNX to produce a clean FP32 ONNX.

Weight QDQ nodes (INT8 initializer → DequantizeLinear) are folded to float32
by dequantizing the INT8 weights in numpy.  Activation QDQ pairs
(QuantizeLinear → DequantizeLinear on non-constant tensors) are removed by
rewiring consumers directly to the QL's input.  When a DQL output feeds a
graph output directly, the QL is removed but the DQL is kept (wired to the
QL's original input) so the graph output tensor name remains valid.

Usage:
    python strip_qdq.py camera.qdq.onnx camera.fp32.onnx
    python strip_qdq.py bev.qdq.onnx   bev.fp32.onnx
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper, helper, TensorProto


def _initializer_map(model: onnx.ModelProto) -> dict:
    return {init.name: init for init in model.graph.initializer}


def _make_node(node, new_inputs):
    return helper.make_node(
        node.op_type,
        inputs=new_inputs,
        outputs=list(node.output),
        name=node.name,
        domain=node.domain,
        **{attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute},
    )


def _fold_weight_qdq(model: onnx.ModelProto) -> onnx.ModelProto:
    """Fold INT8 initializer → DequantizeLinear → float32 initializer manually."""
    init_map = _initializer_map(model)
    node_list = list(model.graph.node)

    substitution: dict[str, str] = {}
    nodes_to_remove: set[int] = set()
    new_inits: list = []

    for idx, node in enumerate(node_list):
        if node.op_type != 'DequantizeLinear':
            continue
        data_name = node.input[0]
        if data_name not in init_map:
            continue
        data_init = init_map[data_name]
        if data_init.data_type != TensorProto.INT8:
            continue

        # Retrieve int8 data, scale, zero_point
        data = numpy_helper.to_array(data_init).astype(np.float32)

        scale_name = node.input[1]
        if scale_name not in init_map:
            continue
        scale = numpy_helper.to_array(init_map[scale_name]).astype(np.float32)

        zp = np.float32(0)
        if len(node.input) > 2 and node.input[2] and node.input[2] in init_map:
            zp = numpy_helper.to_array(init_map[node.input[2]]).astype(np.float32)

        # Determine quantization axis for per-channel weights
        axis = None
        for attr in node.attribute:
            if attr.name == 'axis':
                axis = int(attr.i)
                break

        if axis is not None and scale.ndim > 0 and scale.size > 1:
            shape = [1] * data.ndim
            shape[axis] = -1
            scale = scale.reshape(shape)
            if zp.ndim > 0:
                zp = zp.reshape(shape)

        float_data = (data - zp) * scale

        new_name = f'__w_{data_name}'
        new_init = numpy_helper.from_array(float_data.astype(np.float32), name=new_name)
        new_inits.append(new_init)

        dql_output = node.output[0]
        substitution[dql_output] = new_name
        nodes_to_remove.add(idx)

    if not substitution:
        print('[strip_qdq] No weight DQL nodes found to fold', flush=True)
        return model

    print(f'[strip_qdq] Manually folded {len(substitution)} weight DQL nodes', flush=True)

    new_nodes = []
    for idx, node in enumerate(node_list):
        if idx in nodes_to_remove:
            continue
        new_inputs = [substitution.get(inp, inp) for inp in node.input]
        new_nodes.append(_make_node(node, new_inputs))

    new_init_list = [i for i in model.graph.initializer
                     if i.name not in {init_map[k].name for k in substitution
                                       if k in {substitution[k2] for k2 in substitution}
                                       }
                     ] + new_inits
    # Simpler: keep all original initializers + add new float32 ones
    new_init_list = list(model.graph.initializer) + new_inits

    # Build graph applying substitution to outputs as well
    new_outputs = []
    for out in model.graph.output:
        new_name = substitution.get(out.name, out.name)
        if new_name != out.name:
            new_vi = onnx.ValueInfoProto()
            new_vi.CopyFrom(out)
            new_vi.name = new_name
            new_outputs.append(new_vi)
        else:
            new_outputs.append(out)

    new_graph = helper.make_graph(
        new_nodes,
        model.graph.name,
        list(model.graph.input),
        new_outputs,
        initializer=new_init_list,
    )
    surviving = set()
    for n in new_nodes:
        surviving.update(n.input)
        surviving.update(n.output)
    for vi in model.graph.value_info:
        if vi.name in surviving:
            new_graph.value_info.append(vi)

    new_model = helper.make_model(new_graph, opset_imports=list(model.opset_import))
    new_model.ir_version = model.ir_version
    return new_model


def _strip_activation_qdq(model: onnx.ModelProto) -> onnx.ModelProto:
    """Remove QuantizeLinear → DequantizeLinear pairs on non-constant tensors.

    If a DQL output is a graph output, the QL is removed but the DQL is kept
    (rewired to the QL's original input) so the graph output name stays valid.
    """
    init_map = _initializer_map(model)
    graph_output_names = {out.name for out in model.graph.output}

    node_list = list(model.graph.node)
    # substitution maps an old tensor name → replacement tensor name
    substitution: dict[str, str] = {}
    nodes_to_remove: set[int] = set()

    for idx, node in enumerate(node_list):
        if node.op_type != 'QuantizeLinear':
            continue
        ql_input = node.input[0]
        if ql_input in init_map:
            continue  # weight QL — skip
        ql_output = node.output[0]
        for jdx, other in enumerate(node_list):
            if other.op_type != 'DequantizeLinear' or other.input[0] != ql_output:
                continue
            dql_output = other.output[0]
            if dql_output in graph_output_names:
                # Remove both QL and DQL; the graph output will be remapped
                # from dql_output → ql_input in the new_outputs section.
                substitution[dql_output] = ql_input
                nodes_to_remove.add(idx)
                nodes_to_remove.add(jdx)
            else:
                substitution[dql_output] = ql_input
                nodes_to_remove.add(idx)
                nodes_to_remove.add(jdx)

    if not substitution:
        print('[strip_qdq] No activation QDQ pairs found — model unchanged', flush=True)
        return model

    print(f'[strip_qdq] Removing {len(nodes_to_remove)} activation QDQ nodes '
          f'({len(substitution)} pairs)', flush=True)

    new_nodes = []
    for idx, node in enumerate(node_list):
        if idx in nodes_to_remove:
            continue
        new_inputs = [substitution.get(inp, inp) for inp in node.input]
        new_nodes.append(_make_node(node, new_inputs))

    # Graph outputs: names are tensor names; apply substitution
    new_outputs = []
    for out in model.graph.output:
        new_name = substitution.get(out.name, out.name)
        if new_name != out.name:
            new_vi = onnx.ValueInfoProto()
            new_vi.CopyFrom(out)
            new_vi.name = new_name
            new_outputs.append(new_vi)
        else:
            new_outputs.append(out)

    new_graph = helper.make_graph(
        new_nodes,
        model.graph.name,
        list(model.graph.input),
        new_outputs,
        initializer=list(model.graph.initializer),
    )
    surviving = set()
    for n in new_nodes:
        surviving.update(n.input)
        surviving.update(n.output)
    surviving.update(out.name for out in new_outputs)
    for vi in model.graph.value_info:
        if vi.name in surviving:
            new_graph.value_info.append(vi)

    new_model = helper.make_model(new_graph, opset_imports=list(model.opset_import))
    new_model.ir_version = model.ir_version
    return new_model


def strip_qdq(input_path: Path, output_path: Path) -> None:
    print(f'[strip_qdq] Loading {input_path}', flush=True)
    model = onnx.load(str(input_path))

    ql_before = sum(1 for n in model.graph.node if n.op_type == 'QuantizeLinear')
    dql_before = sum(1 for n in model.graph.node if n.op_type == 'DequantizeLinear')
    print(f'[strip_qdq] Before: {ql_before} QuantizeLinear, {dql_before} DequantizeLinear', flush=True)

    print('[strip_qdq] Folding weight QDQ (manual INT8→float32)...', flush=True)
    model = _fold_weight_qdq(model)

    ql_after_fold = sum(1 for n in model.graph.node if n.op_type == 'QuantizeLinear')
    dql_after_fold = sum(1 for n in model.graph.node if n.op_type == 'DequantizeLinear')
    print(f'[strip_qdq] After weight fold: {ql_after_fold} QuantizeLinear, {dql_after_fold} DequantizeLinear',
          flush=True)

    print('[strip_qdq] Removing activation QDQ pairs...', flush=True)
    model = _strip_activation_qdq(model)

    ql_final = sum(1 for n in model.graph.node if n.op_type == 'QuantizeLinear')
    dql_final = sum(1 for n in model.graph.node if n.op_type == 'DequantizeLinear')
    print(f'[strip_qdq] Final: {ql_final} QuantizeLinear, {dql_final} DequantizeLinear', flush=True)

    print('[strip_qdq] Checking model...', flush=True)
    try:
        onnx.checker.check_model(model)
        print('[strip_qdq] Model check passed', flush=True)
    except Exception as e:
        print(f'[strip_qdq] WARNING: model check failed: {e}', flush=True)

    onnx.save(model, str(output_path))
    print(f'[strip_qdq] Saved -> {output_path}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()
    strip_qdq(Path(args.input), Path(args.output))


if __name__ == '__main__':
    main()
