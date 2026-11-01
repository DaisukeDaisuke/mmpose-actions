"""Standard ONNX graph transforms; no model-specific architecture reimplementation."""
import copy

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def append_simcc_decoder(model):
    """Return model-space XY pixels, relative Z metres, and official raw scores.

    The three large SimCC tensors remain on the execution device. The public
    outputs are only 532 float32 elements (2128 bytes) for 133 joints.
    """
    model = copy.deepcopy(model)
    graph = model.graph
    if len(graph.output) != 3:
        raise ValueError('Expected three SimCC outputs')
    names = [v.name for v in graph.output]
    shapes = [[d.dim_value for d in v.type.tensor_type.shape.dim] for v in graph.output]
    if shapes != [[1, 133, 576], [1, 133, 768], [1, 133, 576]]:
        raise ValueError(f'Unexpected SimCC shapes: {shapes}')
    prefix = 'compact_decoder/'
    if any(n.name.startswith(prefix) for n in graph.node):
        raise ValueError('Decoder already exists')

    def const(name, value, dtype=np.float32):
        key = prefix + name
        graph.initializer.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), key))
        return key

    def op(kind, inputs, name, **attrs):
        key = prefix + name
        graph.node.append(helper.make_node(kind, inputs, [key], name=key, **attrs))
        return key

    zero, negative, two = const('zero', 0), const('negative', -1), const('split', 2)
    depth_half, one, z_range = const('depth_half', 144), const('one', 1), const('z_range', 2.1744869)
    axes = const('unsqueeze_axes', [-1], np.int64)
    max_x = op('ReduceMax', [names[0]], 'max_x', axes=[2], keepdims=0)
    max_y = op('ReduceMax', [names[1]], 'max_y', axes=[2], keepdims=0)
    score = op('Min', [max_x, max_y], 'score')
    valid = op('Greater', [score, zero], 'valid')
    coords = []
    for axis, name in zip('xyz', names):
        index = op('ArgMax', [name], axis + '_index', axis=2, keepdims=0, select_last_index=0)
        index = op('Cast', [index], axis + '_float', to=TensorProto.FLOAT)
        index = op('Where', [valid, index, negative], axis + '_valid')
        value = op('Div', [index, two], axis + '_pixel')
        if axis == 'z':
            value = op('Div', [value, depth_half], 'z_scaled')
            value = op('Sub', [value, one], 'z_centered')
            value = op('Mul', [value, z_range], 'z_metres')
        coords.append(op('Unsqueeze', [value, axes], axis + '_column'))
    graph.node.append(helper.make_node('Concat', coords, ['keypoints'], name=prefix + 'keypoints', axis=2))
    graph.node.append(helper.make_node('Identity', [score], ['scores'], name=prefix + 'scores'))
    del graph.output[:]
    graph.output.extend([
        helper.make_tensor_value_info('keypoints', TensorProto.FLOAT, [1, 133, 3]),
        helper.make_tensor_value_info('scores', TensorProto.FLOAT, [1, 133]),
    ])
    onnx.checker.check_model(model)
    return model


def widen_float16_for_cpu_reference(model):
    """Test rounded weights/constants with f32 compute on CPU, not native fp16.

    This does not measure fp16 activation rounding, GPU shader correctness, or
    GPU latency. Those limitations are recorded in each validation report.
    """
    model = copy.deepcopy(model)

    def tensor(t):
        if t.data_type == TensorProto.FLOAT16:
            converted = numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float32), t.name)
            t.CopyFrom(converted)

    def graph(g):
        for t in g.initializer:
            tensor(t)
        for info in [*g.input, *g.output, *g.value_info]:
            if info.type.tensor_type.elem_type == TensorProto.FLOAT16:
                info.type.tensor_type.elem_type = TensorProto.FLOAT
        for node in g.node:
            for attr in node.attribute:
                if node.op_type == 'Cast' and attr.name == 'to' and attr.i == TensorProto.FLOAT16:
                    attr.i = TensorProto.FLOAT
                elif attr.type == onnx.AttributeProto.TENSOR:
                    tensor(attr.t)
                elif attr.type == onnx.AttributeProto.GRAPH:
                    graph(attr.g)

    graph(model.graph)
    onnx.checker.check_model(model)
    return model
