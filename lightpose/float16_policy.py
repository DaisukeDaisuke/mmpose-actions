"""RTMPose FP16 conversion with native subnormal rounding, without a 1e-7 floor."""
import copy
import onnx
from onnx import TensorProto


def selective_float16(model, selected_names, blocked_ops):
    """Keep excluded layers' original FP32 parameters, not rounded-and-upcast copies.

    onnxconverter-common converts every initializer, even when its consuming
    node is blocked. Restore excluded inputs explicitly and prune dead casts.
    """
    from onnxconverter_common import float16
    selected_names = set(selected_names)
    blocked_ops = set(blocked_ops)
    result = float16.convert_float_to_float16(copy.deepcopy(model), keep_io_types=True,
        min_positive_val=0.0, max_finite_val=65504.0,
        op_block_list=sorted(blocked_ops), node_block_list=[n.name for n in model.graph.node if n.name not in selected_names])
    originals = {t.name: t for t in model.graph.initializer}
    converted_nodes = {node.name: node for node in result.graph.node}
    restored = {}
    for original in model.graph.node:
        if original.name in selected_names and original.op_type not in blocked_ops:
            continue
        node = converted_nodes[original.name]
        for index, name in enumerate(original.input):
            tensor = originals.get(name)
            if tensor is None or tensor.data_type != TensorProto.FLOAT:
                continue
            if name not in restored:
                clone = copy.deepcopy(tensor)
                clone.name = 'preserved_fp32/' + name
                result.graph.initializer.append(clone)
                restored[name] = clone.name
            node.input[index] = restored[name]
        if original.op_type == 'Constant':
            del node.attribute[:]
            node.attribute.extend(copy.deepcopy(original.attribute))
    producers = {output: i for i, node in enumerate(result.graph.node) for output in node.output}
    needed, keep, pending = set(), set(), [v.name for v in result.graph.output]
    while pending:
        name = pending.pop()
        if name in needed:
            continue
        needed.add(name)
        if name in producers:
            index = producers[name]
            keep.add(index)
            pending.extend(result.graph.node[index].input)
    nodes = [copy.deepcopy(n) for i, n in enumerate(result.graph.node) if i in keep]
    tensors = [copy.deepcopy(t) for t in result.graph.initializer if t.name in needed]
    info = [copy.deepcopy(v) for v in result.graph.value_info if v.name in needed]
    del result.graph.node[:]
    result.graph.node.extend(nodes)
    del result.graph.initializer[:]
    result.graph.initializer.extend(tensors)
    del result.graph.value_info[:]
    result.graph.value_info.extend(info)
    onnx.checker.check_model(result)
    return result
