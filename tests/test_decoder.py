import sys
from pathlib import Path
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from onnx_graph import append_simcc_decoder, selective_float16, widen_float16_for_cpu_reference
from onnx import numpy_helper


class DecoderGraphTests(unittest.TestCase):
    def test_unselected_layer_keeps_exact_fp32_weights(self):
        a = np.array([[1.0002, .1], [.2, 1.0003]], np.float32)
        b = np.array([[1.0002, .3], [.4, 1.0003]], np.float32)
        graph = helper.make_graph([
            helper.make_node('MatMul', ['input', 'a'], ['features'], name='/backbone/MatMul'),
            helper.make_node('MatMul', ['features', 'b'], ['output'], name='/head/MatMul')],
            'selection-test', [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2])],
            initializer=[numpy_helper.from_array(a, 'a'), numpy_helper.from_array(b, 'b')])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
        model.ir_version = 9
        converted = selective_float16(model, {'/backbone/MatMul'}, set())
        head = next(n for n in converted.graph.node if n.name == '/head/MatMul')
        stored = next(t for t in converted.graph.initializer if t.name == head.input[1])
        self.assertEqual(stored.data_type, TensorProto.FLOAT)
        np.testing.assert_array_equal(numpy_helper.to_array(stored), b)
        evaluated = widen_float16_for_cpu_reference(converted)
        session = ort.InferenceSession(evaluated.SerializeToString(), providers=['CPUExecutionProvider'])
        source = np.array([[.5, 1]], np.float32)
        observed = session.run(None, {'input': source})[0]
        np.testing.assert_allclose(observed, source @ a.astype(np.float16).astype(np.float32) @ b, atol=1e-6)

    def test_exact_maximum_ties_missing_and_depth_denominator(self):
        names = ['x', 'y', 'z']
        shapes = [[1, 133, n] for n in [576, 768, 576]]
        graph = helper.make_graph([], 'decoder-test',
                                  [helper.make_tensor_value_info(n, TensorProto.FLOAT, s) for n, s in zip(names, shapes)],
                                  [helper.make_tensor_value_info(n, TensorProto.FLOAT, s) for n, s in zip(names, shapes)])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
        model.ir_version = 9
        compact = append_simcc_decoder(model)
        session = ort.InferenceSession(compact.SerializeToString(), providers=['CPUExecutionProvider'])
        x, y, z = [np.full(shape, -1, np.float32) for shape in shapes]
        x[0, :, 288] = .7
        y[0, :, 384] = .6
        z[0, :, 288] = 1
        z[0, 1, 432] = 2
        x[0, 2, 288] = 0  # no positive x response -> all coordinates use missing index -1
        x[0, 3, 12] = .7  # tie: pick the first index, matching NumPy/PyTorch
        points, scores = session.run(None, dict(zip(names, [x, y, z])))
        self.assertEqual(points.shape, (1, 133, 3))
        np.testing.assert_allclose(points[0, 0], [144, 192, 0], atol=1e-6)
        self.assertAlmostEqual(float(points[0, 1, 2]), 2.1744869 / 2, places=6)
        self.assertEqual(float(points[0, 2, 0]), -.5)
        self.assertEqual(float(points[0, 3, 0]), 6)
        self.assertAlmostEqual(float(scores[0, 0]), .6, places=6)
        self.assertEqual(float(scores[0, 2]), 0)


if __name__ == '__main__':
    unittest.main()
