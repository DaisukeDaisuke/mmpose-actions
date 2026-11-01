import sys
import unittest
from pathlib import Path
import numpy as np
import onnxruntime as ort
from onnx import helper,numpy_helper,TensorProto
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from head_gemm import lower_head_convolutions


class HeadGemmTests(unittest.TestCase):
    def compare_case(self,channels,kernel,pads,strides,dilations,splits):
        rng=np.random.default_rng(951)
        data=rng.normal(size=(1,channels,9,11)).astype(np.float32)
        weight=rng.normal(size=(5,channels,kernel,kernel)).astype(np.float32)
        bias=rng.normal(size=(5,)).astype(np.float32)
        graph=helper.make_graph([helper.make_node('Conv',['input','weight','bias'],['output'],name='target',pads=pads,strides=strides,dilations=dilations)],'test',
            [helper.make_tensor_value_info('input',TensorProto.FLOAT,list(data.shape))],
            [helper.make_tensor_value_info('output',TensorProto.FLOAT,[None]*4)],
            [numpy_helper.from_array(weight,'weight'),numpy_helper.from_array(bias,'bias')])
        model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)]);model.ir_version=9
        changed,report=lower_head_convolutions(model,splits,{'target'})
        options=ort.SessionOptions();options.intra_op_num_threads=1
        expected=ort.InferenceSession(model.SerializeToString(),options,providers=['CPUExecutionProvider']).run(None,{'input':data})[0]
        actual=ort.InferenceSession(changed.SerializeToString(),options,providers=['CPUExecutionProvider']).run(None,{'input':data})[0]
        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-4)
        self.assertEqual(report[0]['weightShape'],list(weight.shape))
        self.assertFalse(any(n.op_type=='Conv' for n in changed.graph.node))
        self.assertTrue(all(t.data_type in [TensorProto.FLOAT,TensorProto.INT64] for t in changed.graph.initializer))

    def test_seven_by_seven_padding_and_bias(self):
        self.compare_case(4,7,[3,3,3,3],[1,1],[1,1],1)

    def test_stride_and_dilation(self):
        self.compare_case(4,3,[2,2,2,2],[2,2],[2,2],4)

    def test_eight_way_reduction(self):
        self.compare_case(8,3,[1,1,1,1],[1,1],[1,1],8)

    def test_asymmetric_padding(self):
        self.compare_case(4,3,[0,2,1,0],[1,2],[1,1],1)


if __name__=='__main__':
    unittest.main()
