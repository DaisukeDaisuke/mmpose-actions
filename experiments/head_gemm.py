"""Lower only the two measured RTMW3D head convolutions to FP32 im2col+MatMul."""
import copy
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

TARGETS={'/head/final_layer/conv/Conv','/head/final_layer2/conv/Conv'}


def lower_score_minimum(model, target='compact_decoder/score'):
    """Select the smaller finite FP32 score using registered GPU ops.

    ORT 1.27 JSEP registers Less and Where, but not the variadic Min op.
    This placement fix is separate from the measured Conv bottleneck.
    """
    model=copy.deepcopy(model)
    nodes=[];found=False
    for node in model.graph.node:
        if node.name==target:
            if node.op_type!='Min' or len(node.input)!=2:
                raise ValueError('Expected the two-input decoder score minimum')
            selected=target+'/gpu_less'
            nodes.append(helper.make_node('Less',list(node.input),[selected],name=selected))
            nodes.append(helper.make_node('Where',[selected,*node.input],list(node.output),name=target+'/gpu_select'))
            found=True
        else:nodes.append(copy.deepcopy(node))
    if not found:raise ValueError('Decoder score minimum was not found')
    del model.graph.node[:];model.graph.node.extend(nodes)
    onnx.checker.check_model(model)
    return model


def lower_head_convolutions(model, splits=1, targets=TARGETS):
    original=model
    model=onnx.shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True)
    shapes={v.name:[d.dim_value for d in v.type.tensor_type.shape.dim] for v in [*model.graph.input,*model.graph.value_info,*model.graph.output]}
    tensors={v.name:v for v in model.graph.initializer}
    new_nodes=[]
    changed=[]
    for node in model.graph.node:
        if node.name not in targets:
            new_nodes.append(copy.deepcopy(node))
            continue
        if node.op_type!='Conv':
            raise ValueError('Target must be a Conv: '+node.name)
        attrs={a.name:helper.get_attribute_value(a) for a in node.attribute}
        if attrs.get('group',1)!=1 or attrs.get('auto_pad',b'NOTSET')!=b'NOTSET':
            raise ValueError('Only explicit-padding ungrouped target convolutions are supported')
        shape=shapes[node.input[0]]
        if len(shape)!=4 or not all(d>0 for d in shape):
            raise ValueError('Target requires a fully static NCHW input: '+str(shape))
        weight=numpy_helper.to_array(tensors[node.input[1]])
        if weight.dtype!=np.float32 or weight.ndim!=4:
            raise ValueError('FP32 convolution weights are required')
        batch,channels,height,width=shape
        out_channels,in_channels,kh,kw=weight.shape
        if in_channels!=channels:
            raise ValueError('Input channel mismatch')
        sh,sw=attrs.get('strides',[1,1])
        dh,dw=attrs.get('dilations',[1,1])
        pt,pl,pb,pr=attrs.get('pads',[0,0,0,0])
        oh=(height+pt+pb-dh*(kh-1)-1)//sh+1
        ow=(width+pl+pr-dw*(kw-1)-1)//sw+1
        shared=channels*kh*kw
        if splits<1 or shared%splits:
            raise ValueError('Split count must divide the reduction dimension')
        prefix=node.name+'/explicit_gemm/'

        def constant(name,value,dtype=np.int64):
            key=prefix+name
            model.graph.initializer.append(numpy_helper.from_array(np.asarray(value,dtype=dtype),key))
            return key

        def operation(kind,inputs,name,**attributes):
            key=prefix+name
            new_nodes.append(helper.make_node(kind,inputs,[key],name=key,**attributes))
            return key

        source=node.input[0]
        if any([pt,pl,pb,pr]):
            source=operation('Pad',[source,constant('pads',[0,0,pt,pl,0,0,pb,pr]),constant('zero',0,np.float32)],'padded',mode='constant')
        axes=constant('spatial_axes',[2,3])
        steps=constant('spatial_steps',[sh,sw])
        extra_axis=constant('kernel_axis',[2])
        patches=[]
        for y in range(kh):
            for x in range(kw):
                start=[y*dh,x*dw]
                stop=[start[0]+oh*sh,start[1]+ow*sw]
                patch=operation('Slice',[source,constant(f'start_{y}_{x}',start),constant(f'stop_{y}_{x}',stop),axes,steps],f'patch_{y}_{x}')
                patches.append(operation('Unsqueeze',[patch,extra_axis],f'patch_axis_{y}_{x}'))
        # [N,C,KH*KW,OH,OW] preserves the original C,KH,KW reduction order.
        columns=operation('Concat',patches,'columns',axis=2)
        columns=operation('Reshape',[columns,constant('column_shape',[batch,shared,oh*ow])],'column_matrix')
        columns=operation('Transpose',[columns],'spatial_matrix',perm=[0,2,1])
        matrix=weight.reshape(out_channels,shared).T.copy()
        if splits>1:
            columns=operation('Reshape',[columns,constant('split_shape',[batch,oh*ow,splits,shared//splits])],'split_matrix')
            columns=operation('Transpose',[columns],'split_batch',perm=[0,2,1,3])
            matrix=matrix.reshape(splits,shared//splits,out_channels)
        product=operation('MatMul',[columns,constant('weights',matrix,np.float32)],'product')
        if splits>1:
            product=operation('ReduceSum',[product,constant('sum_axis',[1])],'split_sum',keepdims=0)
        if len(node.input)>2 and node.input[2]:
            product=operation('Add',[product,node.input[2]],'bias')
        product=operation('Transpose',[product],'channels_first',perm=[0,2,1])
        new_nodes.append(helper.make_node('Reshape',[product,constant('output_shape',[batch,out_channels,oh,ow])],list(node.output),name=prefix+'restore_nchw'))
        changed.append({'node':node.name,'inputShape':shape,'weightShape':list(weight.shape),'outputShape':[batch,out_channels,oh,ow],
                        'kernel':[kh,kw],'pads':[pt,pl,pb,pr],'strides':[sh,sw],'dilations':[dh,dw],'splits':splits,
                        'convolutionFLOPs':2*batch*oh*ow*out_channels*shared,'columnBytes':batch*oh*ow*shared*4})
    if {v['node'] for v in changed}!=set(targets):
        raise ValueError('Every explicitly selected convolution must be found')
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    used={name for n in model.graph.node for name in n.input}
    kept=[copy.deepcopy(t) for t in model.graph.initializer if t.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    # Verify the two transformations never round any parameter or change the IO contract.
    for tensor in original.graph.initializer:
        if tensor.name in used:
            stored=next(t for t in model.graph.initializer if t.name==tensor.name)
            if stored.SerializeToString()!=tensor.SerializeToString():
                raise ValueError('An unmodified initializer changed')
    onnx.checker.check_model(model)
    return model,changed
