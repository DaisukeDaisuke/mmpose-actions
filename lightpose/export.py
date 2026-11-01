"""Actions-only RTMPose-t/s COCO-WholeBody133 export; no retraining."""
import argparse, hashlib, json, sys, shutil, urllib.request
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
from onnxconverter_common import float16
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

p=argparse.ArgumentParser()
p.add_argument('stage',choices=['export','quantize'])
p.add_argument('--policy',default='fp32')
p.add_argument('--model',choices=['rtmpose-t','rtmpose-s'],required=True)
a=p.parse_args()
root=Path.cwd()
model_id=a.model+'-256x192'
sys.path.insert(0,str(root/'scripts'))
from onnx_graph import selective_float16, widen_float16_for_cpu_reference
out=root/'release'
out.mkdir(exist_ok=True)
base=root/'base'
base.mkdir(exist_ok=True)
def session(model):
    options=ort.SessionOptions(); options.intra_op_num_threads=2
    return ort.InferenceSession(model,options,providers=['CPUExecutionProvider'])
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def xy(outputs):
    return np.stack([v.argmax(-1)/2 for v in outputs],-1)
def crops(ids):
    upstream=root/'upstream'
    annotations=json.loads((upstream/'tests/data/coco/test_coco.json').read_text(encoding='utf-8'))
    images={v['id']:v for v in annotations['images']}
    result=[]
    for image_id in ids:
        ann=next(v for v in annotations['annotations'] if v['image_id']==image_id and v.get('num_keypoints',0)>=14)
        im=Image.open(upstream/'tests/data/coco'/images[image_id]['file_name']).convert('RGB')
        x,y,w,h=ann['bbox']; cx=x+w/2; cy=y+h/2; w*=1.25; h*=1.25
        if w/h>.75: h=w/.75
        else: w=h*.75
        patch=im.transform((192,256),Image.Transform.AFFINE,(w/192,0,cx-w/2,0,h/256,cy-h/2),Image.Resampling.BILINEAR)
        result.append(((np.asarray(patch,np.float32)-[123.675,116.28,103.53])/[58.395,57.12,57.375]).transpose(2,0,1)[None].astype(np.float32))
    return result

if a.stage=='export':
    import torch
    from torch import nn
    from mmpose.apis import init_model
    torch.set_num_threads(2)
    size=a.model.rsplit('-',1)[1]
    config=root/f'upstream/projects/rtmpose/rtmpose/wholebody_2d_keypoint/rtmpose-{size}_8xb64-270e_coco-wholebody-256x192.py'
    checkpoint=root/'.cache/checkpoints'/f'{model_id}.pth'
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    checksum=checkpoint.with_suffix('.sha256')
    if checkpoint.is_file() and (not checksum.is_file() or digest(checkpoint)!=checksum.read_text(encoding='utf-8').strip()):
        checkpoint.unlink()
    filename={'rtmpose-t':'rtmpose-t_simcc-ucoco_dw-ucoco_270e-256x192-dcf277bf_20230728.pth','rtmpose-s':'rtmpose-s_simcc-ucoco_dw-ucoco_270e-256x192-3fd922c8_20230728.pth'}[a.model]
    source='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/'+filename
    if not checkpoint.is_file():
        temporary=checkpoint.with_suffix('.part')
        with urllib.request.urlopen(source,timeout=180) as response,temporary.open('wb') as destination:
            shutil.copyfileobj(response,destination)
        if temporary.stat().st_size<1000000: raise RuntimeError('Incomplete checkpoint download')
        temporary.replace(checkpoint)
        checksum.write_text(digest(checkpoint),encoding='utf-8')
        print('Official checkpoint downloaded once; SHA-256 recorded for cache reuse')
    else: print('Reusing checksum-verified checkpoint cache')
    url=str(checkpoint)
    from mmengine.config import Config
    cfg=Config.fromfile(str(config))
    cfg.model.backbone.init_cfg=None
    model=init_model(cfg,url,device='cpu').eval()
    class Network(nn.Module):
        def __init__(self,model):
            super().__init__(); self.backbone=model.backbone; self.neck=model.neck if model.with_neck else None; self.head=model.head
        def forward(self,image):
            features=self.backbone(image)
            if self.neck is not None: features=self.neck(features)
            return self.head(features)
    net=Network(model).eval()
    validation=crops([40083,197388]); calibration=crops([785,196141])
    path=base/'simcc-fp32.onnx'
    with torch.inference_mode():
        expected=[list(net(torch.from_numpy(v))) for v in validation]
        torch.onnx.export(net,torch.from_numpy(validation[0]),str(path),opset_version=17,input_names=['input'],output_names=['simcc_x','simcc_y'],do_constant_folding=True)
    onnx.checker.check_model(str(path))
    runtime=session(str(path)); observed=[]
    for inp,ref in zip(validation,expected):
        values=runtime.run(None,{'input':inp})
        assert [list(v.shape) for v in values]==[[1,133,384],[1,133,512]]
        for x,y in zip(values,ref): np.testing.assert_allclose(x,y.numpy(),rtol=.003,atol=.0005)
        observed.append(xy(values))
    np.savez_compressed(base/'data.npz',**{f'input_{i}':v for i,v in enumerate(validation)},**{f'xy_{i}':v for i,v in enumerate(observed)},**{f'calibration_{i}':v for i,v in enumerate(calibration)})
    (base/'provenance.json').write_text(json.dumps({'model':model_id,'checkpoint':url,'checkpointSource':source,'checkpointSha256':digest(checkpoint),'upstream':'https://github.com/open-mmlab/mmpose','outputs':'2D133; depth must be estimated by calibrated geometry, bone constraints and time continuity','validationImages':[40083,197388],'calibrationImages':[785,196141]}),encoding='utf-8')
    print(a.model+' WholeBody133 2D export complete; no depth output is fabricated')
else:
    policy=a.policy
    model=onnx.load(str(base/'simcc-fp32.onnx'))
    data=np.load(base/'data.npz')
    groups=['backbone'] if policy.endswith('-backbone') else ['neck','head'] if policy.endswith('-neck-head') else ['neck'] if policy.endswith('-neck') else ['head'] if policy.endswith('-head') else ['backbone','neck','head']
    selected=[n.name for n in model.graph.node if any('/'+g+'/' in n.name for g in groups)]
    target=out/f'{model_id}-{policy}.onnx'
    if policy=='fp32': onnx.save(model,str(target))
    elif policy.startswith('fp16'):
        assert selected,groups
        variant=selective_float16(model,set(selected),sorted(set(float16.DEFAULT_OP_BLOCK_LIST)|{'ReduceL2','ReduceSum','Sqrt','Pow'}))
        onnx.save(variant,str(target))
    else:
        selected=[n.name for n in model.graph.node if n.name in selected and n.op_type in ['Conv','MatMul','Gemm']]
        assert selected,groups
        class Reader(CalibrationDataReader):
            def __init__(self): self.iterator=iter([{'input':data[k]} for k in data.files if k.startswith('calibration_')])
            def get_next(self): return next(self.iterator,None)
        quantize_static(str(base/'simcc-fp32.onnx'),str(target),Reader(),quant_format=QuantFormat.QDQ,activation_type=QuantType.QUInt8,weight_type=QuantType.QInt8,per_channel=True,nodes_to_quantize=selected,extra_options={'WeightSymmetric':True})
    onnx.checker.check_model(str(target))
    variant=onnx.load(str(target))
    runtime=session(widen_float16_for_cpu_reference(variant).SerializeToString() if policy.startswith('fp16') else str(target))
    reports=[]
    for i in range(2):
        values=runtime.run(None,{'input':data[f'input_{i}']})
        assert all(np.isfinite(v).all() for v in values)
        coords=xy(values); error=np.linalg.norm(coords-data[f'xy_{i}'],axis=-1)[0]
        reports.append({name:float(np.quantile(error[ids],.95)) for name,ids in {'body':list(range(17)),'hips':[11,12],'knees':[13,14],'ankles':[15,16],'feet':list(range(17,23))}.items()})
    report={'model':a.model,'file':target.name,'policy':policy,'bytes':target.stat().st_size,'sha256':digest(target),'selectedNodes':selected,'xyP95CropPixels':reports,'smallFixtureDriftPassed':all(v<=3 for r in reports for v in r.values()),'browserGPUValidated':False,'speedRanking':None,'notes':['Drift from FP32 is not ground-truth accuracy.','FP16 or INT8 does not guarantee faster WebGPU inference.','All 133 joints retained, including ankles/toes/heels; layer names are neural network components, not anatomical regions.']}
    target.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))
