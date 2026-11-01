"""Build and validate the measured FP32 head optimization on GitHub Actions."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
from head_gemm import lower_head_convolutions,lower_score_minimum

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from quality import drift,passes_fullbody_gate

SOURCE_REVISION='88ee5cb728cb2ed3efaf43011c9dd4908a6504b7'
SOURCE_BASE='https://huggingface.co/RabbitDaisuke/rtmw3d-webgpu-onnx/resolve/'+SOURCE_REVISION+'/'


def download(url,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen(url,timeout=180) as response,path.open('wb') as file:
        while chunk:=response.read(1024*1024):
            file.write(chunk)


def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda:file.read(1024*1024),b''):
            result.update(chunk)
    return result.hexdigest()


def validation_inputs(upstream):
    folder=upstream/'tests/data/coco'
    data=json.loads((folder/'test_coco.json').read_text(encoding='utf-8'))
    images={item['id']:item for item in data['images']}
    for image_id in [40083,197388]:
        person=next(item for item in data['annotations'] if item['image_id']==image_id and item.get('num_keypoints',0)>=14)
        image=Image.open(folder/images[image_id]['file_name']).convert('RGB')
        x,y,w,h=person['bbox'];cx,cy=x+w/2,y+h/2;w*=1.25;h*=1.25
        if w/h>.75:h=w/.75
        else:w=h*.75
        crop=image.transform((288,384),Image.Transform.AFFINE,(w/288,0,cx-w/2,0,h/384,cy-h/2),Image.Resampling.BILINEAR)
        array=np.asarray(crop,dtype=np.float32)
        array=(array-np.asarray([123.675,116.28,103.53],np.float32))/np.asarray([58.395,57.12,57.375],np.float32)
        yield image_id,array.transpose(2,0,1)[None].copy()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--splits',type=int,choices=[0,1,8],required=True)
    parser.add_argument('--upstream',type=Path,required=True);parser.add_argument('--output',type=Path,default=Path('release/experiments'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    temporary=ROOT/'.cache/head-gemm';temporary.mkdir(parents=True,exist_ok=True)
    download(SOURCE_BASE+'models.json',temporary/'catalog.json')
    catalog=json.loads((temporary/'catalog.json').read_text(encoding='utf-8'))
    source=next(v for v in catalog['variants'] if v['id']=='rtmw3d-l-fp32')
    path=temporary/'rtmw3d-l-fp32.onnx';download(SOURCE_BASE+source['file'],path)
    if digest(path)!=source['sha256']:raise ValueError('Source ONNX hash mismatch')
    model=onnx.load(path)
    if args.splits:transformed,changed=lower_head_convolutions(model,args.splits)
    else:transformed,changed=model,[]
    transformed=lower_score_minimum(transformed)
    del model;gc.collect()
    identifier='rtmw3d-l-fp32-'+('head-gemm'+('-split8' if args.splits==8 else '') if args.splits else 'gpu-decoder')
    destination=args.output/(identifier+'.onnx');onnx.save_model(transformed,destination)
    del transformed;gc.collect();onnx.checker.check_model(str(destination))
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    baseline=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
    candidate=ort.InferenceSession(str(destination),options,providers=['CPUExecutionProvider'])
    comparisons=[]
    for image_id,data in validation_inputs(args.upstream):
        reference,reference_scores=baseline.run(None,{'input':data})
        points,scores=candidate.run(None,{'input':data})
        if points.shape!=(1,133,3) or scores.shape!=(1,133) or not np.isfinite(points).all() or not np.isfinite(scores).all():
            raise ValueError('All 133 finite 3D joints must remain')
        np.testing.assert_allclose(scores,reference_scores,rtol=2e-3,atol=2e-3)
        comparisons.append({'publicFixtureImageId':image_id,'parts':drift(points,scores,reference,reference_scores),
                            'scoreMaxAbsoluteError':float(np.max(np.abs(scores-reference_scores)))})
    passed=passes_fullbody_gate(comparisons)
    if not passed:raise ValueError('The equivalent FP32 head must pass every body/foot smoke gate')
    report={'id':identifier,'model':'rtmw3d-l','policy':identifier.removeprefix('rtmw3d-l-'),
        'file':'experiments/'+destination.name,'sizeBytes':destination.stat().st_size,'sha256':digest(destination),
        'precision':'fp32','experimental':True,'qualityPassed':passed,'browserGPUValidated':False,
        'sourceRevision':SOURCE_REVISION,'sourceModelSha256':source['sha256'],
        'upstream':'https://github.com/open-mmlab/mmpose','weightsUnchangedFP32':True,'gpuDecoderMinimum':'Less+Where',
        'input':source['input'],'outputs':source['outputs'],'outputBytesPerFrame':2128,'changedConvolutions':changed,
        'comparisons':comparisons,'referenceLatencyMs':[],
        'note':'Same trained weights and 133-point 3D contract; explicit im2col and optional batched split-K reduction. CPU conversion tests are not WebGPU performance or target-video accuracy measurements.'}
    (args.output/(identifier+'.json')).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'id':identifier,'qualityPassed':passed,'changedConvolutions':changed,'sizeBytes':report['sizeBytes']},indent=2))


if __name__=='__main__':main()
