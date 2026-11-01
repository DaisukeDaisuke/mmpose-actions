import test from 'node:test';
import assert from 'node:assert/strict';
import {MODES,GROUPS,createExecution,qualityGate,summarizeProfile,compare} from '../benchmark/execution.js';
import {poseWorldResult} from '../benchmark/mediapipe-adapter.js';

globalThis.GPUBufferUsage={STORAGE:1,COPY_SRC:2,COPY_DST:4,MAP_READ:8};
globalThis.GPUMapMode={READ:1};
function fakeRuntime(){
  const calls=[],buffers=[],options=[];let releases=0;
  const device={features:new Set(),createBuffer({size}){
    const b={bytes:new Uint8Array(size),destroyed:false,destroy(){this.destroyed=true;},async mapAsync(){calls.push('map');await Promise.resolve();},getMappedRange(){return this.bytes.buffer;},unmap(){calls.push('unmap');}};buffers.push(b);return b;
  },queue:{writeBuffer(b,offset,data){b.bytes.set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength),offset);},submit(commands){for(const command of commands)command();}},createCommandEncoder(){
    const commands=[];return {copyBufferToBuffer(from,sourceOffset,to,destOffset,size){commands.push(()=>to.bytes.set(from.bytes.subarray(sourceOffset,sourceOffset+size),destOffset));},finish(){return()=>commands.forEach(f=>f());}};
  }};
  class Tensor{
    constructor(type,data,dims){Object.assign(this,{type,data,dims});}
    dispose(){this.disposed=true;}
    static fromGpuBuffer(gpuBuffer,settings){const t=new Tensor(settings.dataType,null,settings.dims);t.gpuBuffer=gpuBuffer;return t;}
  }
  const runtime={Tensor,env:{webgpu:{device}},InferenceSession:{async create(bytes,settings){
    options.push(settings);return {async run({input},fetches){
      const value=input.gpuBuffer?new Float32Array(input.gpuBuffer.bytes.buffer)[0]:input.data[0];
      const points=new Float32Array(399).fill(value),scores=new Float32Array(133).fill(1);
      calls.push({input,fetches});
      if(fetches){fetches.keypoints.gpuBuffer.bytes.set(new Uint8Array(points.buffer));fetches.scores.gpuBuffer.bytes.set(new Uint8Array(scores.buffer));return fetches;}
      return {keypoints:new Tensor('float32',points,[1,133,3]),scores:new Tensor('float32',scores,[1,133])};
    },async release(){releases++;}};
  }}};
  return {runtime,calls,buffers,options,get releases(){return releases;}};
}
test('GPU replay reuses addresses but reads fresh coordinates every run',async()=>{
  const mock=fakeRuntime(),runner=await createExecution(mock.runtime,new Uint8Array(),MODES.find(m=>m.id==='capture'));
  const first=await runner.run(new Float32Array([1])),second=await runner.run(new Float32Array([9]));
  assert.equal(first.points[0],1);assert.equal(second.points[0],9);assert.equal(first.points[398],1);assert.equal(second.scores[132],1);
  const runs=mock.calls.filter(v=>typeof v==='object');
  assert.equal(runs[0].input,runs[1].input);assert.equal(runs[0].fetches,runs[1].fetches);
  assert.equal(mock.calls.filter(v=>v==='map').length,2);
  assert.equal(mock.options[0].enableGraphCapture,true);
  assert.equal(mock.options[0].extra.session.disable_cpu_ep_fallback,'1');
  await runner.release();assert.equal(mock.releases,1);assert.ok(mock.buffers.every(b=>b.destroyed));
});
test('CPU IO baseline returns independent copies and releases its tensors',async()=>{
  const mock=fakeRuntime(),runner=await createExecution(mock.runtime,new Uint8Array(),MODES[0]);
  const out=await runner.run(new Float32Array([3]));assert.equal(out.points.length,399);assert.equal(out.points[0],3);
  assert.equal(mock.calls[0].input.disposed,true);assert.equal(mock.options[0].preferredOutputLocation,undefined);
  await runner.release();assert.equal(mock.releases,1);
});
test('a nonfinite GPU output cannot become a successful measurement',async()=>{
  const mock=fakeRuntime(),runner=await createExecution(mock.runtime,new Uint8Array(),MODES[1]);
  await assert.rejects(runner.run(new Float32Array([NaN])),/有限値/);await runner.release();
  assert.ok(mock.buffers.every(b=>b.destroyed));
});
test('allocation failure releases already acquired resources',async()=>{
  const mock=fakeRuntime();mock.runtime.Tensor.fromGpuBuffer=()=>{throw Error('allocation failure');};
  await assert.rejects(createExecution(mock.runtime,new Uint8Array(),MODES[1]),/allocation failure/);
  assert.equal(mock.releases,1);assert.ok(mock.buffers.every(b=>b.destroyed));
});
test('foot depth drift and missing observations independently fail the full-body gate',()=>{
  const parts=Object.fromEntries(Object.keys(GROUPS).map(name=>[name,{observations:6,xyP95_px:0,zP95_m:0,confidenceLossCount:0}]));
  assert.equal(qualityGate(parts),true);parts.feet.zP95_m=.3926;assert.equal(qualityGate(parts),false);
  parts.feet.zP95_m=0;parts.ankles.observations=0;assert.equal(qualityGate(parts),false);
});
test('drift uses the fixed baseline confidence mask and includes lost predictions',()=>{
  const points=new Float32Array(399),scores=new Float32Array(133).fill(1),observed=points.slice();observed[15*3]=300;
  const lost=scores.slice();lost[15]=0;
  const out=compare([{points:observed,scores:lost}],[{points,scores}],[15]);
  assert.equal(out.xyP95_px,300);assert.equal(out.confidenceLossCount,1);
});
test('GPU profiling nanoseconds are converted to ms and never treated as wall latency',()=>{
  const summary=summarizeProfile([{kernelType:'Conv',kernelName:'a',programName:'p',startTime:1e6,endTime:3e6},{kernelType:'Conv',kernelName:'a',programName:'p',startTime:5e6,endTime:8e6}]);
  assert.equal(summary.available,true);assert.equal(summary.summedKernelMs,5);assert.equal(summary.topKernels[0].dispatches,2);
  assert.equal(summarizeProfile([]).available,false);
});
test('Holistic world alias is accepted only for its verified pinned SDK',()=>{
  const points=Array.from({length:33},()=>({x:0,y:0,z:1}));
  assert.equal(poseWorldResult({za:points},'holistic','0.5.1675471629').points,points);
  assert.equal(poseWorldResult({za:points},'holistic','future').points.length,0);
  assert.equal(poseWorldResult({za:points},'pose','0.5.1675471629').points.length,0);
  assert.equal(poseWorldResult({poseWorldLandmarks:points},'pose','0.5.1675469404').field,'poseWorldLandmarks');
});
