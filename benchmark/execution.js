export const MODES = [
  {id:'standard',label:'標準 NCHW',layout:'NCHW',gpuIO:false,capture:false},
  {id:'gpu-io',label:'GPUバッファ再利用 NCHW',layout:'NCHW',gpuIO:true,capture:false},
  {id:'nhwc',label:'GPUバッファ再利用 NHWC',layout:'NHWC',gpuIO:true,capture:false},
  {id:'capture',label:'Graph Capture NHWC',layout:'NHWC',gpuIO:true,capture:true},
];
export const GROUPS={body:Array.from({length:23},(_,i)=>i),hips:[11,12],knees:[13,14],ankles:[15,16],feet:[17,18,19,20,21,22]};
export function percentile(values,p){const a=[...values].sort((x,y)=>x-y);return a.length?a[Math.floor((a.length-1)*p)]:null;}
export function compare(observed,reference,ids){
  const xy=[],z=[];let lost=0;
  for(let frame=0;frame<observed.length;frame++)for(const i of ids){
    const ref=reference[frame],out=observed[frame];if(ref.scores[i]<=.3)continue;
    xy.push(Math.hypot(out.points[i*3]-ref.points[i*3],out.points[i*3+1]-ref.points[i*3+1]));
    z.push(Math.abs(out.points[i*3+2]-ref.points[i*3+2]));if(out.scores[i]<=.3)lost++;
  }
  return {observations:xy.length,xyP95_px:percentile(xy,.95),zP95_m:percentile(z,.95),confidenceLossCount:lost};
}
export function qualityGate(parts){
  return Object.keys(GROUPS).every(name=>{const p=parts[name];return p?.observations>0&&p.xyP95_px<=3&&p.zP95_m<=.05&&p.confidenceLossCount===0;});
}
export function validateOutput(points,scores){
  if(points.length!==399||scores.length!==133||points.some(v=>!Number.isFinite(v))||scores.some(v=>!Number.isFinite(v)))throw Error('全133関節の出力契約または有限値検査に失敗');
}
// Fixed GPU input/output addresses are required by graph replay. Read fresh
// buffers directly: Tensor.getData() changes its location to CPU and caches data.
export async function createExecution(runtime,modelBytes,mode){
  const options={executionProviders:[{name:'webgpu',preferredLayout:mode.layout}],graphOptimizationLevel:'all'};
  if(mode.gpuIO)options.preferredOutputLocation='gpu-buffer';
  if(mode.capture){options.enableGraphCapture=true;options.extra={session:{disable_cpu_ep_fallback:'1'}};}
  let session,device,input,fetches,inputBuffer,staging;
  const buffers=[];
  const release=async()=>{
    try{if(session)await session.release();}
    finally{session=null;input?.dispose();input=null;for(const t of Object.values(fetches??{}))t.dispose();fetches=null;for(const b of buffers.splice(0))b.destroy();}
  };
  try{
    session=await runtime.InferenceSession.create(modelBytes,options);
    device=await runtime.env.webgpu.device;
    if(mode.gpuIO){
      const make=(size,usage)=>{const b=device.createBuffer({size:Math.ceil(size/16)*16,usage});buffers.push(b);return b;};
      const usage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST;
      inputBuffer=make(3*384*288*4,usage);
      input=runtime.Tensor.fromGpuBuffer(inputBuffer,{dataType:'float32',dims:[1,3,384,288]});
      fetches={
        keypoints:runtime.Tensor.fromGpuBuffer(make(399*4,usage),{dataType:'float32',dims:[1,133,3]}),
        scores:runtime.Tensor.fromGpuBuffer(make(133*4,usage),{dataType:'float32',dims:[1,133]}),
      };
      staging=make(532*4,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
    }
    return {session,device,release,async run(data){
      const started=performance.now();let points,scores,runMs,readbackMs=0;
      if(mode.gpuIO){
        device.queue.writeBuffer(inputBuffer,0,data);
        await session.run({input},fetches);runMs=performance.now()-started;
        const read=performance.now(),encoder=device.createCommandEncoder();
        encoder.copyBufferToBuffer(fetches.keypoints.gpuBuffer,0,staging,0,399*4);
        encoder.copyBufferToBuffer(fetches.scores.gpuBuffer,0,staging,399*4,133*4);
        device.queue.submit([encoder.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        try{const packed=new Float32Array(staging.getMappedRange().slice(0));points=packed.slice(0,399);scores=packed.slice(399);}
        finally{staging.unmap();}
        readbackMs=performance.now()-read;
      }else{
        const tensor=new runtime.Tensor('float32',data,[1,3,384,288]);let out;
        try{out=await session.run({input:tensor});points=out.keypoints.data.slice();scores=out.scores.data.slice();runMs=performance.now()-started;}
        finally{for(const t of Object.values(out??{}))t.dispose();tensor.dispose();}
      }
      validateOutput(points,scores);
      return {points,scores,totalMs:performance.now()-started,runMs,readbackMs};
    }};
  }catch(error){await release().catch(()=>{});throw error;}
}
export function summarizeProfile(events){
  const entries=new Map();let totalNs=0;
  for(const e of events){
    const duration=e.endTime-e.startTime;if(!Number.isFinite(duration)||duration<0)continue;
    totalNs+=duration;const key=[e.kernelType,e.kernelName,e.programName].join('|');
    if(!entries.has(key))entries.set(key,{kernelType:e.kernelType,kernelName:e.kernelName,programName:e.programName,dispatches:0,totalMs:0});
    const item=entries.get(key);item.dispatches++;item.totalMs+=duration/1e6;
  }
  return {available:events.length>0,dispatches:events.length,summedKernelMs:totalNs/1e6,topKernels:[...entries.values()].sort((a,b)=>b.totalMs-a.totalMs).slice(0,25),
    note:'Separate instrumented run; summed dispatch durations do not include CPU execution, transfers or queue gaps.'};
}
