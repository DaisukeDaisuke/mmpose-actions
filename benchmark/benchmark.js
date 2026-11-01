import {MODES,GROUPS,percentile as pct,compare,qualityGate,createExecution,summarizeProfile} from './execution.js';
const $=id=>document.getElementById(id),REPO='RabbitDaisuke/rtmw3d-webgpu-onnx';
const video=$('video'),crop=$('crop'),ctx=crop.getContext('2d',{willReadFrequently:true});
let catalog,revision,base,ort,stopped=false,results=[],videoURL,runContext,preparedInputs;
const runtimes=new Map();
const status=s=>$('status').textContent=s;
async function hash(bytes){return [...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(x=>x.toString(16).padStart(2,'0')).join('');}
async function get(url){const r=await fetch(url,{credentials:'omit',referrerPolicy:'no-referrer'});if(!r.ok)throw Error(`${r.status}: ${url}`);return r;}
async function bytes(file,sha){
  const key=base+file,cache=await caches.open('rtmw3d-comparison-v1');let r=await cache.match(key);
  if(r){const b=await r.arrayBuffer();if(await hash(b)===sha)return b;await cache.delete(key);}
  status(`取得・照合中: ${file}`);r=await get(key);const b=await r.arrayBuffer();if(await hash(b)!==sha)throw Error(`SHA-256不一致: ${file}`);
  try{await cache.put(key,new Response(b));}catch{status('キャッシュ容量不足。取得済みモデルで計測を続けます。');}return b;
}
async function loadRuntime(flavour='patched'){
  if(runtimes.has(flavour))return runtimes.get(flavour);if(!navigator.gpu)throw Error('WebGPU対応のPCブラウザで開いてください');
  const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});if(!adapter)throw Error('WebGPUアダプターを取得できません');
  const manifest=catalog.runtime,entry=manifest.files.find(f=>f.path===(flavour==='original'?manifest.originalEntrypoint:manifest.entrypoint)),glue=manifest.files.find(f=>f.path.endsWith('.jsep.mjs'));
  if(!entry)throw Error('このHFリビジョンには未パッチJSEPランタイムがありません');
  const js=URL.createObjectURL(new Blob([await bytes(entry.path,entry.sha256)],{type:'text/javascript'}));
  const module=URL.createObjectURL(new Blob([await bytes(glue.path,glue.sha256)],{type:'text/javascript'}));
  ort=await import(js);ort.env.wasm.numThreads=1;ort.env.wasm.proxy=false;
  ort.env.wasm.wasmPaths={mjs:module,wasm:base+'runtime/ort-wasm-simd-threaded.jsep.wasm'};
  ort.env.webgpu.powerPreference='high-performance';ort.env.webgpu.profiling={mode:'off',ondata:()=>{}};
  runtimes.set(flavour,ort);return ort;
}
async function seek(t){if(Math.abs(video.currentTime-t)<.001&&video.readyState>=2)return;await new Promise((resolve,reject)=>{const timeout=setTimeout(()=>reject(Error('動画シークがタイムアウトしました')),15000);video.addEventListener('seeked',()=>{clearTimeout(timeout);resolve();},{once:true});video.currentTime=t;});}
async function sampleInputs(){
  if(!video.videoWidth)throw Error('ローカル動画を選択してください');video.pause();
  const vw=video.videoWidth,vh=video.videoHeight,x=Number($('x').value)/100*vw,y=Number($('y').value)/100*vh;
  let w=Number($('w').value)/100*vw,h=Number($('h').value)/100*vh;
  if(w<=0||h<=0)throw Error('ROIサイズを確認してください');const cx=x+w/2,cy=y+h/2;
  if(w/h>.75)h=w/.75;else w=h*.75;
  const inputs=[],mean=[123.675,116.28,103.53],std=[58.395,57.12,57.375];
  for(const f of[.2,.5,.8]){await seek(Math.min(video.duration-.01,video.duration*f));ctx.fillStyle='black';ctx.fillRect(0,0,288,384);ctx.save();ctx.setTransform(288/w,0,0,384/h,-(cx-w/2)*288/w,-(cy-h/2)*384/h);ctx.drawImage(video,0,0);ctx.restore();
    const pixels=ctx.getImageData(0,0,288,384).data,n=288*384,data=new Float32Array(n*3);for(let i=0;i<n;i++)for(let c=0;c<3;c++)data[c*n+i]=(pixels[i*4+c]-mean[c])/std[c];inputs.push(data);
  }return inputs;
}
function render(){
  $('results').replaceChildren();for(const r of results){const tr=document.createElement('tr');
    const metrics=r.metrics,part=p=>p?.observations?`${p.xyP95_px.toFixed(2)}px / ${(p.zP95_m*1000).toFixed(1)}mm`:'観測不足';
    const verdict=r.error||(r.reference?'基準':r.qualityPassed===true?'差分基準内':r.qualityPassed===false?'差分基準外':'計測中');
    for(const value of[r.id,r.executionLabel,(r.sizeBytes/1e6).toFixed(1),metrics?`${metrics.p50Ms.toFixed(2)} / ${metrics.p95Ms.toFixed(2)}`:'—',metrics?metrics.aggregateFPS.toFixed(1):'—',metrics?metrics.perInputFPS.toFixed(1):'—',part(r.body),part(r.ankles),part(r.feet),verdict]){const td=document.createElement('td');td.textContent=value;tr.append(td);}if(r.error||r.qualityPassed===false)tr.className='bad';$('results').append(tr);
  }
}
async function profile(runtime,data,mode,inputs){
  const events=[];let runner;
  runtime.env.webgpu.profiling={mode:'default',ondata:e=>events.push(e)};
  try{
    runner=await createExecution(runtime,data,mode);
    for(let i=0;i<3;i++)await runner.run(inputs[i%inputs.length]);
    await runner.device.queue.onSubmittedWorkDone();await new Promise(resolve=>setTimeout(resolve,0));events.length=0;
    await runner.run(inputs[0]);await runner.device.queue.onSubmittedWorkDone();await new Promise(resolve=>setTimeout(resolve,0));
    return summarizeProfile(events);
  }finally{if(runner)await runner.release();runtime.env.webgpu.profiling={mode:'off',ondata:()=>{}};}
}
async function measure(){
  stopped=false;results=[];render();$('start').disabled=true;$('stop').disabled=false;$('download').disabled=true;$('native-input').disabled=true;
  try{
    const selected=[...document.querySelectorAll('#models input:checked')].map(x=>x.value),set=new Set(selected);
    for(const id of selected){const family=catalog.variants.find(v=>v.id===id).model;set.add(family+'-fp32');}
    const variants=catalog.variants.filter(v=>set.has(v.id)).toSorted((a,b)=>(a.policy==='fp32'?0:1)-(b.policy==='fp32'?0:1)||a.id.localeCompare(b.id));
    const requestedModes=[...document.querySelectorAll('#modes input:checked')].map(x=>x.value);
    if(!selected.length||!requestedModes.length)throw Error('モデルと実行経路を選択してください');
    const modes=MODES.filter(m=>m.id==='standard'||requestedModes.includes(m.id));
    const runtime=await loadRuntime(),inputs=await sampleInputs(),reference=new Map(),views=Number($('views').value),iterations=Math.max(12,Math.min(1000,Number($('iterations').value)||200));
    preparedInputs=inputs;$('native-input').disabled=false;
    runContext={source:{width:video.videoWidth,height:video.videoHeight,duration:video.duration},inputShape:[1,3,384,288],sampleFractions:[.2,.5,.8],roiPercent:Object.fromEntries(['x','y','w','h'].map(k=>[k,Number($(k).value)])),views,iterations,warmupRuns:5};
    for(const variant of variants){
      if(stopped)break;const data=await bytes(variant.file,variant.sha256);
      const routes=modes.filter(mode=>requestedModes.includes(mode.id)||(variant.policy==='fp32'&&mode.id==='standard')).map(mode=>({mode,flavour:'patched'}));
      if($('original').checked&&variant.policy==='fp32')routes.push({mode:MODES[0],flavour:'original'});
      for(const {mode,flavour} of routes){
        if(stopped)break;
        const record={id:variant.id,sizeBytes:variant.sizeBytes,views,requestedBackend:'webgpu',executionMode:mode.id,executionLabel:mode.label+(flavour==='original'?' / 未パッチJSEP':''),preferredLayout:mode.layout,graphCaptureRequested:mode.capture,cpuFallbackDisabled:mode.capture,runtime:flavour==='original'?'1.27.0-JSEP-original':'1.27.0-JSEP-f32-accumulation-patch',hfRevision:revision};
        results.push(record);let runner;
        try{
          status(`初期化: ${variant.id} / ${record.executionLabel}`);const load=performance.now(),activeRuntime=await loadRuntime(flavour);
          activeRuntime.env.webgpu.profiling={mode:'off',ondata:()=>{}};
          runner=await createExecution(activeRuntime,data,mode);record.loadMs=performance.now()-load;
          record.enabledGPUFeatures=[...runner.device.features];
          for(let i=0;i<5;i++)await runner.run(inputs[i%inputs.length]);
          const times=[],submitTimes=[],readbacks=[],observed=[];status(`計測: ${variant.id} / ${mode.label}`);const started=performance.now();
          for(let i=0;i<iterations&&!stopped;i++){
            const out=await runner.run(inputs[i%inputs.length]);times.push(out.totalMs);submitTimes.push(out.runMs);readbacks.push(out.readbackMs);
            if(i<inputs.length)observed.push({points:Array.from(out.points),scores:Array.from(out.scores)});
          }
          const elapsed=performance.now()-started;record.interrupted=stopped;
          if(times.length)record.metrics={p50Ms:pct(times,.5),p95Ms:pct(times,.95),aggregateFPS:times.length/elapsed*1000,perInputFPS:times.length/elapsed*1000/views,actualInferences:times.length,wallMs:elapsed,sessionTimesMs:times,submitP50Ms:pct(submitTimes,.5),readbackP50Ms:pct(readbacks,.5),includesUploadAndReadback:true,profilingEnabledDuringTiming:false};
          record.reference=variant.policy==='fp32'&&mode.id==='standard'&&flavour==='patched';
          if(record.reference&&!stopped)reference.set(variant.model,observed);
          const ref=reference.get(variant.model);
          if(ref&&observed.length===ref.length&&!stopped){for(const [name,ids]of Object.entries(GROUPS))record[name]=compare(observed,ref,ids);record.qualityPassed=qualityGate(record);}
          await runner.release();runner=null;
          if($('profile').checked&&!stopped){status(`演算別診断: ${variant.id} / ${record.executionLabel}`);try{record.gpuProfile=await profile(activeRuntime,data,mode,inputs);}catch(error){record.profilingError=String(error.message||error);}}
        }catch(e){record.error=String(e.message||e);}finally{if(runner)await runner.release().catch(()=>{});render();}
      }
    }
    status(stopped?'停止しました':'計測完了。GPU完了と座標の読戻しを含む時間です。Graph Capture失敗は結果欄に記録します。');$('download').disabled=false;
  }catch(e){status(String(e.message||e));}finally{$('start').disabled=false;$('stop').disabled=true;}
}
$('video-file').onchange=e=>{if(!e.target.files[0])return;if(videoURL)URL.revokeObjectURL(videoURL);videoURL=URL.createObjectURL(e.target.files[0]);video.src=videoURL;video.load();};
$('start').onclick=measure;$('stop').onclick=()=>{stopped=true;};
$('download').onclick=()=>{const data={schema:'rtmw3d-gpu-comparison/v2',createdAt:new Date().toISOString(),source:runContext?.source,runContext,results,notes:['All variants output all 133 joints','Differences are relative to the architecture FP32 standard-path result, not ground truth','Timings include input upload, GPU completion and fresh 133-joint CPU readback','GPU timestamp profiling uses a separate session and is excluded from speed measurements','requestedBackend is the requested provider; it does not prove that every node ran on GPU','1–4 inputs measure a shared-session workload, not synchronized physical cameras','GPU utilization, VRAM, device identity, video names and video contents are not exported']};const u=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=u;a.download='rtmw3d-fullbody-gpu-comparison.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);};
$('native-input').onclick=()=>{if(!preparedInputs)return;if(new Uint8Array(new Uint32Array([1]).buffer)[0]!==1)throw Error('この端末ではlittle-endian入力を保存できません');const url=URL.createObjectURL(new Blob(preparedInputs,{type:'application/octet-stream'})),a=document.createElement('a');a.href=url;a.download='rtmw3d-native-inputs.f32';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
try{
  const model=await(await get(`https://huggingface.co/api/models/${REPO}`)).json();revision=model.sha;base=`https://huggingface.co/${REPO}/resolve/${revision}/`;catalog=await(await get(base+'models.json')).json();
  if(catalog.keypointCount!==133)throw Error('全身133点のカタログではありません');$('models').replaceChildren();
  const experimentalResponse=await fetch(base+'experiments/catalog.json',{credentials:'omit',referrerPolicy:'no-referrer'});
  if(experimentalResponse.ok){const extra=await experimentalResponse.json();if(extra.keypointCount!==133||extra.schema!=='rtmw3d-head-experiments/v1')throw Error('実験カタログの全身出力契約が不正です');for(const v of extra.variants){if(!catalog.variants.some(existing=>existing.id===v.id))catalog.variants.push(v);}}
  else if(experimentalResponse.status!==404)throw Error('実験カタログの取得に失敗: '+experimentalResponse.status);
  for(const v of catalog.variants){const label=document.createElement('label'),checkbox=document.createElement('input');checkbox.type='checkbox';checkbox.value=v.id;checkbox.checked=v.id==='rtmw3d-l-fp32'||(v.experimental&&v.qualityPassed);label.append(checkbox,document.createTextNode(`${v.id} · ${(v.sizeBytes/1e6).toFixed(0)}MB${v.experimental?' · 等価FP32実験':''}`));$('models').append(label);}
  for(const m of MODES){const label=document.createElement('label'),checkbox=document.createElement('input');checkbox.type='checkbox';checkbox.value=m.id;checkbox.checked=m.id==='nhwc'||(m.id==='capture'&&catalog.variants.some(v=>v.gpuDecoderMinimum));label.append(checkbox,document.createTextNode(m.label));$('modes').append(label);}
  $('start').disabled=false;status(`HF revision ${revision}\n全候補：全133点。FP16・INT8の速さを仮定しません。`);
}catch(e){status(String(e.message||e));}
