const $=id=>document.getElementById(id),REPO='RabbitDaisuke/rtmw3d-webgpu-onnx';
const video=$('video'),crop=$('crop'),ctx=crop.getContext('2d',{willReadFrequently:true});
let catalog,revision,base,ort,stopped=false,results=[],videoURL;
const status=s=>$('status').textContent=s;
const pct=(values,p)=>{const s=values.toSorted((a,b)=>a-b);return s.length?s[Math.floor((s.length-1)*p)]:null;};
async function hash(bytes){return [...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(x=>x.toString(16).padStart(2,'0')).join('');}
async function get(url){const r=await fetch(url,{credentials:'omit',referrerPolicy:'no-referrer'});if(!r.ok)throw Error(`${r.status}: ${url}`);return r;}
async function bytes(file,sha){
  const key=base+file,cache=await caches.open('rtmw3d-comparison-v1');let r=await cache.match(key);
  if(r){const b=await r.arrayBuffer();if(await hash(b)===sha)return b;await cache.delete(key);}
  status(`取得・照合中: ${file}`);r=await get(key);const b=await r.arrayBuffer();if(await hash(b)!==sha)throw Error(`SHA-256不一致: ${file}`);
  try{await cache.put(key,new Response(b));}catch{status('キャッシュ容量不足。取得済みモデルで計測を続けます。');}return b;
}
async function loadRuntime(){
  if(ort)return ort;if(!navigator.gpu)throw Error('WebGPU対応のPCブラウザで開いてください');
  const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});if(!adapter)throw Error('WebGPUアダプターを取得できません');
  const manifest=catalog.runtime,entry=manifest.files.find(f=>f.path===manifest.entrypoint),glue=manifest.files.find(f=>f.path.endsWith('.jsep.mjs'));
  const js=URL.createObjectURL(new Blob([await bytes(entry.path,entry.sha256)],{type:'text/javascript'}));
  const module=URL.createObjectURL(new Blob([await bytes(glue.path,glue.sha256)],{type:'text/javascript'}));
  ort=await import(js);ort.env.wasm.numThreads=1;ort.env.wasm.proxy=false;
  ort.env.wasm.wasmPaths={mjs:module,wasm:base+'runtime/ort-wasm-simd-threaded.jsep.wasm'};
  ort.env.webgpu.powerPreference='high-performance';
  return ort;
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
function compare(observed,reference,ids){
  const xy=[],z=[];let lost=0;
  for(let frame=0;frame<observed.length;frame++)for(const i of ids){const ref=reference[frame],out=observed[frame];if(ref.scores[i]<=.3)continue;
    xy.push(Math.hypot(out.points[i*3]-ref.points[i*3],out.points[i*3+1]-ref.points[i*3+1]));z.push(Math.abs(out.points[i*3+2]-ref.points[i*3+2]));if(out.scores[i]<=.3)lost++;
  }return {observations:xy.length,xyP95_px:pct(xy,.95),zP95_m:pct(z,.95),confidenceLossCount:lost};
}
function render(){
  $('results').replaceChildren();for(const r of results){const tr=document.createElement('tr');
    const metrics=r.metrics,part=p=>p?.observations?`${p.xyP95_px.toFixed(2)}px / ${(p.zP95_m*1000).toFixed(1)}mm`:'観測不足';
    for(const value of[r.id,(r.sizeBytes/1e6).toFixed(1),metrics?`${metrics.p50Ms.toFixed(2)} / ${metrics.p95Ms.toFixed(2)}`:'—',metrics?metrics.aggregateFPS.toFixed(1):'—',metrics?metrics.perInputFPS.toFixed(1):'—',part(r.ankles),part(r.feet),r.error||'推論完了']){const td=document.createElement('td');td.textContent=value;tr.append(td);}if(r.error)tr.className='bad';$('results').append(tr);
  }
}
async function measure(){
  stopped=false;results=[];render();$('start').disabled=true;$('stop').disabled=false;
  try{
    const selected=[...document.querySelectorAll('#models input:checked')].map(x=>x.value),set=new Set(selected);
    for(const id of selected){const family=catalog.variants.find(v=>v.id===id).model;set.add(family+'-fp32');}
    const variants=catalog.variants.filter(v=>set.has(v.id)).toSorted((a,b)=>(a.policy==='fp32'?0:1)-(b.policy==='fp32'?0:1)||a.id.localeCompare(b.id));
    const runtime=await loadRuntime(),inputs=await sampleInputs(),reference=new Map(),views=Number($('views').value),iterations=Math.max(12,Math.min(1000,Number($('iterations').value)||40));
    for(const variant of variants){if(stopped)break;const record={id:variant.id,sizeBytes:variant.sizeBytes,views,backend:'webgpu',runtime:'1.27.0-JSEP-f32-accumulation-patch',hfRevision:revision};results.push(record);let session;
      try{
        const data=await bytes(variant.file,variant.sha256);status(`初期化: ${variant.id}`);const load=performance.now();
        session=await runtime.InferenceSession.create(data,{executionProviders:['webgpu'],graphOptimizationLevel:'all'});record.loadMs=performance.now()-load;
        for(let i=0;i<3;i++){const t=new runtime.Tensor('float32',inputs[i%inputs.length],[1,3,384,288]);const out=await session.run({input:t});Object.values(out).forEach(x=>x.dispose());t.dispose();}
        const times=[],observed=[];status(`計測: ${variant.id} · ${views}入力相当`);const started=performance.now();
        for(let i=0;i<iterations&&!stopped;i++){
          const input=new runtime.Tensor('float32',inputs[i%inputs.length],[1,3,384,288]),t=performance.now();
          const out=await session.run({input});const points=out.keypoints.data,scores=out.scores.data;times.push(performance.now()-t);
          if(points.length!==399||scores.length!==133||points.some(v=>!Number.isFinite(v))||scores.some(v=>!Number.isFinite(v)))throw Error('全133関節の出力契約または有限値検査に失敗');
          if(i<inputs.length)observed.push({points:Array.from(points),scores:Array.from(scores)});Object.values(out).forEach(x=>x.dispose());input.dispose();
        }
        const elapsed=performance.now()-started;
        record.metrics={p50Ms:pct(times,.5),p95Ms:pct(times,.95),aggregateFPS:times.length/elapsed*1000,perInputFPS:times.length/elapsed*1000/views,actualInferences:times.length,wallMs:elapsed,sessionTimesMs:times};
        if(variant.policy==='fp32')reference.set(variant.model,observed);
        const ref=reference.get(variant.model);if(ref&&observed.length===ref.length){record.body=compare(observed,ref,Array.from({length:23},(_,i)=>i));record.ankles=compare(observed,ref,[15,16]);record.feet=compare(observed,ref,[17,18,19,20,21,22]);}
      }catch(e){record.error=String(e.message||e);}finally{if(session)await session.release().catch(()=>{});render();}
    }status(stopped?'停止しました':'計測完了。合計fpsと入力あたりfpsを分けて確認してください。');$('download').disabled=false;
  }catch(e){status(String(e.message||e));}finally{$('start').disabled=false;$('stop').disabled=true;}
}
$('video-file').onchange=e=>{if(!e.target.files[0])return;if(videoURL)URL.revokeObjectURL(videoURL);videoURL=URL.createObjectURL(e.target.files[0]);video.src=videoURL;video.load();};
$('start').onclick=measure;$('stop').onclick=()=>{stopped=true;};
$('download').onclick=()=>{const data={schema:'rtmw3d-gpu-comparison/v1',createdAt:new Date().toISOString(),source:{width:video.videoWidth,height:video.videoHeight,duration:video.duration},results,notes:['All variants output all 133 joints','Differences are relative to the same architecture FP32 result, not ground truth','1–4 inputs measure a shared-session workload, not synchronized physical cameras','GPU utilization and VRAM are not measured']};const u=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=u;a.download='rtmw3d-fullbody-gpu-comparison.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);};
try{
  const model=await(await get(`https://huggingface.co/api/models/${REPO}`)).json();revision=model.sha;base=`https://huggingface.co/${REPO}/resolve/${revision}/`;catalog=await(await get(base+'models.json')).json();
  if(catalog.keypointCount!==133)throw Error('全身133点のカタログではありません');$('models').replaceChildren();
  for(const v of catalog.variants){const label=document.createElement('label'),checkbox=document.createElement('input');checkbox.type='checkbox';checkbox.value=v.id;checkbox.checked=['rtmw3d-l-fp32','rtmw3d-l-fp16-backbone','rtmw3d-l-int8-backbone'].includes(v.id);label.append(checkbox,document.createTextNode(`${v.id} · ${(v.sizeBytes/1e6).toFixed(0)}MB`));$('models').append(label);}
  $('start').disabled=false;status(`HF revision ${revision}\n全候補：全133点。FP16・INT8の速さを仮定しません。`);
}catch(e){status(String(e.message||e));}
