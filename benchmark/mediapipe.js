import {poseWorldResult} from './mediapipe-adapter.js';
const $=id=>document.getElementById(id),video=$('video');
const packages={pose:{version:'0.5.1675469404',global:'Pose'},holistic:{version:'0.5.1675471629',global:'Holistic'}};
const sources=new Map();let busy=false,stopped=false,rows=[],context,videoURL;
const pct=(a,p)=>{const s=[...a].sort((x,y)=>x-y);return s.length?s[Math.floor((s.length-1)*p)]:null;};
const status=s=>$('status').textContent=s;
async function load(kind){
  if(sources.has(kind))return sources.get(kind);
  const spec=packages[kind],base=`https://cdn.jsdelivr.net/npm/@mediapipe/${kind}@${spec.version}/`;
  await new Promise((resolve,reject)=>{const script=document.createElement('script');script.src=base+kind+'.js';script.crossOrigin='anonymous';script.referrerPolicy='no-referrer';script.onload=resolve;script.onerror=()=>reject(Error('MediaPipeスクリプトの取得に失敗'));document.head.append(script);});
  const result={Class:globalThis[spec.global],base,version:spec.version};if(!result.Class)throw Error('MediaPipeクラスが見つかりません');sources.set(kind,result);return result;
}
async function seek(time){if(Math.abs(video.currentTime-time)<.001&&video.readyState>=2)return;await new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(Error('動画のシークに失敗')),15000);video.addEventListener('seeked',()=>{clearTimeout(timer);resolve();},{once:true});video.currentTime=time;});}
async function samples(){
  if(!video.videoWidth)throw Error('ローカル動画を選択してください');video.pause();
  const frames=[],x=Number($('x').value)/100*video.videoWidth,y=Number($('y').value)/100*video.videoHeight;
  let w=Number($('w').value)/100*video.videoWidth,h=Number($('h').value)/100*video.videoHeight;
  if(w<=0||h<=0)throw Error('ROIサイズが不正です');const cx=x+w/2,cy=y+h/2;if(w/h>.75)h=w/.75;else w=h*.75;
  for(const f of[.2,.5,.8]){
    await seek(Math.min(video.duration-.01,video.duration*f));const canvas=document.createElement('canvas');canvas.width=288;canvas.height=384;const ctx=canvas.getContext('2d');
    ctx.fillStyle='black';ctx.fillRect(0,0,288,384);ctx.setTransform(288/w,0,0,384/h,-(cx-w/2)*288/w,-(cy-h/2)*384/h);ctx.drawImage(video,0,0);frames.push(canvas);
  }return frames;
}
function render(){
  $('results').replaceChildren();for(const row of rows){const tr=document.createElement('tr'),m=row.metrics;
    for(const v of[row.id,m?`${m.p50Ms.toFixed(2)} / ${m.p95Ms.toFixed(2)}`:'—',m?m.aggregateFPS.toFixed(1):'—',m?m.perInputFPS.toFixed(1):'—',`${row.poseFoundFrames??0}/${row.measuredFrames??0}`,`${row.feetVisibleFrames??0}/${row.measuredFrames??0}`,row.worldFoundFrames?`${row.worldFoundFrames}/${row.measuredFrames}`:'取得なし',row.error||'計測完了']){const td=document.createElement('td');td.textContent=v;tr.append(td);}if(row.error||!row.poseFoundFrames)tr.className='bad';$('results').append(tr);
  }
}
$('video-file').onchange=e=>{if(!e.target.files[0])return;if(videoURL)URL.revokeObjectURL(videoURL);videoURL=URL.createObjectURL(e.target.files[0]);video.src=videoURL;video.load();};
$('start').onclick=async()=>{
  if(busy)return;busy=true;stopped=false;rows=[];render();$('start').disabled=true;$('stop').disabled=false;$('download').disabled=true;
  try{
    const frames=await samples(),iterations=Math.max(12,Math.min(1000,Number($('iterations').value)||200)),views=Number($('views').value);
    context={source:{width:video.videoWidth,height:video.videoHeight,duration:video.duration},roiPercent:Object.fromEntries(['x','y','w','h'].map(k=>[k,Number($(k).value)])),sampleFractions:[.2,.5,.8],inputCanvas:[288,384],iterations,views};
    const choices=[...document.querySelectorAll('#models input:checked')].map(i=>i.value);
    if(!choices.length)throw Error('比較候補を選択してください');
    for(const id of choices){
      if(stopped)break;const [kind,complexity]=id.split(':'),record={id:kind+'-'+(complexity==='0'?'lite':'full'),requestedBackend:'MediaPipe GPU delegate',backendVerified:false,packageVersion:packages[kind].version,views,poseFoundFrames:0,feetVisibleFrames:0,worldFoundFrames:0,measuredFrames:0};rows.push(record);let model,lastResult;
      try{
        status(`初期化: ${record.id}`);const module=await load(kind);model=new module.Class({locateFile:file=>module.base+file});
        model.setOptions({modelComplexity:Number(complexity),useCpuInference:false,smoothLandmarks:false,enableSegmentation:false,smoothSegmentation:false,refineFaceLandmarks:false,minDetectionConfidence:.5,minTrackingConfidence:.5});
        model.onResults(result=>{lastResult=result;});await model.initialize();
        const run=async frame=>{lastResult=null;await model.send({image:frame});if(!lastResult)throw Error('MediaPipe結果コールバックが返りません');return lastResult;};
        for(let i=0;i<5;i++)await run(frames[i%frames.length]);
        const times=[],started=performance.now();status(`計測: ${record.id}`);
        for(let i=0;i<iterations&&!stopped;i++){
          const begin=performance.now(),result=await run(frames[i%frames.length]);times.push(performance.now()-begin);record.measuredFrames++;
          const points=result.poseLandmarks??[],worldResult=poseWorldResult(result,kind,packages[kind].version),world=worldResult.points;record.worldResultField=worldResult.field;
          if(points.length===33&&points.every(p=>Number.isFinite(p.x)&&Number.isFinite(p.y))){record.poseFoundFrames++;if([27,28,29,30,31,32].every(j=>(points[j].visibility??0)>=.3))record.feetVisibleFrames++;}
          if(world.length===33&&world.every(p=>Number.isFinite(p.x)&&Number.isFinite(p.y)&&Number.isFinite(p.z)))record.worldFoundFrames++;
          record.lastOutputCounts={pose:points.length,world:world.length,face:result.faceLandmarks?.length??0,leftHand:result.leftHandLandmarks?.length??0,rightHand:result.rightHandLandmarks?.length??0};
        }
        const wallMs=performance.now()-started;record.interrupted=stopped;
        if(times.length)record.metrics={p50Ms:pct(times,.5),p95Ms:pct(times,.95),aggregateFPS:times.length/wallMs*1000,perInputFPS:times.length/wallMs*1000/views,wallMs,actualInferences:times.length,sessionTimesMs:times};
        record.trackingUsable=record.measuredFrames>0&&record.poseFoundFrames===record.measuredFrames&&record.feetVisibleFrames===record.measuredFrames;
      }catch(error){record.error=String(error.message||error);}finally{if(model)await model.close().catch(()=>{});render();}
    }status(stopped?'停止しました':'計測完了。追跡率とworld座標の取得有無を速度と併せて確認してください。');$('download').disabled=false;
  }catch(error){status(String(error.message||error));}finally{busy=false;$('start').disabled=false;$('stop').disabled=true;}
};
$('stop').onclick=()=>{stopped=true;};
$('download').onclick=()=>{
  const report={schema:'mediapipe-fullbody-comparison/v1',createdAt:new Date().toISOString(),context,results:rows,notes:['Local video only; video contents, names and landmark coordinates are not exported','Same three sampled crops are repeated; this is not a continuous-motion jitter or accuracy test','Includes MediaPipe detection/tracking/pose and optional face/hand pipeline; RTMW3D timings exclude person detection','GPU delegate is requested but individual kernel placement is not profiled','33-point pose topology includes ankles, heels and foot-index landmarks','World coordinates, when available, are hip-relative model estimates, not calibrated room position','Pinned Holistic 0.5.1675471629 world_landmarks output is read from its source-verified minified za field; missing data is never fabricated','Detection failure can produce misleadingly high FPS: inspect pose/foot coverage']};
  const url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='mediapipe-fullbody-comparison.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
