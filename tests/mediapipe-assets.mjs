import assert from 'node:assert/strict';
const specs=[['pose','0.5.1675469404','pose_web.binarypb'],['holistic','0.5.1675471629','holistic.binarypb']];
for(const [kind,version,graph] of specs){
  const base=`https://cdn.jsdelivr.net/npm/@mediapipe/${kind}@${version}/`;
  const response=await fetch(base+kind+'.js',{signal:AbortSignal.timeout(45000)});assert.equal(response.status,200);
  const source=await response.text();
  if(kind==='holistic')assert.match(source,/za:\{type:"proto",stream:"world_landmarks"/);
  else assert.ok(source.includes('poseWorldLandmarks'));
  const files=[graph,kind+'_solution_simd_wasm_bin.js',kind+'_solution_simd_wasm_bin.wasm',kind+'_solution_packed_assets_loader.js','pose_landmark_lite.tflite','pose_landmark_full.tflite'];
  const checks=await Promise.all(files.map(async file=>{const r=await fetch(base+file,{method:'HEAD',signal:AbortSignal.timeout(45000)});assert.equal(r.status,200,kind+'/'+file);return file;}));
  console.log(kind,version,'verified:',checks.join(', '));
}
