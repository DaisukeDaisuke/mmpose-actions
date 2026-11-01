import {mkdir,writeFile,readFile}from 'node:fs/promises';
import {createHash}from 'node:crypto';
import path from 'node:path';
import {EXPECTED_ORT_VERSION,patchOrtWebGpuF32Accumulate}from '../patches/onnxruntime-web-f32-patch.js';

const root=path.resolve(process.argv[2]||'release'),out=path.join(root,'runtime');
await mkdir(out,{recursive:true});
const base=`https://cdn.jsdelivr.net/npm/onnxruntime-web@${EXPECTED_ORT_VERSION}/`;
async function fetchBytes(url){for(let attempt=0;attempt<3;attempt++){
  try{const r=await fetch(url,{signal:AbortSignal.timeout(180000)});if(!r.ok)throw Error(`${url}: HTTP ${r.status}`);return Buffer.from(await r.arrayBuffer());}
  catch(e){if(attempt===2)throw e;await new Promise(r=>setTimeout(r,1000*(attempt+1)));}
}}
const pkg=JSON.parse((await fetchBytes(base+'package.json')).toString('utf8'));
if(pkg.version!==EXPECTED_ORT_VERSION)throw Error(`ORT version mismatch ${pkg.version}`);
const hash=b=>createHash('sha256').update(b).digest('hex');
// This supplied patch targets the JSEP WebGPU kernels in the all bundle.
// The 1.27.0 webgpu-only bundle uses a different backend and does not match it.
const original=await fetchBytes(base+'dist/ort.all.min.mjs');
const patched=Buffer.from(patchOrtWebGpuF32Accumulate(original.toString('utf8')),'utf8');
if(hash(original)===hash(patched))throw Error('Expected upstream MatMul accumulator patch to modify the original runtime');
if(patchOrtWebGpuF32Accumulate(patched.toString('utf8'))!==patched.toString('utf8'))throw Error('Patch is not idempotent');
await writeFile(path.join(out,'ort.all.min.mjs'),patched);
const files=[{path:'runtime/ort.all.min.mjs',size:patched.length,sha256:hash(patched),upstreamSha256:hash(original)}];
for(const name of['ort-wasm-simd-threaded.jsep.mjs','ort-wasm-simd-threaded.jsep.wasm']){
  const bytes=await fetchBytes(base+'dist/'+name);await writeFile(path.join(out,name),bytes);files.push({path:'runtime/'+name,size:bytes.length,sha256:hash(bytes)});
}
const patch=await readFile(new URL('../patches/onnxruntime-web-f32-patch.js',import.meta.url));
await writeFile(path.join(out,'onnxruntime-web-f32-patch.js'),patch);
const license=await fetchBytes('https://raw.githubusercontent.com/microsoft/onnxruntime/v1.27.0/LICENSE');await writeFile(path.join(out,'LICENSE-onnxruntime.txt'),license);
const manifest={version:EXPECTED_ORT_VERSION,entrypoint:'runtime/ort.all.min.mjs',backend:'JSEP WebGPU',patch:'user-supplied-webgpu-matmul-f32-accumulation',patchSha256:hash(patch),
  upstream:'https://github.com/microsoft/onnxruntime',files,browserGPUValidated:false,
  note:'FP16 MatMul accumulation uses f32 intermediates. Native WebGPU correctness and speed require a GPU browser benchmark.'};
await writeFile(path.join(out,'manifest.json'),JSON.stringify(manifest,null,2)+'\n');
console.log(JSON.stringify(manifest,null,2));
