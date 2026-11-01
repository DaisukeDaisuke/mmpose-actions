import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync}from 'node:fs';
import {patchOrtWebGpuF32Accumulate,EXPECTED_ORT_VERSION}from '../patches/onnxruntime-web-f32-patch.js';
test('patch rejects unrelated or partially matching bundles instead of silently succeeding',()=>{
  assert.equal(EXPECTED_ORT_VERSION,'1.27.0');
  assert.throws(()=>patchOrtWebGpuF32Accumulate('unrelated runtime'),/neither/);
  assert.throws(()=>patchOrtWebGpuF32Accumulate('var acc: array<vec4<${n}>, rowPerThread>;'),/Unexpected/);
});
test('patch source is version-pinned and includes explicit f32 accumulation',()=>{
  const src=readFileSync(new URL('../patches/onnxruntime-web-f32-patch.js',import.meta.url),'utf8');
  assert.ok(src.includes('vec4<f32>(BCached0) * f32(ACached0[i])'));
  assert.ok(src.includes('Failed to verify onnxruntime-web replacement'));
});
