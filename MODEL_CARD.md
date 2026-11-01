---
license: apache-2.0
library_name: onnxruntime
pipeline_tag: keypoint-detection
tags:
  - onnx
  - webgpu
  - wholebody-pose
  - pose-estimation
  - 3d
  - quantization
  - rtmw3d
---
# RTMW3D full-body ONNX experiments for WebGPU
Browser-ready RTMW3D-L and RTMW3D-X variants for full-body camera tracking, including both ankles, toes and heels. Every candidate preserves all 133 joints and root-relative depth.
Original models and implementation: **https://github.com/open-mmlab/mmpose**
Reproducible builds: https://github.com/DaisukeDaisuke/mmpose-actions
## Benefits
Compare a lighter whole-body architecture and multiple quantization policies without changing preprocessing, joint order or the output contract. Layer-selected experiments show which numerical changes preserve hips, knees, ankles and feet; they never restrict output to the face or hands.
The SimCC decoder runs inside ONNX. Outputs shrink from 1,021,440 bytes to **2,128 bytes per inference**, while retaining all 133 joints. This reduces GPU-to-CPU readback traffic and JavaScript decoding work.
FP32 references, FP16 experiments and calibrated INT8 QDQ experiments share reproducible comparisons. File sizes, checksums, selected layers and per-region drift are inspectable. Choose a model using measured GPU latency and full-body stability rather than assuming lower precision is faster.
## Candidate matrix
For each of RTMW3D-L and RTMW3D-X: FP32 reference; FP16 throughout; FP16 feature extractor only; FP16 feature fusion and all-joint output layers; INT8 throughout; INT8 feature extractor only; INT8 feature fusion only; INT8 all-joint output layers only.
Architecture names backbone, neck and head refer to network components. The head predicts all body, foot, face and hand joints; it does not mean a face-only model.
## Input and output
Input input: float32 NCHW [1,3,384,288]. RGB person crop, normalized once with mean [123.675,116.28,103.53] and std [58.395,57.12,57.375] in 0–255 pixel units.
Output keypoints: float32 [1,133,3]. X/Y are pixels in the 288×384 crop; Z is pelvis-relative metres. Output scores: float32 [1,133], the original minimum of the X/Y peak values. Scores are not calibrated probabilities. Map crop XY back into the original image before camera geometry.
Joint order: body 0–16, including ankles 15,16; feet 17–22; face 23–90; left hand 91–111; right hand 112–132.
## GPU comparison
Serve the bundled benchmark directory and use the supplied matching runtime. It compares identical locally selected video frames with WebGPU, reports median/p95 session time, and keeps model load and warm-up separate. Videos remain on the device. The 1–4 input setting measures inference workload with one reused session; it does not claim synchronized physical cameras or multi-view reconstruction.
models.json contains the complete catalog. COMPARISON.md summarizes size, the small-fixture drift gate, and CPU reference timings. CPU timings from different runners are not a GPU ranking. FP16 or INT8 may be slower or unsupported on a particular WebGPU runtime/GPU.
## Runtime
runtime/ort.all.min.mjs is ONNX Runtime Web 1.27.0 with the supplied JSEP WebGPU MatMul FP32-accumulation patch. The original patch and matching JSEP runtime files are included. Patch counts, complete application and idempotence are checked during the build.
## Validation scope
FP32 exports are compared with the official PyTorch model. Calibration and validation use disjoint image IDs. Body, hips, knees, ankles and six foot points have independent drift checks so face landmarks cannot hide foot degradation. Reports retain failed quality gates and missing observations rather than silently dropping joints or inconvenient candidates.
FP16 CPU reference checks use FP16-rounded weights/constants with FP32 compute. They do not validate native FP16 activation rounding or GPU shaders. The fixture is a conversion smoke test, not a full COCO/H3WB accuracy benchmark, temporal tracking evaluation or measured 60fps guarantee.
These models estimate relative 3D pose. Absolute camera distance, room position, floor contact, 1–4-camera calibration/synchronization/fusion, and SteamVR tracker delivery belong to the tracking application. They are not implied by a successful ONNX inference.
## Sources and license
Official project: https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose3d
MMPose: https://github.com/open-mmlab/mmpose
Models use Apache-2.0; see LICENSE. ONNX Runtime uses MIT; see runtime/LICENSE-onnxruntime.txt. The pinned upstream revision, official checkpoint URL and checkpoint SHA-256 are recorded in each architecture report.
