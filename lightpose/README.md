---
license: apache-2.0
library_name: onnxruntime
pipeline_tag: keypoint-detection
tags:
- onnx
- rtmpose
- mmpose
- wholebody
---
# RTMPose-t/s WholeBody133 ONNX
These small models retain ankles, toes, heels, face and fingers while reducing the compute required for single-person 2D wholebody tracking. They support comparing a lighter detector against RTMW using the same 133-point layout and calibrated pseudo-3D geometry.
Upstream: https://github.com/open-mmlab/mmpose
Parent checkpoints: official RTMPose-t/s COCO-WholeBody133, ucoco/DWPose-distilled 270-epoch weights, input 256×192. These are not the Halpe26 checkpoints. The official table lists 0.5/0.9 GFLOPs and WholeBody AP 48.5/53.8 for tiny/small; neither predicts learned 3D coordinates.
Input: float32 RGB NCHW [1,3,256,192], ImageNet mean [123.675,116.28,103.53], std [58.395,57.12,57.375] for 0–255 pixels. Outputs: simcc_x [1,133,384], simcc_y [1,133,512]. SimCC split ratio is 2. Keep the top-down ROI transform when mapping coordinates back to the original image.
Each model has FP32, FP16 all/backbone/head, and calibrated INT8 QDQ all/backbone/head variants. Layer names refer to neural network components; every variant retains all anatomical outputs. Reports include FP32 drift for hips, knees, ankles and foot joints. Fixture drift is not ground-truth accuracy, and file size or precision does not establish browser speed. Browser WebGPU performance and target-video accuracy remain unmeasured until run on the target device.
Build workflow: https://github.com/DaisukeDaisuke/mmpose-actions/actions/workflows/build-lightpose.yml
Distribution: https://huggingface.co/RabbitDaisuke/RTMPose-Wholebody-onnx
MMPose copyright and Apache-2.0 text are retained in LICENSE. These files convert the official trained checkpoints to ONNX and apply the named numerical transformations; no retraining is performed.
The checkpoint is downloaded only on cache miss and retained with a SHA-256 sidecar in Actions cache. Existing RTMW and RTMW3D models are preserved.
