---
license: mit
base_model: teemosliang/SDPose-Wholebody
library_name: onnx
pipeline_tag: keypoint-detection
tags:
- onnx
- mmpose
- wholebody
- pose-estimation
---
# SDPose WholeBody133 ONNX
These ONNX exports make the trained SDPose wholebody model available outside PyTorch while retaining all 133 body, foot, face and finger heatmaps. They support evaluating SDPose's robustness to unusual poses, occlusion and appearance changes in ONNX applications, alongside smaller tracking models.
Parent model: https://huggingface.co/teemosliang/SDPose-Wholebody
Parent revision: 0c8153e39458bc8ae87fb68790c141bf0f03c11b
SDPose implementation: https://github.com/t-s-liang/SDPose-OOD/tree/90517ab286390131199630394e546d9a14e4c1c1
MMPose: https://github.com/open-mmlab/mmpose
Build workflow: https://github.com/DaisukeDaisuke/mmpose-actions/actions/workflows/build-sdpose.yml
## Inference contract
The original 1024×768 input resolution is preserved. Use a top-down person crop, RGB float32 NCHW [1,3,1024,768], normalized to [-1,1]. Person detection is separate; no YOLO weights are included.
Run the VAE encoder with image and posterior_noise [1,4,128,96], then feed its latent output [1,4,128,96] to the pose component. The noise is an explicit standard-normal input because the upstream VAE samples its posterior. Reuse identical noise when comparing variants; zero noise selects the posterior mean and changes the upstream sampling policy.
The pose component returns heatmaps [1,133,256,192]. The empty prompt embedding, timestep 999, task embedding [sin(1),sin(0),cos(1),cos(0)] and latent scaling 0.18215 follow upstream. The selected feature is the second-last UNet up-block; the unused final up-block/noise-prediction head is omitted. The heatmap head preserves SDPose's InstanceNorm and SiLU layers and strictly loads every trained head parameter.
Use MMPose UDPHeatmap postprocessing with input_size=(768,1024), heatmap_size=(192,256), then invert the ROI affine transform. Raw argmax is not equivalent to UDP subpixel refinement. Flip-test ensembling is disabled and requires a second mirrored inference with COCO-WholeBody flip indices if enabled by the application. This model predicts 2D; camera geometry or a separate model is needed for 3D.
## Variants and files
catalog.json lists five complete pipeline pairs: FP32, FP16, INT8 weights throughout, INT8 UNet weights only, and INT8 heatmap-head weights only. The latter two reuse the FP32 VAE. Each pair has a VAE encoder and a pose graph; this preserves the model stages and permits separate weight caching. Download the ONNX file together with every external weight file in its manifest. Large FP32 graphs require external tensor data.
INT8 variants use symmetric weight-only DequantizeLinear nodes for Conv, ConvTranspose, MatMul and Gemm. Activations remain FP32. These are storage/precision experiments, not a claim of native INT8 GPU execution. FP16 keeps float32 external inputs and outputs. Original FP32 weights remain the reference, and no variant is automatically declared fastest or accurate based on its name.
ONNX graph and file hashes are checked in Actions. Browser WebGPU compatibility, target-GPU speed, output drift and target-video accuracy are not established by these checks. SDPose uses a diffusion U-Net and should not be described as a lightweight RTMPose replacement. Existing RTMW/RTMW3D distribution files are not modified.
