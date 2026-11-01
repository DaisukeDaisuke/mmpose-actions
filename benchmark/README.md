# 実行性能と代替モデルの比較
この比較は、精度形式の名前だけで採用せず、必要な全身関節を維持したまま実行経路とモデルの選択を判断できるようにする。
## RTMW3Dの実行経路
標準NCHW、GPU入力・出力バッファ再利用NCHW、同NHWC、Graph Capture NHWCを同じ動画・モデル・正規化入力で比較する。標準FP32を固定基準にし、ほかの経路を自己比較して誤差ゼロにはしない。Graph CaptureではCPUフォールバックを無効にし、未対応の場合は失敗理由を残す。GPUバッファ再利用でも毎回入力を書き換え、座標を新しく読み戻す。GPU完了前のコマンド投入時間を推論完了時間として扱わない。
GPU timestamp profilingは別セッションで取得し、速度計測へ混ぜない。集計したカーネル時間にはCPU演算・転送・キュー待ちを含まない。通常版との比較は同じORT 1.27.0 JSEPのパッチ前後で行う。別実装のwebgpu-only版と混同しない。
出典：https://onnxruntime.ai/docs/tutorials/web/performance-diagnosis.html
## ネイティブGPU診断
ブラウザで「同じ入力を端末内へ保存」を選び、3枚の正規化済み入力を保存する。このファイルは画像内容を含むため端末内で使用する。通常の結果JSONへ画像や関節座標を含めない。同じFP32 ONNXと入力を、numpyとonnxruntime-gpuが利用できるGPU診断環境で比較する。GPUプロバイダーが利用できない場合にCPU計測へ置き換えない。
```sh
python native_benchmark.py --model rtmw3d-l-fp32.onnx --input rtmw3d-native-inputs.f32 --provider cuda --iterations 200
```
TensorRTを利用できる既存環境では`--provider tensorrt`を指定できる。これは性能診断用であり、ブラウザ推論の製品構成を変更しない。速度計測後、別セッションでGPU・CPUのノード割当てを記録する。
## モデルを維持する条件と代替
元モデル・実装：https://github.com/open-mmlab/mmpose
公式のRTMW3D-L / X評価はCOCO-WholeBody AP 67.8 / 68.0、H3WB MPJPE 56 / 57mmである。異なる評価指標の差と実動画での足の追従品質を混同せず、1mm差から統計的な優位を主張しない。現在のRTMPose3Dモデル一覧ではLとXを確認した。
出典：https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose3d
全133点の相対3Dを維持する場合、RTMW3D-Lを比較基準として残す。腰・膝・足首・踵・足先の追跡を主目的とし、顔と指の詳細を不要にできる場合は、MediaPipe Poseの33点も比較できる。Holisticは33身体点に顔468点と左右各21手点を加えるため、Poseとの処理負荷を別に測る。座標契約が異なり、543点を共通座標系の133点3Dモデルと同一視しない。
出典：https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker
出典：https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/holistic.md
MediaPipe比較ページはPose / HolisticそれぞれLite / Fullを選択できる。Poseは0.5.1675469404、Holisticは0.5.1675471629の公式配布SDKに固定する。このHolistic JSではworld_landmarksが`za`という短縮名で返ることを配布ソースで確認し、その版だけ明示的に対応する。身体33点、足6点の可視性、world33点の取得率も記録する。未検出による見かけの高fpsを追跡成功にしない。検出・追跡を含むパイプライン全体の時間なので、人物検出を含まないRTMW3Dモデル単体とは測定範囲が異なる。三つの静止時点の反復は、連続動画の揺れ・遮蔽復帰・距離精度の検証ではない。
複数台の同期・内部外部校正・共通関節の三角測量を行う場合は、各視点の軽量2Dモデルも候補にできる。RTMPose Halpe26は足首と足部を含み、公式256×192版のt / s / mはそれぞれ3.51 / 5.70 / 13.93Mパラメーター、0.37 / 0.70 / 1.95GFLOPsである。WebGPU速度と足の品質は別途実測する。単眼の2D結果だけで前後移動や足の3D姿勢が完成したとは扱わない。
出典：https://github.com/open-mmlab/mmpose/blob/main/configs/body_2d_keypoint/rtmpose/body8/rtmpose_body8-halpe26.md
