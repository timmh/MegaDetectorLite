# MegaDetectorLite

This repository provides scripts to convert the weights of [MegaDetector 5.0](https://github.com/microsoft/CameraTraps/releases/tag/v5.0) into the ONNX and TensorRT formats to simplify the usage in embedded devices. [yolort](https://github.com/zhiqwang/yolov5-rt-stack/) further optimizes the runtime performance by embedding the pre- and post-processing steps into the models.

## Converted Weights
Already converted weights can be found under [releases](https://github.com/timmh/MegaDetectorLite/releases).

## TODOs
- [ ] enable batch sizes larger than 1 and flexible spatial dimensions for the TensorRT models
- [ ] perform benchmarks on some embedded devices
- [ ] automatically compare detection performance of derived models

## Acknowledgments
The [test.jpg](test.jpg) file is provided by the [Caltech Camera Traps](https://lila.science/datasets/caltech-camera-traps) dataset under the terms of the [Community Data License Agreement (permissive variant)](https://cdla.dev/permissive-1-0/).
