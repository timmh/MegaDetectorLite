import argparse
import numpy as np
import cv2
import torch
import onnx
from yolort.models import YOLO, YOLOv5
from yolort.utils import get_image_from_url, read_image_to_tensor
from yolort.utils.image_utils import to_numpy
from yolort.runtime.ort_helper import export_onnx
from yolort.runtime import PredictorORT


def get_image_tensor_from_url(url, device):
    image = get_image_from_url(url)
    image = read_image_to_tensor(image, is_half=False)
    image = image.to(device)
    return image


def get_image_tensor_from_filename(filename, device):
    image = cv2.imread(filename)
    image = read_image_to_tensor(image, is_half=False)
    image = image.to(device)
    return image


def main(input_weights, output_weights):

    device = torch.device("cpu")

    size = (640, 640)  # Used for pre-processing
    size_divisible = 64
    score_thresh = 0.35
    nms_thresh = 0.45
    opset_version = 11
    batch_size = 10
    enable_dynamic_batch_size = False

    filename = "test.jpg"
    images = torch.stack([get_image_tensor_from_filename(filename, device)] * batch_size)

    if not enable_dynamic_batch_size:
        model = YOLOv5.load_from_yolov5(
            input_weights,
            size=size,
            size_divisible=size_divisible,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
        )
    else:
        model = YOLO.load_from_yolov5(
            input_weights,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
        )

    model = model.eval()
    model = model.to(device)

    # Predict using PyTorch
    with torch.no_grad():
        out_pytorch = model(images)
    inputs, _ = torch.jit._flatten(images)
    outputs, _ = torch.jit._flatten(out_pytorch)
    inputs = list(map(to_numpy, inputs))
    outputs = list(map(to_numpy, outputs))

    # Export the ONNX model
    export_onnx(model=model, onnx_path=output_weights, opset_version=opset_version, batch_size=batch_size, skip_preprocess=True)

    # Load the ONNX model
    onnx_model = onnx.load(output_weights)

    # Check that the model is well formed
    onnx.checker.check_model(onnx_model)

    # Predict using ONNX model
    y_runtime = PredictorORT(output_weights, device="cpu")
    ort_outs1 = y_runtime.predict(np.stack(inputs)[0])

    # Check whether prediction match
    for i in range(0, len(outputs)):
        torch.testing.assert_close(outputs[i], ort_outs1[i], rtol=1e-04, atol=1e-07)

    print("Exported model has been tested, and the result looks good!")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("input_weights", type=str, help="path to the YOLOv5 weights")
    argparser.add_argument("output_weights", type=str, help="path to the output weights")
    args = argparser.parse_args()
    main(args.input_weights, args.output_weights)