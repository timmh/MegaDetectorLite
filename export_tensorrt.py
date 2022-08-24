import argparse
import cv2
import torch
from yolort.utils import get_image_from_url, read_image_to_tensor
from yolort.utils.image_utils import to_numpy
from yolort.runtime.trt_helper import export_tensorrt_engine
from yolort.runtime import PredictorTRT


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


def main(input_weights, output_weights_onnx, output_weights_tensorrt, batch_size, width, height, score_thresh, nms_thresh, detections_per_img, precision):
    # create dummy input
    input_sample = torch.rand(batch_size, 3, width, height)

    # Export the TensorRT model
    export_tensorrt_engine(
        input_weights,
        score_thresh=score_thresh,
        nms_thresh=nms_thresh,
        onnx_path=output_weights_onnx,
        engine_path=output_weights_tensorrt,
        input_sample=input_sample,
        detections_per_img=detections_per_img,
        precision=precision
    )

    # Test the TensorRT model
    y_runtime = PredictorTRT(output_weights_tensorrt, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    y_runtime.warmup()
    y_runtime.predict("test.jpg")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("input_weights", type=str, help="path to the YOLOv5 weights")
    argparser.add_argument("output_weights_onnx", type=str, help="path to the ONNX output weights")
    argparser.add_argument("output_weights_tensorrt", type=str, help="path to the TensorRT output weights")
    argparser.add_argument("--batch_size", type=int, default=1, help="batch size which the exported model should support")
    argparser.add_argument("--width", type=int, default=1280, help="image width which the exported model should support")
    argparser.add_argument("--height", type=int, default=1280, help="image height which the exported model should support")
    argparser.add_argument("--score_thresh", type=float, default=0.05, help="score threshold")
    argparser.add_argument("--nms_thresh", type=float, default=0.45, help="non-maximum-suppression threshold")
    argparser.add_argument("--detections_per_img", type=int, default=100, help="maximum number of detections per image")
    argparser.add_argument("--precision", type=str, default="fp16", help="precision of the exported model weights")
    args = argparser.parse_args()
    main(args.input_weights, args.output_weights_onnx, args.output_weights_tensorrt, args.batch_size, args.width, args.height, args.score_thresh, args.nms_thresh, args.detections_per_img, args.precision)