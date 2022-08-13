from collections import OrderedDict
import time
import os
import glob
import re
import cv2
import torch
import matplotlib.pyplot as plt
from yolort.runtime import PredictorTRT
from yolort.utils import read_image_to_tensor
from yolort.runtime import PredictorTRT


def get_image_tensor_from_filename(filename, device, is_half=False):
    image = cv2.imread(filename)
    image = read_image_to_tensor(image, is_half=is_half)
    image = image.to(device)
    return image


def main():
    n_images = 64
    n_iters = 100

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    images = [get_image_tensor_from_filename("test.jpg", device=device, is_half=True)] * n_images

    times = {}
    for weights in sorted(glob.glob(os.path.join("weights", "*_640_*.engine"))):
        batch_size = int(re.search(r"_(\d+).engine", os.path.basename(weights))[1])
        assert n_images % batch_size == 0

        predictor = PredictorTRT(weights, device=device)
        samples = predictor.transform(images)[0].tensors
        predictor.warmup()
        start = time.perf_counter()
        for _ in range(n_iters):
            for i in range(n_images // batch_size):
                predictor(samples[i * batch_size : (i + 1) * batch_size].clone())
        end = time.perf_counter()

        times[batch_size] = ((end - start) / 1000) / (n_images * n_iters)

    times = OrderedDict(sorted(times.items(), key=lambda kv: kv[0]))
    
    plt.bar(range(len(times)), times.values())
    plt.xticks(range(len(times)), list(times.keys()))
    plt.xlabel("Batch Size")
    plt.ylabel("Inference time [ms]/[image]")
    plt.savefig("benchmark.png", bbox_inches="tight", dpi=300, transparent=True)
    plt.show()


if __name__ == "__main__":
    main()