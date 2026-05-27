#!/usr/bin/env python3
"""
Test client for the yolo11x_ensemble.

Usage:
    python client.py --image dog.jpg --url localhost:8001
"""

import argparse
import sys

import cv2
import numpy as np
import tritonclient.grpc as grpcclient


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--url", default="localhost:8001")
    parser.add_argument("--model", default="yolo11x_ensemble")
    parser.add_argument("--out", default="output.jpg")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"Could not read image: {args.image}", file=sys.stderr)
        sys.exit(1)

    # Add batch dim: [1, H, W, 3] uint8
    batched = np.expand_dims(img, 0).astype(np.uint8)

    client = grpcclient.InferenceServerClient(url=args.url)

    inp = grpcclient.InferInput("RAW_IMAGE", batched.shape, "UINT8")
    inp.set_data_from_numpy(batched)

    outputs = [
        grpcclient.InferRequestedOutput("DETECTIONS"),
        grpcclient.InferRequestedOutput("NUM_DETECTIONS"),
    ]

    result = client.infer(args.model, inputs=[inp], outputs=outputs)

    dets = result.as_numpy("DETECTIONS")          # [1, N, 6]
    num = result.as_numpy("NUM_DETECTIONS")       # [1, 1]

    n = int(num[0, 0])
    print(f"Found {n} detections")

    vis = img.copy()
    for i in range(n):
        x1, y1, x2, y2, score, cls = dets[0, i]
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cls = int(cls)
        label = f"{COCO_CLASSES[cls] if cls < len(COCO_CLASSES) else cls} {score:.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, label, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        print(f"  [{i}] {label} @ ({x1},{y1})-({x2},{y2})")

    cv2.imwrite(args.out, vis)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
