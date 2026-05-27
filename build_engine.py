#!/usr/bin/env python3
"""
Export yolo11x.pt -> ONNX -> TensorRT (model.plan).

Intended to run INSIDE the Triton container so TRT versions match the server
exactly. Invoked via `docker compose run --rm build`.

Usage (from triton_models/ on the host):
    docker compose run --rm build \
        python build_engine.py --weights yolo11x.pt --fp16
"""

import argparse
import os
import subprocess
from pathlib import Path

from ultralytics import YOLO


def build(weights: str, output: str, batch: int, imgsz: int, fp16: bool, workspace_gb: int):
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Ultralytics blanks CUDA_VISIBLE_DEVICES to force CPU export; save and restore.
    saved_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")

    print(f"[1/2] Exporting {weights} -> ONNX (dynamic batch, imgsz={imgsz})")
    model = YOLO(weights)
    onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=True,
        simplify=True,
        opset=17,
    )
    print(f"      ONNX: {onnx_path}")

    if saved_cvd is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = saved_cvd

    print(f"[2/2] Building TensorRT engine -> {out_path}")
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={out_path}",
        f"--minShapes=images:1x3x{imgsz}x{imgsz}",
        f"--optShapes=images:{batch}x3x{imgsz}x{imgsz}",
        f"--maxShapes=images:{batch}x3x{imgsz}x{imgsz}",
        f"--memPoolSize=workspace:{workspace_gb * 1024}",
    ]
    if fp16:
        cmd.append("--fp16")

    print("      Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"\nDone: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="yolo11x.pt")
    p.add_argument("--output", default="yolo11x/1/model.plan")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--workspace-gb", type=int, default=4)
    args = p.parse_args()

    build(args.weights, args.output, args.batch, args.imgsz, args.fp16, args.workspace_gb)
