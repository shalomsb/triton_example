# Custom Triton image: base + opencv (for Python backends) + ultralytics (for engine export).
# Same image is used to serve AND to one-shot build the model.plan.
#
#   docker compose build
#
FROM nvcr.io/nvidia/tritonserver:26.04-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
        libxcb1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        opencv-python-headless==4.10.0.84 \
        ultralytics \
        onnx \
        onnxslim \
        onnxruntime

# Base image already has trtexec on PATH and TensorRT Python bindings installed.
# Default entrypoint (tritonserver) is preserved from the base image.
