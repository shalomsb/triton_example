# YOLO11x Triton Ensemble

Single docker for the server (with the engine built inside it), one host venv for the client.

```
yolo11x_ensemble
  ├─ yolo11x_preprocess  (Python: letterbox + BGR->RGB + normalize + HWC->CHW)
  ├─ yolo11x             (TensorRT: model.plan)
  └─ yolo11x_postprocess (Python: decode + descale + class-aware NMS)
```

## Layout

```
.
├── launch.sh              -- build engine / run server
├── Dockerfile             -- nvcr.io/nvidia/tritonserver:26.04-py3 + opencv + ultralytics
├── build_engine.py        -- runs inside the container, exports yolo11x.pt -> model.plan
├── yolo11x.pt             -- you provide this
├── triton_models/         -- mounted into the container at /models (ro)
│   ├── yolo11x_ensemble/{config.pbtxt, 1/}
│   ├── yolo11x_preprocess/{config.pbtxt, 1/model.py}
│   ├── yolo11x/{config.pbtxt, 1/model.plan}     <-- built by launch.sh -b
│   └── yolo11x_postprocess/{config.pbtxt, 1/model.py}
└── client/                -- host venv (opencv + PIL + tritonclient[grpc])
    ├── client.py
    ├── requirements.txt
    └── README.md
```

## Quickstart

```bash
# 0. Drop yolo11x.pt next to launch.sh (or set WEIGHTS=...)
#    https://github.com/ultralytics/assets/releases

# 1. Build image + engine (one shot)
./launch.sh -b

# 2. Run server
./launch.sh -r          # blocks; ctrl-c to stop

# 3. Client (separate terminal, on the host)
cd client
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python client.py --image dog.jpg --url localhost:8001
```

## I/O

**Input** `RAW_IMAGE`: `uint8 [B, H, W, 3]`, BGR.

**Outputs:**
- `DETECTIONS`: `fp32 [B, N, 6]` -> `[x1, y1, x2, y2, score, class_id]` in original coords
- `NUM_DETECTIONS`: `int32 [B, 1]` -> valid count per image, rest zero-padded

## Notes

- **Engine portability**: a `.plan` built on RTX 5070 Ti (sm_120) will not load on
  Orin (sm_87). Rebuild on the target. Same goes for TRT version bumps.
- **Image build is cached**: `launch.sh -b` only rebuilds the docker image if
  it's missing. Force rebuild with `docker rmi yolo11x-triton:local`.
- **Re-export weights**: delete `triton_models/yolo11x/1/model.plan` and run
  `-b` again. The Dockerfile doesn't need to change.
- **Ports**: 8000 HTTP, 8001 gRPC (client uses this), 8002 Prometheus.
