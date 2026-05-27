# Triton Client (host venv)

External gRPC client for `yolo11x_ensemble`. Runs on the host, no Docker.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python client.py --image dog.jpg --url localhost:8001 --out output.jpg
```

## Server health check

```bash
curl -sf http://localhost:8000/v2/health/ready && echo OK
curl -s  http://localhost:8000/v2/models/yolo11x_ensemble | python -m json.tool
```
