# Triton Client (host venv)

External gRPC client for `yolo11x_ensemble`. Runs on the host, no Docker.

## Setup & Run

```bash
cd client
uv run client.py --image dog.jpg --url localhost:8001 --out output.jpg
```

## Server health check

```bash
curl -sf http://localhost:8000/v2/health/ready && echo OK
curl -s  http://localhost:8000/v2/models/yolo11x_ensemble | python -m json.tool
```
