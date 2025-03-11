# Hướng dẫn chạy API với Triton Inference Server và FastAPI

## 1. Chạy Triton Server trước

```bash
docker run --rm --gpus all \
    -p 2000:2000 -p 2001:2001 -p 2002:2002 \
    -v $(pwd)/models:/models \
    nvcr.io/nvidia/tritonserver:23.06-py3 \
    tritonserver --http-port=2000 --grpc-port=2001 --metrics-port=2002 \
                 --model-repository=/models --model-control-mode=explicit
```

## 2. Chạy FastAPI

```bash
uvicorn api.main:app --host 0.0.0.0 --port 3000
```

## 3. Test bằng Postman hoặc `curl`

```bash
curl -X POST "http://localhost:3000/predict/" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test.jpg" --output output.jpg
```

- `test.jpg`: ảnh input cần xử lý
- `output.jpg`: ảnh output từ model super resolution
