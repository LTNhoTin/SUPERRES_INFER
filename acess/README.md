# Hướng dẫn chạy Super Resolution với Triton Inference Server

## 1. Chạy Triton Inference Server
Trước khi thực hiện suy luận (inference), bạn cần chạy Triton Server để host mô hình.

### Chạy Triton Server:
```bash
docker run --rm --gpus all \
    -p 2000:2000 -p 2001:2001 -p 2002:2002 \
    -v $(pwd)/models:/models \
    nvcr.io/nvidia/tritonserver:23.06-py3 \
    tritonserver --http-port=2000 --grpc-port=2001 --metrics-port=2002 \
                 --model-repository=/models --model-control-mode=explicit
```

### Kiểm tra xem mô hình đã được load chưa:
```bash
curl -X GET localhost:2000/v2/models
```

### Nếu mô hình chưa load, có thể load thủ công:
```bash
curl -X POST localhost:2000/v2/repository/models/super_resolution/load
```

---

## 2. Chạy Inference
Sau khi Triton Server đã hoạt động, chạy script `inference.py` để thực hiện suy luận.

### Cài đặt các thư viện cần thiết:
```bash
pip install tritonclient[http] numpy pillow torchvision
```

### Thực hiện suy luận:
```bash
python /home/tiennv/datnvt/ltnt/superres-infer/api/inference.py
```

### Lưu ý:
- Đảm bảo có ảnh đầu vào `test.jpg` trong thư mục `/home/tiennv/datnvt/ltnt/superres-infer/api/acess/`
- Ảnh kết quả sẽ được lưu với tên `output_super_resolution.jpg`

---

## 3. Kết luận
Sau khi chạy thành công, bạn sẽ có ảnh đã được xử lý bằng mô hình Super Resolution và có thể kiểm tra kết quả.
