# Hướng dẫn sử dụng TRISM với Triton Inference Server

## 1 Cài đặt TRISM
Trước tiên, cần cài đặt thư viện `trism` để có thể kết nối và chạy inference với Triton.

```bash
pip install trism
# Hoặc cài từ GitHub:
pip install git+https://github.com/hieupth/trism
```

## 2 Chạy Triton Server
Trước khi chạy inference, cần đảm bảo **Triton Server** đang chạy với mô hình `super_resolution`.

```bash
docker run --rm --gpus all \
    -p 2000:2000 -p 2001:2001 -p 2002:2002 \
    -v $(pwd)/models:/models \
    nvcr.io/nvidia/tritonserver:23.06-py3 \
    tritonserver --http-port=2000 --grpc-port=2001 --metrics-port=2002 \
                 --model-repository=/models --model-control-mode=explicit
```

## 3 Chạy inference với TRISM
Sau khi Triton Server đã khởi động, có thể chạy script `trism_infer.py` để thực hiện suy luận trên một ảnh đầu vào.

```bash
python trism/trism_infer.py
```

**Kết quả mong đợi:**
- Ảnh `output_trism.jpg` sẽ được lưu vào thư mục.
- Kiểm tra output shape: `(1, 1, 672, 672)`
