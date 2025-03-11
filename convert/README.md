# Convert ONNX to TensorRT

## 1. Giới thiệu
Hướng dẫn này giúp bạn chuyển đổi mô hình ONNX sang TensorRT (`.plan`) bằng Docker mà không cần cài đặt TensorRT trên máy.

---

## 2. Cấu trúc thư mục
```
superres-infer/
│── models/
│   ├── onnx/
│   │   ├── super_resolution.onnx    # Model ONNX
│   ├── super_resolution/
│   │   ├── 1/
│   │   │   ├── model.plan           # Model TensorRT (sẽ tạo)
│
│── convert/
│   ├── convert_onnx.py       # Script chuyển đổi
│   ├── README.md                    # Hướng dẫn này
```
Lưu ý:
- Đặt file `.onnx` trong `models/onnx/`.
- Model TensorRT (`.plan`) sẽ được lưu trong `models/super_resolution/1/`.

---

## 3. Cách Convert ONNX → TensorRT (`.plan`)

Chạy container với TensorRT 8.6.1:
```bash
docker run --rm --gpus all -it \
    -v $(pwd)/models:/workspace/models \
    nvcr.io/nvidia/tensorrt:23.06-py3 bash
```

Sau khi vào môi trường Docker TensorRT, chạy lệnh chuyển đổi ONNX sang TensorRT:
```bash
trtexec --onnx=/workspace/models/onnx/super_resolution.onnx \
        --saveEngine=/workspace/models/super_resolution/1/model.plan \
        --explicitBatch --verbose
```
Nếu thành công, file `model.plan` sẽ xuất hiện trong `models/super_resolution/1/`.

---

## 4. Kiểm tra Model TensorRT

Chạy inference thử nghiệm:
```bash
trtexec --loadEngine=/workspace/models/super_resolution/1/model.plan --verbose
```
Nếu model chạy đúng, bạn sẽ thấy thời gian inference và thông tin về GPU.

Test với batch size = 0:
```bash
trtexec --loadEngine=/workspace/models/super_resolution/1/model.plan --shapes=input:1x3x224x224 --verbose
```
Nếu gặp lỗi, kiểm tra lại:
- Tên tensor input/output có đúng không? (`trtexec --inspect`)
- Có dùng đúng dynamic shape không?
```bash
trtexec --onnx=/workspace/models/onnx/super_resolution.onnx \
        --saveEngine=/workspace/models/super_resolution/1/model.plan \
        --explicitBatch --verbose \
        --minShapes=input:1x3x224x224 \
        --optShapes=input:4x3x224x224 \
        --maxShapes=input:8x3x224x224
```

---
