# Super Resolution Inference with Triton Server & FastAPI

## Giới thiệu
Dự án này sử dụng **Triton Inference Server** để triển khai mô hình **Super Resolution**, đồng thời cung cấp **API inference** thông qua **FastAPI** và **trism**.

- **Convert model** từ ONNX sang TensorRT.
- **Triển khai model** bằng **Triton Inference Server**.
- **Xây dựng API** để gửi request inference thông qua **FastAPI**.
- **Test model inference** bằng **trism**.

---

