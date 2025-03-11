import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from trism import TritonModel

# Kết nối tới mô hình trên Triton Inference Server
model = TritonModel(
    model="super_resolution",  # Tên model trong Triton
    version=1,                 # Phiên bản model (hoặc 0 để chọn latest)
    url="localhost:2001",      # URL của Triton Server (dùng gRPC)
    grpc=True                  # Sử dụng gRPC thay vì HTTP
)

print("Model Inputs:")
for inp in model.inputs:
    print(f"name: {inp.name}, shape: {inp.shape}, datatype: {inp.dtype}")

print("\nModel Outputs:")
for out in model.outputs:
    print(f"name: {out.name}, shape: {out.shape}, datatype: {out.dtype}")

img = Image.open("/home/tiennv/datnvt/ltnt/superres-infer/acess/test.jpg").convert("YCbCr")
img_y, img_cb, img_cr = img.split()[0], img.split()[1], img.split()[2]  # Lấy 3 kênh

# Resize ảnh và chuyển đổi thành tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

img_y = transform(img_y).unsqueeze(0).numpy()  # Shape: [1, 1, 224, 224]

# Gửi request inference
output = model.run(data=[img_y.astype(np.float32)])  # Chuyển dtype về FP32
print("Inference hoàn tất!")

output_array = output["output"]
print("Output Shape:", output_array.shape)  # [1, 1, 672, 672]

# Chuyển kết quả thành ảnh Grayscale
img_out_y = Image.fromarray(np.uint8((output_array[0] * 255.0).clip(0, 255)[0]), mode='L')

# Ghép lại thành ảnh RGB với các kênh Cb và Cr đã resize
final_img = Image.merge(
    "YCbCr", [
        img_out_y,
        img_cb.resize(img_out_y.size, Image.BICUBIC),
        img_cr.resize(img_out_y.size, Image.BICUBIC),
    ]
).convert("RGB")

# Lưu ảnh kết quả
final_img.save("/home/tiennv/datnvt/ltnt/superres-infer/acess/output_trism.jpg")
print("Ảnh kết quả đã được lưu output_trism.jpg")
