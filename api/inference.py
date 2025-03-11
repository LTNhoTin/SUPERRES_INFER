import numpy as np
import tritonclient.http as httpclient
from PIL import Image
import torchvision.transforms as transforms

TRITON_URL = "localhost:2000"  # Đổi từ 8000 sang 2000

# Kết nối với Triton Server
client = httpclient.InferenceServerClient(url=TRITON_URL)

# Kiểm tra xem server có đang chạy không
if not client.is_server_live():
    raise Exception("Triton Server không hoạt động! Hãy kiểm tra lại.")

# Load ảnh và tiền xử lý: Chuyển sang YCbCr và lấy kênh Y (Grayscale)
img = Image.open("/home/tiennv/datnvt/ltnt/superres-infer/acess/test.jpg").convert("YCbCr")
img_y, img_cb, img_cr = img.split()  # Lấy 3 kênh Y, Cb, Cr

# Resize ảnh về 224x224 và chuyển đổi sang tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

img_y = transform(img_y).unsqueeze(0).numpy()  # Shape: [1, 1, 224, 224]

# Định nghĩa input & output cho Triton
inputs = httpclient.InferInput("input", img_y.shape, "FP32")
inputs.set_data_from_numpy(img_y, binary_data=True)

outputs = httpclient.InferRequestedOutput("output", binary_data=True)

# Gửi request inference
try:
    results = client.infer(model_name="super_resolution", inputs=[inputs], outputs=[outputs])
    inference_output = results.as_numpy("output")  # Shape: [1, 1, ?, ?]

    # Hậu xử lý: Chuyển output thành ảnh Grayscale
    img_out_y = Image.fromarray(np.uint8((inference_output[0] * 255.0).clip(0, 255)[0]), mode='L')

    # Ghép lại thành ảnh RGB với các kênh Cb và Cr đã resize
    final_img = Image.merge(
        "YCbCr", [
            img_out_y,
            img_cb.resize(img_out_y.size, Image.BICUBIC),
            img_cr.resize(img_out_y.size, Image.BICUBIC),
        ]
    ).convert("RGB")

    final_img.save("/home/tiennv/datnvt/ltnt/superres-infer/acess/output_super_resolution.jpg")
    print("Inference hoàn tất! Ảnh đã được lưu với tên **output_super_resolution.jpg**")

except Exception as e:
    print(f"Lỗi trong quá trình inference: {e}")
