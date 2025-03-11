import io
import numpy as np
import tritonclient.http as httpclient
from PIL import Image
import torchvision.transforms as transforms
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse

app = FastAPI()

# Kết nối với Triton Server
TRITON_SERVER_URL = "localhost:2000"  # Chỉnh lại nếu cần
client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)

# Tiền xử lý ảnh
def preprocess_image(image: Image.Image):
    img_ycbcr = image.convert("YCbCr")
    img_y, img_cb, img_cr = img_ycbcr.split()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    img_y = transform(img_y).unsqueeze(0).numpy()  # Shape: [1, 1, 224, 224]
    return img_y, img_cb, img_cr

# Hậu xử lý ảnh
def postprocess_image(inference_output, img_cb, img_cr):
    img_out_y = Image.fromarray(np.uint8((inference_output[0] * 255.0).clip(0, 255)[0]), mode='L')
    
    # Ghép lại thành ảnh RGB với các kênh Cb và Cr
    final_img = Image.merge(
        "YCbCr", [
            img_out_y,
            img_cb.resize(img_out_y.size, Image.BICUBIC),
            img_cr.resize(img_out_y.size, Image.BICUBIC),
        ]
    ).convert("RGB")
    
    return final_img

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Đọc ảnh từ request
    image = Image.open(io.BytesIO(await file.read()))
    
    # Tiền xử lý ảnh
    img_y, img_cb, img_cr = preprocess_image(image)

    # Định nghĩa input & output cho Triton
    inputs = httpclient.InferInput("input", img_y.shape, "FP32")
    inputs.set_data_from_numpy(img_y, binary_data=True)
    
    outputs = httpclient.InferRequestedOutput("output", binary_data=True)

    # Gửi request đến Triton Inference Server
    results = client.infer(model_name="super_resolution", inputs=[inputs], outputs=[outputs])
    inference_output = results.as_numpy("output")

    # Hậu xử lý ảnh
    final_img = postprocess_image(inference_output, img_cb, img_cr)

    # Trả ảnh về dưới dạng response
    img_io = io.BytesIO()
    final_img.save(img_io, format="JPEG")
    img_io.seek(0)
    
    return StreamingResponse(img_io, media_type="image/jpeg")

# Chạy server với: uvicorn api.main:app --host 0.0.0.0 --port 3000 --reload
