import requests

# Địa chỉ của API
url = "http://localhost:3000/predict/"

file_path = "/home/tiennv/datnvt/ltnt/superres-infer/acess/test.jpg"

# Gửi request POST với ảnh
with open(file_path, "rb") as image:
    files = {"file": image}
    response = requests.post(url, files=files)

if response.status_code == 200:
    with open("/home/tiennv/datnvt/ltnt/superres-infer/acess/output.jpg", "wb") as f:
        f.write(response.content)
    print("Inference thành công! Ảnh đã được lưu: acess/output.jpg")
else:
    print("Lỗi:", response.text)
