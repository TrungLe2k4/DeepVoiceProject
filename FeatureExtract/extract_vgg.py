import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Thư mục chứa ảnh spectrogram
spectrogram_folder = "D:/DeepVoice/Spectrograms"
output_csv = "D:/DeepVoice/Dataset/vgg512_features.csv"

# Tạo thư mục chứa đặc trưng nếu chưa tồn tại
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# Kiểm tra thư mục chứa spectrograms
if not os.path.exists(spectrogram_folder):
    print(f"Thư mục {spectrogram_folder} không tồn tại!")
    exit()

# Load mô hình VGG16 (bỏ phần fully connected - lấy convolutional features)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
vgg16.classifier = torch.nn.Identity()  # Loại bỏ phần fully connected
vgg16.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))  # Đảm bảo kích thước đầu ra 1x1
vgg16 = vgg16.to(device)
vgg16.eval()

# Tiền xử lý ảnh cho VGG16
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Hàm trích xuất đặc trưng (512 chiều)
def extract_vgg_features(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = vgg16.features(image)
        pooled = vgg16.avgpool(features)  # [B, 512, 1, 1]
        flattened = torch.flatten(pooled, 1)  # [B, 512]
    return flattened.cpu().numpy().flatten()

# Duyệt qua từng thư mục 'fake' và 'real'
data = []
categories = ['fake', 'real']

for category in categories:
    category_path = os.path.join(spectrogram_folder, category)
    if not os.path.exists(category_path):
        print(f"Không tìm thấy thư mục: {category_path}")
        continue

    image_files = [f for f in os.listdir(category_path) if f.endswith('.png')]
    print(f"Số lượng ảnh {category}: {len(image_files)}")

    for file_name in tqdm(image_files, desc=f"Trích xuất VGG16 - {category}"):
        file_path = os.path.join(category_path, file_name)
        image = Image.open(file_path).convert("RGB")
        features = extract_vgg_features(image)
        data.append([file_name, category] + features.tolist())

# Tạo DataFrame và lưu ra CSV
columns = ["file", "label"] + [f"vgg_{i}" for i in range(512)]
df = pd.DataFrame(data, columns=columns)
df.to_csv(output_csv, index=False)

print(f"Đã lưu đặc trưng VGG16 (512 chiều) vào: {output_csv}")
