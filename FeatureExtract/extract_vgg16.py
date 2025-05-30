import os
import time
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler
import joblib

# ========== Cấu hình ==========
input_root = "D:/DeepVoice/Spectrograms"
output_csv = "D:/DeepVoice/Dataset/vgg16_features.csv"
labels = ["real", "fake"]

# ========== Chuẩn bị mô hình VGG16 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
vgg16.classifier = torch.nn.Identity()
vgg16.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
vgg16 = vgg16.to(device)
vgg16.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_vgg_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = vgg16.features(image)
        pooled = vgg16.avgpool(features)
        flattened = torch.flatten(pooled, 1)
    return flattened.cpu().numpy().flatten()

# ========== Trích xuất đặc trưng ==========
start_time = time.time()
data = []
error_log = []

for label in labels:
    folder_path = os.path.join(input_root, label)
    if not os.path.exists(folder_path):
        print(f"⚠️ Không tìm thấy thư mục: {folder_path}")
        continue

    for file_name in tqdm(os.listdir(folder_path), desc=f"Processing {label}"):
        if not file_name.lower().endswith(".png"):
            continue
        file_path = os.path.join(folder_path, file_name)
        try:
            features = extract_vgg_features(file_path)
            data.append([file_name, label] + features.tolist())
        except Exception as e:
            print(f"❌ Lỗi {file_name}: {e}")
            error_log.append((file_name, str(e)))

columns = ["file", "label"] + [f"vgg_{i}" for i in range(512)]
df = pd.DataFrame(data, columns=columns)
scaler = StandardScaler()
X = df.drop(columns=["file", "label"])
X_scaled = scaler.fit_transform(X).astype(np.float32)
df.iloc[:, 2:] = X_scaled

os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df.to_csv(output_csv, index=False)

# Lưu scaler
scaler_path = output_csv.replace(".csv", "_scaler.pkl")
joblib.dump(scaler, scaler_path)

# Ghi log lỗi nếu có
if error_log:
    with open("vgg_error_log.txt", "w", encoding="utf-8") as errf:
        for name, msg in error_log:
            errf.write(f"{name}: {msg}\n")

elapsed_time = time.time() - start_time
print(f"✅ Đã lưu đặc trưng VGG16 tại: {output_csv}")
print(f"🕒 Thời gian chạy: {elapsed_time:.2f} giây")
