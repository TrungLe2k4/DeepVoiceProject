import pandas as pd
from sklearn.model_selection import train_test_split
import os

# ==================== Đường dẫn tới các file CSV gốc ====================
dataset_folder = "D:/DeepVoice/Dataset"

input_files = {
    "fft_mfcc_features.csv": "fft_mfcc",
    "logmel_features.csv": "logmel",
    "vgg16_features.csv": "vgg16"
}

for filename, prefix in input_files.items():
    print(f" Đang xử lý {filename}")
    full_path = os.path.join(dataset_folder, filename)
    if not os.path.exists(full_path):
        print(f" Không tìm thấy file: {full_path}")
        continue

    df = pd.read_csv(full_path)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

    train_path = os.path.join(dataset_folder, f"{prefix}_train.csv")
    test_path = os.path.join(dataset_folder, f"{prefix}_test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f" Đã lưu: {train_path} và {test_path}")

print(" Hoàn tất chia dữ liệu!")    
