import os
import time
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import joblib

# ========== Cấu hình ==========
input_root = "D:/DeepVoice/Audio_Cleaned"
output_csv = "D:/DeepVoice/Dataset/logmel_features.csv"
sample_rate = 16000
duration = 5  # giây
target_length = sample_rate * duration
labels = ["real", "fake"]

# ========== Bộ biến đổi âm thanh ==========
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=1024,
    hop_length=160,
    win_length=400,
    n_mels=128
)
db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)

# ======= Hàm trích xuất Log-Mel features =========
def extract_logmel_features(file_path):
    waveform, sr = torchaudio.load(file_path)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    waveform = waveform[:, :target_length]  # cắt

    mel_spec = mel_transform(waveform)
    log_mel = db_transform(mel_spec)
    log_mel_np = log_mel.squeeze().numpy()

    mean_features = np.mean(log_mel_np, axis=1)
    std_features = np.std(log_mel_np, axis=1)
    return np.concatenate([mean_features, std_features])

# ========== Trích xuất và ghi file ==========
start_time = time.time()
data = []
error_log = []

for label in labels:
    folder_path = os.path.join(input_root, label)
    if not os.path.exists(folder_path):
        print(f"⚠️ Không tìm thấy thư mục: {folder_path}")
        continue

    for file_name in tqdm(os.listdir(folder_path), desc=f"Processing {label}"):
        if not file_name.lower().endswith(".wav"):
            continue
        file_path = os.path.join(folder_path, file_name)
        try:
            features = extract_logmel_features(file_path)
            data.append([file_name, label] + features.tolist())
        except Exception as e:
            print(f"❌ Lỗi {file_name}: {e}")
            error_log.append((file_name, str(e)))

columns = ["file", "label"] + [f"logmel_mean_{i+1}" for i in range(128)] + [f"logmel_std_{i+1}" for i in range(128)]
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
    with open("logmel_error_log.txt", "w", encoding="utf-8") as errf:
        for name, msg in error_log:
            errf.write(f"{name}: {msg}\n")

elapsed_time = time.time() - start_time
print(f"✅ Đã lưu đặc trưng LogMel tại: {output_csv}")
print(f"🕒 Thời gian chạy: {elapsed_time:.2f} giây")