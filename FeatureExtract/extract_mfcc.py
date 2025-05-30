import os
import time
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import joblib

# ========== Cấu hình ==========
input_root = "D:/DeepVoice/Audio_Cleaned"
output_csv = "D:/DeepVoice/Dataset/fft_mfcc_features.csv"
sample_rate = 16000
duration = 5  # giây
target_length = sample_rate * duration
labels = ["real", "fake"]

# ========== Hàm trích xuất đặc trưng ==========
def extract_fft_mfcc(file_path):
    audio, sr = librosa.load(file_path, sr=sample_rate)
    audio = audio[:target_length]
    
    fft = np.abs(np.fft.rfft(audio))
    fft_mean = np.mean(fft)
    fft_std = np.std(fft)
    
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_mean = np.mean(mel_spec)
    mel_std = np.std(mel_spec)
    
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
    mfcc_mean = np.mean(mfcc, axis=1).tolist()
    delta_mean = np.mean(mfcc_delta, axis=1).tolist()
    delta2_mean = np.mean(mfcc_delta2, axis=1).tolist()
    
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))
    
    # Đặc trưng phụ
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_mean = np.mean(chroma, axis=1).tolist()  # 12 chiều
    
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    spectral_contrast_mean = np.mean(spectral_contrast, axis=1).tolist()  # 7 chiều
    
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
    tonnetz_mean = np.mean(tonnetz, axis=1).tolist()  # 6 chiều
    
    return [fft_mean, fft_std, mel_mean, mel_std, spec_centroid, spec_bw, zcr] \
           + mfcc_mean + delta_mean + delta2_mean \
           + chroma_mean + spectral_contrast_mean + tonnetz_mean

# Cập nhật danh sách cột khi tạo DataFrame:

columns = [
    "file", "label", "fft_mean", "fft_std", "mel_mean", "mel_std",
    "spectral_centroid", "spectral_bandwidth", "zcr"
] + [f"mfcc_{i+1}" for i in range(13)] + [f"delta_{i+1}" for i in range(13)] + [f"delta2_{i+1}" for i in range(13)] \
  + [f"chroma_{i+1}" for i in range(12)] + [f"spectral_contrast_{i+1}" for i in range(7)] + [f"tonnetz_{i+1}" for i in range(6)]

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
            features = extract_fft_mfcc(file_path)
            data.append([file_name, label] + features)
        except Exception as e:
            print(f"❌ Lỗi {file_name}: {e}")
            error_log.append((file_name, str(e)))

# Cột đặc trưng
columns = [
    "file", "label",
    "fft_mean", "fft_std", "mel_mean", "mel_std",
    "spectral_centroid", "spectral_bandwidth", "zcr"
] + [f"mfcc_{i+1}" for i in range(13)] \
  + [f"delta_{i+1}" for i in range(13)] \
  + [f"delta2_{i+1}" for i in range(13)] \
  + [f"chroma_{i+1}" for i in range(12)] \
  + [f"spectral_contrast_{i+1}" for i in range(7)] \
  + [f"tonnetz_{i+1}" for i in range(6)]

# Ghi DataFrame
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
    with open("mfcc_error_log.txt", "w", encoding="utf-8") as errf:
        for name, msg in error_log:
            errf.write(f"{name}: {msg}\n")

elapsed_time = time.time() - start_time
print(f"✅ Đã lưu đặc trưng FFT/MFCC tại: {output_csv}")
print(f"🕒 Thời gian chạy: {elapsed_time:.2f} giây")
