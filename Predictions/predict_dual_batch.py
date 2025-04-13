import os
import librosa
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# === CẤU HÌNH ===
FEATURE_NAME = "vgg512"     # fft_mfcc, logmel, vgg512
MODEL_TYPE = "xgb"             # svm, rf, xgb
USE_RAW_AUDIO = False         # True = dùng Audio/, False = dùng Audio_Cleaned/

BASE_RAW = "D:/DeepVoice/Audio"
BASE_CLEANED = "D:/DeepVoice/Audio_Cleaned"
INPUT_BASE = BASE_RAW if USE_RAW_AUDIO else BASE_CLEANED

FEATURE_TO_FOLDER = {
    "vgg512": "VGG16",
    "fft_mfcc": "FFT_MFCC",
    "logmel": "LogMel"
}
MODEL_DIR = f"D:/DeepVoice/Models/{FEATURE_TO_FOLDER[FEATURE_NAME]}"
VGG_FEATURE_CSV = "D:/DeepVoice/Dataset/vgg512_features.csv"
OUTPUT_BASE = "D:/DeepVoice/Predictions"

# === LOAD VÀ PAD ÂM THANH ===
def load_and_pad(file_path, target_sr=16000, duration=5):
    y, sr = librosa.load(file_path, sr=target_sr)
    target_len = target_sr * duration
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y, target_sr

# === TRÍCH XUẤT ĐẶC TRƯNG ===
def extract_fft_mfcc(file_path):
    y, sr = load_and_pad(file_path)
    fft = np.abs(np.fft.fft(y))[:len(y)//2]
    fft_mean = np.mean(fft)
    fft_std = np.std(fft)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_mean = np.mean(mel)
    mel_std = np.std(mel)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs = np.mean(mfcc, axis=1)
    return np.array([fft_mean, fft_std, mel_mean, mel_std] + list(mfccs)).reshape(1, -1)

def extract_logmel(file_path):
    y, sr = load_and_pad(file_path)
    logmel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr))
    logmel_mean = np.mean(logmel, axis=1)
    return logmel_mean.reshape(1, -1)

def extract_vgg(file_path):
    df = pd.read_csv(VGG_FEATURE_CSV)
    file_name = os.path.basename(file_path).replace(".wav", ".png")
    row = df[df['file'] == file_name]
    if row.empty:
        print(f"[⚠️] Không tìm thấy ảnh spectrogram tương ứng cho: {file_name}")
        return None
    return row.drop(columns=["file", "label"]).values

# === DỰ ĐOÁN ===
def predict_single(file_path):
    scaler_path = os.path.join(MODEL_DIR, f"{FEATURE_NAME}_scaler.pkl")
    model_path = os.path.join(MODEL_DIR, f"{FEATURE_NAME}_{MODEL_TYPE}.pkl")
    
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    if FEATURE_NAME == "fft_mfcc":
        X = extract_fft_mfcc(file_path)
        feature_names = ["fft_mean", "fft_std", "mel_mean", "mel_std"] + [f"mfcc_{i+1}" for i in range(13)]
    elif FEATURE_NAME == "logmel":
        X = extract_logmel(file_path)
        feature_names = [f"logmel_{i+1}" for i in range(X.shape[1])]
    elif FEATURE_NAME == "vgg512":
        X = extract_vgg(file_path)
        if X is None:
            return None
        feature_names = [f"vgg_{i}" for i in range(X.shape[1])]
    else:
        raise ValueError("Tên đặc trưng không hợp lệ.")

    X_df = pd.DataFrame(X, columns=feature_names)
    X_scaled = scaler.transform(X_df)

    prediction = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    label = "Fake" if prediction == 1 else "Real"
    return label, proba[0], proba[1]

# === THỰC THI SONG SONG ===
real_folder = os.path.join(INPUT_BASE, "real")
fake_folder = os.path.join(INPUT_BASE, "fake")
real_files = sorted([f for f in os.listdir(real_folder) if f.endswith(".wav")])
fake_files = sorted([f for f in os.listdir(fake_folder) if f.endswith(".wav")])
max_len = max(len(real_files), len(fake_files))

rows = []
for i in range(max_len):
    row = {}
    if i < len(real_files):
        real_path = os.path.join(real_folder, real_files[i])
        try:
            label, prob_real, prob_fake = predict_single(real_path)
            row.update({
                "real_file": real_files[i],
                "real_predict": label,
                "real_prob_real": round(prob_real, 4),
                "real_prob_fake": round(prob_fake, 4)
            })
        except Exception as e:
            print(f"[❌] Lỗi real: {real_files[i]} | {type(e).__name__}: {e}")
    if i < len(fake_files):
        fake_path = os.path.join(fake_folder, fake_files[i])
        try:
            label, prob_real, prob_fake = predict_single(fake_path)
            row.update({
                "fake_file": fake_files[i],
                "fake_predict": label,
                "fake_prob_real": round(prob_real, 4),
                "fake_prob_fake": round(prob_fake, 4)
            })
        except Exception as e:
            print(f"[❌] Lỗi fake: {fake_files[i]} | {type(e).__name__}: {e}")
    rows.append(row)

# === GHI FILE CSV SONG SONG ===
df = pd.DataFrame(rows)
csv_path = os.path.join(OUTPUT_BASE, f"summary_dual_{FEATURE_NAME}_{MODEL_TYPE}.csv")
df.to_csv(csv_path, index=False, encoding="utf-8")
print(f"\n✅ Đã lưu file CSV dual tại: {csv_path}")
