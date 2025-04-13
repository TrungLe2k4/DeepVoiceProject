import os
import librosa
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# ====== CẤU HÌNH ======
FEATURE_NAME = "vgg512"    # "fft_mfcc", "logmel", "vgg512"
MODEL_TYPE = "svm"          # "svm" hoặc "rf"
AUDIO_PATH = "D:/DeepVoice/Audio_Cleaned/real/real_0250.wav"

MODEL_DIR = f"D:/DeepVoice/Models/{FEATURE_NAME.upper()}"
VGG_FEATURE_CSV = "D:/DeepVoice/Dataset/vgg512_features.csv"
OUTPUT_DIR = "D:/DeepVoice/Predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====== HÀM TRÍCH XUẤT ĐẶC TRƯNG ======
def extract_fft_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=None)
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
    y, sr = librosa.load(file_path, sr=None)
    logmel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr))
    logmel_mean = np.mean(logmel, axis=1)
    return logmel_mean.reshape(1, -1)

def extract_vgg(file_path, feature_csv):
    df = pd.read_csv(feature_csv)
    file_name = os.path.basename(file_path).replace(".wav", ".png")  
    row = df[df['file'] == file_name]
    if row.empty:
        raise ValueError(f"Không tìm thấy đặc trưng VGG512 của file {file_name}")
    X = row.drop(columns=["file", "label"]).values
    return X


# ====== DỰ ĐOÁN & LƯU KẾT QUẢ ======
def predict_and_save(X, feature_name, model_type):
    scaler_path = os.path.join(MODEL_DIR, f"{feature_name}_scaler.pkl")
    model_path = os.path.join(MODEL_DIR, f"{feature_name}_{model_type}.pkl")

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    prob_real, prob_fake = proba[0], proba[1]

    label = "Fake" if prediction == 1 else "Real"
    file_name = os.path.basename(AUDIO_PATH)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # === In kết quả ra màn hình
    print("\n==== KẾT QUẢ DỰ ĐOÁN ====")
    print(f"File: {file_name}")
    print(f"Dự đoán: {label}")
    print(f"Xác suất Fake: {prob_fake:.4f}")
    print(f"Xác suất Real: {prob_real:.4f}")

    # === Lưu kết quả ra file txt
    result_text = (
        f"File: {file_name}\n"
        f"Thời gian: {timestamp}\n"
        f"Đặc trưng: {feature_name}\n"
        f"Mô hình: {model_type}\n"
        f"Dự đoán: {label}\n"
        f"Xác suất Fake: {prob_fake:.4f}\n"
        f"Xác suất Real: {prob_real:.4f}\n"
    )
    result_path = os.path.join(OUTPUT_DIR, f"{file_name}_{feature_name}_{model_type}_result.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(result_text)

    # === Vẽ biểu đồ xác suất
    plt.bar(["Real", "Fake"], [prob_real, prob_fake], color=["green", "red"])
    plt.ylim(0, 1)
    plt.ylabel("Xác suất")
    plt.title(f"{feature_name.upper()} - {file_name} - {label}")
    plt.grid(axis="y")
    plt.tight_layout()
    img_path = os.path.join(OUTPUT_DIR, f"{file_name}_{feature_name}_{model_type}_proba.png")
    plt.savefig(img_path)
    plt.close()

    print(f"> Đã lưu kết quả tại: {result_path}")
    print(f"> Đã lưu biểu đồ tại: {img_path}")

# ====== CHẠY ======
if FEATURE_NAME == "fft_mfcc":
    X = extract_fft_mfcc(AUDIO_PATH)
elif FEATURE_NAME == "logmel":
    X = extract_logmel(AUDIO_PATH)
elif FEATURE_NAME == "vgg512":
    X = extract_vgg(AUDIO_PATH, VGG_FEATURE_CSV)
else:
    raise ValueError("FEATURE_NAME không hợp lệ.")

predict_and_save(X, FEATURE_NAME, MODEL_TYPE)
