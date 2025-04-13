import os
import librosa
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# === CẤU HÌNH ===
FEATURE_NAME = "fft_mfcc"       # fft_mfcc, logmel, vgg512
MODEL_TYPE = "xgb"               # svm, rf, xgb
USE_RAW_AUDIO = False           # True = dùng Audio/, False = dùng Audio_Cleaned/

RAW_FOLDER = "D:/DeepVoice/Audio/real"
CLEANED_FOLDER = "D:/DeepVoice/Audio_Cleaned/real"
INPUT_FOLDER = RAW_FOLDER if USE_RAW_AUDIO else CLEANED_FOLDER

FEATURE_TO_FOLDER = {
    "vgg512": "VGG16",
    "fft_mfcc": "FFT_MFCC",
    "logmel": "LogMel"
}
MODEL_DIR = f"D:/DeepVoice/Models/{FEATURE_TO_FOLDER[FEATURE_NAME]}"
VGG_FEATURE_CSV = "D:/DeepVoice/Dataset/vgg512_features.csv"
PREDICTIONS_DIR = "D:/DeepVoice/Predictions"

# === TẠO FOLDER LƯU KẾT QUẢ PHÂN LOẠI (real / fake) ===
folder_type = "real" if "real" in INPUT_FOLDER.lower() else "fake"
OUTPUT_DIR = os.path.join(PREDICTIONS_DIR, folder_type)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === HỖ TRỢ LOAD & PAD AUDIO ===
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

def extract_vgg(file_path, feature_csv):
    df = pd.read_csv(feature_csv)
    file_name = os.path.basename(file_path).replace(".wav", ".png")
    row = df[df['file'] == file_name]
    if row.empty:
        print(f"[⚠️] Không tìm thấy ảnh spectrogram tương ứng cho: {file_name}")
        return None
    X = row.drop(columns=["file", "label"]).values
    return X

# === DỰ ĐOÁN 1 FILE ===
def predict_single(file_path, X, feature_name, model_type):
    scaler_path = os.path.join(MODEL_DIR, f"{feature_name}_scaler.pkl")
    model_path = os.path.join(MODEL_DIR, f"{feature_name}_{model_type}.pkl")
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    # Đặt đúng tên cột theo đặc trưng
    if feature_name == "fft_mfcc":
        columns = ["fft_mean", "fft_std", "mel_mean", "mel_std"] + [f"mfcc_{i+1}" for i in range(13)]
    elif feature_name == "logmel":
        columns = [f"logmel_{i+1}" for i in range(X.shape[1])]
    elif feature_name == "vgg512":
        columns = [f"vgg_{i+1}" for i in range(X.shape[1])]
    else:
        raise ValueError("FEATURE_NAME không hợp lệ.")

    X_df = pd.DataFrame(X, columns=columns)
    X_scaled = scaler.transform(X_df)

    prediction = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    prob_real, prob_fake = proba[0], proba[1]

    label = "Fake" if prediction == 1 else "Real"
    file_name = os.path.basename(file_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ghi kết quả text
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

    # Vẽ biểu đồ
    plt.bar(["Real", "Fake"], [prob_real, prob_fake], color=["green", "red"])
    plt.ylim(0, 1)
    plt.ylabel("Xác suất")
    plt.title(f"{feature_name.upper()} - {file_name} - {label}")
    plt.tight_layout()
    img_path = os.path.join(OUTPUT_DIR, f"{file_name}_{feature_name}_{model_type}_proba.png")
    plt.savefig(img_path)
    plt.close()

    print(f"{file_name} → {label} | Fake: {prob_fake:.4f} | Real: {prob_real:.4f}")
    return {
        "file": file_name,
        "predict": label,
        "prob_fake": prob_fake,
        "prob_real": prob_real
    }

# === DUYỆT TẤT CẢ FILE .WAV TRONG THƯ MỤC ===
results = []
for file in os.listdir(INPUT_FOLDER):
    if file.endswith(".wav"):
        path = os.path.join(INPUT_FOLDER, file)
        try:
            if FEATURE_NAME == "fft_mfcc":
                X = extract_fft_mfcc(path)
            elif FEATURE_NAME == "logmel":
                X = extract_logmel(path)
            elif FEATURE_NAME == "vgg512":
                X = extract_vgg(path, VGG_FEATURE_CSV)
                if X is None:
                    continue
            else:
                raise ValueError("FEATURE_NAME không hợp lệ.")

            result = predict_single(path, X, FEATURE_NAME, MODEL_TYPE)
            results.append(result)
        except Exception as e:
            print(f"[❌] Lỗi với {file} | {type(e).__name__}: {e}")

# === LƯU KẾT QUẢ TỔNG HỢP CSV ===
df = pd.DataFrame(results)
summary_path = os.path.join(OUTPUT_DIR, f"summary_{FEATURE_NAME}_{MODEL_TYPE}.csv")
df.to_csv(summary_path, index=False, encoding="utf-8")
print(f"\n✅ Đã lưu kết quả tổng hợp tại: {summary_path}")
