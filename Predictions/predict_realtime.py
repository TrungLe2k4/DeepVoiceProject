import os
import librosa
import numpy as np
import joblib
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import argparse

# ====== CẤU HÌNH ======
FEATURE_NAME = "fft_mfcc"  # fft_mfcc, logmel, vgg512
MODEL_TYPE = "svm"         # svm, rf, xgb
VGG_FEATURE_CSV = "D:/DeepVoice/Dataset/vgg512_features.csv"
MODEL_DIR = f"D:/DeepVoice/Models/{FEATURE_NAME.upper()}"
OUTPUT_DIR = "D:/DeepVoice/Predictions"
TEMP_AUDIO_PATH = "D:/DeepVoice/temp_realtime.wav"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====== TRÍCH XUẤT ĐẶC TRƯNG ======
def extract_fft_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=None)
    fft = np.abs(np.fft.fft(y))[:len(y)//2]
    return np.array([
        np.mean(fft),
        np.std(fft),
        np.mean(librosa.feature.melspectrogram(y=y, sr=sr)),
        np.std(librosa.feature.melspectrogram(y=y, sr=sr)),
        *np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    ]).reshape(1, -1)

def extract_logmel(file_path):
    y, sr = librosa.load(file_path, sr=None)
    logmel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr))
    return np.mean(logmel, axis=1).reshape(1, -1)

def extract_vgg(file_path, feature_csv):
    df = pd.read_csv(feature_csv)
    file_name = os.path.basename(file_path).replace(".wav", ".png")
    row = df[df['file'] == file_name]
    if row.empty:
        raise ValueError(f"Không tìm thấy đặc trưng VGG512 của {file_name}")
    return row.drop(columns=["file", "label"]).values

# ====== DỰ ĐOÁN & VẼ ======
def predict_and_plot(X, feature_name, model_type, file_name):
    scaler = joblib.load(os.path.join(MODEL_DIR, f"{feature_name}_scaler.pkl"))
    model = joblib.load(os.path.join(MODEL_DIR, f"{feature_name}_{model_type}.pkl"))
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    label = "Fake" if pred == 1 else "Real"

    print(f"{file_name} → {label} | Real: {proba[0]:.4f} | Fake: {proba[1]:.4f}")

    plt.bar(["Real", "Fake"], proba, color=["green", "red"])
    plt.title(f"{file_name} - {label}")
    plt.ylim(0, 1)
    plt.tight_layout()
    img_path = os.path.join(OUTPUT_DIR, f"{file_name}_{feature_name}_{model_type}_proba.png")
    plt.savefig(img_path)
    plt.close()

    return {
        "file": file_name,
        "predict": label,
        "prob_real": proba[0],
        "prob_fake": proba[1]
    }

# ====== GHI ÂM MIC ======
def record_audio(duration=5, sr=16000):
    print("\U0001F399️ Đang ghi âm...")
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    sf.write(TEMP_AUDIO_PATH, recording, sr)
    print("✅ Đã ghi âm xong.")
    return TEMP_AUDIO_PATH

# ====== MAIN ======
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mic", action="store_true", help="Dự đoán từ microphone")
    parser.add_argument("--folder", type=str, help="Thư mục chứa file .wav để dự đoán")
    args = parser.parse_args()

    files = []
    results = []

    if args.mic:
        mic_path = record_audio()
        files.append(mic_path)
    elif args.folder:
        files = [os.path.join(args.folder, f) for f in os.listdir(args.folder) if f.endswith(".wav")]
    else:
        print("❗ Hãy dùng --mic để ghi âm hoặc --folder PATH để dự đoán từ thư mục.")
        return

    for file_path in files:
        try:
            if FEATURE_NAME == "fft_mfcc":
                X = extract_fft_mfcc(file_path)
            elif FEATURE_NAME == "logmel":
                X = extract_logmel(file_path)
            elif FEATURE_NAME == "vgg512":
                X = extract_vgg(file_path, VGG_FEATURE_CSV)
            else:
                raise ValueError("Tên đặc trưng không hợp lệ.")

            result = predict_and_plot(X, FEATURE_NAME, MODEL_TYPE, os.path.basename(file_path))
            results.append(result)
        except Exception as e:
            print(f"❌ Lỗi với {file_path}: {e}")

    # Ghi file kết quả nếu có nhiều hơn 1
    if len(results) > 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df = pd.DataFrame(results)
        csv_path = os.path.join(OUTPUT_DIR, f"summary_realtime_{FEATURE_NAME}_{MODEL_TYPE}_{timestamp}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"\n🔽 Đã lưu kết quả tổng hợp tại: {csv_path}")

if __name__ == "__main__":
    main()

#Cách Chạy trên cmd
# Ghi âm và dự đoán
    #python predict_realtime_batch.py --mic
# Dự đoán từ thư mục
    #python predict_realtime_batch.py --folder D:/DeepVoice/Audio/real
