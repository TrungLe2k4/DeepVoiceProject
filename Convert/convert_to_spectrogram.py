import os
import time
import librosa
import numpy as np
import csv
from PIL import Image
from tqdm import tqdm

# ========== Cấu hình ==========
input_root = "D:/DeepVoice/Audio_Cleaned"
output_root = "D:/DeepVoice/Spectrograms"
sample_rate = 16000
duration = 5
target_length = sample_rate * duration
n_fft = 1024
hop_length = 160
win_length = 400
n_mels = 128
labels = ["real", "fake"]

# ========== Hàm tạo ảnh spectrogram ==========
def convert_to_spectrogram(file_path, output_path):
    try:
        duration_y = librosa.get_duration(path=file_path)
        if duration_y < 1.0:
            return False, f"File quá ngắn (duration={duration_y:.2f}s)"

        y, sr = librosa.load(file_path, sr=sample_rate)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        else:
            y = y[:target_length]

        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
            win_length=win_length, n_mels=n_mels
        )
        S_dB = librosa.power_to_db(S, ref=np.max)
        S_norm = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min()) * 255
        S_norm = S_norm.astype(np.uint8)

        # Convert ảnh sang RGB cho mô hình VGG
        img = Image.fromarray(S_norm).convert("RGB")
        img = img.resize((224, 224))
        img.save(output_path)
        return True, ""
    except Exception as e:
        return False, str(e)

# ========== Xử lý ==========
start_time = time.time()
os.makedirs(output_root, exist_ok=True)
error_log = []

for label in labels:
    input_folder = os.path.join(input_root, label)
    output_folder = os.path.join(output_root, label)
    os.makedirs(output_folder, exist_ok=True)

    files = [f for f in os.listdir(input_folder) if f.lower().endswith(".wav")]
    print(f"📁 {label}: {len(files)} file đầu vào")

    for file_name in tqdm(files, desc=f"Tạo ảnh spectrogram cho {label}"):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name.replace(".wav", ".png"))
        ok, msg = convert_to_spectrogram(input_path, output_path)
        if not ok:
            error_log.append((label, file_name, msg))

# ========== Ghi log lỗi nếu có ==========
if error_log:
    log_path = os.path.join(output_root, "spectrogram_error_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as logf:
        writer = csv.writer(logf)
        writer.writerow(["Label", "File", "Lý do lỗi"])
        writer.writerows(error_log)
    print(f"⚠️ Ghi log lỗi tại: {log_path}")

elapsed_time = time.time() - start_time
print(f"✅ Đã tạo xong ảnh spectrogram vào {output_root}")