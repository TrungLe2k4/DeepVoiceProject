import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm

# ======= Cấu hình =========
input_root = "D:/DeepVoice/Audio_Cleaned"         # Thư mục gốc chứa các file âm thanh (gồm 'real' và 'fake')
output_csv = "D:/DeepVoice/Dataset/logmel_features.csv"  # File CSV lưu kết quả
sample_rate = 16000       # Sample rate mong muốn
duration = 5              # Độ dài audio cố định (giây)
n_mels = 128              # Số dải Mel
n_fft = 1024              # Số điểm FFT
hop_length = 160          # Khoảng cách giữa các frame (~10ms nếu sr=16000)
win_length = 400          # Độ dài cửa sổ (~25ms nếu sr=16000)

# ======= Khởi tạo các transform =========
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length,
    n_mels=n_mels
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

    target_length = sample_rate * duration
    if waveform.shape[1] < target_length:
        pad_amount = target_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
    else:
        waveform = waveform[:, :target_length]

    mel_spec = mel_transform(waveform)
    log_mel = db_transform(mel_spec)
    features = torch.mean(log_mel, dim=2).squeeze().numpy()
    return features

# ======= Xử lý các file trong 'real' và 'fake' =========
data = []

for label in ['real', 'fake']:
    folder_path = os.path.join(input_root, label)
    if not os.path.exists(folder_path):
        print(f"Không tìm thấy thư mục: {folder_path}")
        continue

    for file_name in tqdm(os.listdir(folder_path), desc=f"Processing {label}"):
        if file_name.lower().endswith(".wav"):
            file_path = os.path.join(folder_path, file_name)
            try:
                features = extract_logmel_features(file_path)
                data.append([file_name, label] + features.tolist())  # Dùng 'real' hoặc 'fake' trực tiếp
            except Exception as e:
                print(f"Lỗi xử lý {file_name}: {e}")

# ======= Lưu ra CSV =========
columns = ["file", "label"] + [f"logmel_{i+1}" for i in range(n_mels)]
df = pd.DataFrame(data, columns=columns)
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df.to_csv(output_csv, index=False)

print(f"Đã lưu đặc trưng Log-Mel vào: {output_csv}")
