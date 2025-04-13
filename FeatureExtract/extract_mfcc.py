import os
import librosa
import numpy as np
import pandas as pd
import librosa.display
from tqdm import tqdm

# Đường dẫn thư mục chứa file WAV đã tiền xử lý
input_folder = "D:/DeepVoice/Audio_Cleaned"
output_csv = "D:/DeepVoice/Dataset/fft_mfcc.csv"

# Tạo danh sách để lưu dữ liệu
data = []

# Duyệt qua từng file trong thư mục con (real, fake)
for label in ["real", "fake"]:
    subfolder = os.path.join(input_folder, label)
    
    for file_name in tqdm(os.listdir(subfolder), desc=f"Xử lý {label}"):
        if file_name.endswith(".wav"):
            file_path = os.path.join(subfolder, file_name)
            audio, sr = librosa.load(file_path, sr=16000)

            # FFT
            fft = np.abs(np.fft.rfft(audio))
            fft_mean = np.mean(fft)
            fft_std = np.std(fft)

            # Mel Spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
            mel_mean = np.mean(mel_spec)
            mel_std = np.std(mel_spec)

            # MFCC
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)

            # Thêm vào danh sách dữ liệu
            data.append([file_name, label, fft_mean, fft_std, mel_mean, mel_std, *mfcc_mean])

# Lưu vào CSV
columns = ["file", "label", "fft_mean", "fft_std", "mel_mean", "mel_std"] + [f"mfcc_{i+1}" for i in range(13)]
df = pd.DataFrame(data, columns=columns)
df.to_csv(output_csv, index=False)

print(f"\nĐã lưu đặc trưng vào: {output_csv}")
