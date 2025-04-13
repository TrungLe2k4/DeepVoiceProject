import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Thư mục chứa file âm thanh đầu vào (có 2 thư mục con real, fake)
input_root = "D:/DeepVoice/Audio_Cleaned"
# Thư mục lưu spectrograms
output_root = "D:/DeepVoice/Spectrograms"
os.makedirs(output_root, exist_ok=True)

labels = ["real", "fake"]

for label in labels:
    input_folder = os.path.join(input_root, label)
    output_folder = os.path.join(output_root, label)
    os.makedirs(output_folder, exist_ok=True)

    audio_files = [f for f in os.listdir(input_folder) if f.endswith(".wav")]
    print(f"\nSố lượng file trong '{label}': {len(audio_files)}")

    for file_name in tqdm(audio_files, desc=f"Đang tạo spectrograms cho {label}"):
        file_path = os.path.join(input_folder, file_name)

        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=16000)

            # Bỏ qua file quá ngắn
            if len(audio) < 1024:
                print(f"Bỏ qua {file_name} vì quá ngắn ({len(audio)} mẫu)")
                continue

            # Tạo Mel Spectrogram
            S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=1024, hop_length=512)
            S_dB = librosa.power_to_db(S, ref=np.max)

            # Vẽ và lưu ảnh spectrogram
            plt.figure(figsize=(5, 5))
            librosa.display.specshow(S_dB, sr=sr, hop_length=512, cmap="viridis")
            plt.axis("off")

            output_path = os.path.join(output_folder, file_name.replace(".wav", ".png"))
            plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
            plt.close()

        except Exception as e:
            print(f"Lỗi xử lý {file_name} ({label}): {e}")

print(f"\nĐã lưu tất cả spectrograms vào thư mục: {output_root}")
