import os
import librosa
import soundfile as sf
from tqdm import tqdm
from pydub import AudioSegment, effects

# Đường dẫn
input_root = "D:/DeepVoice/Audio"
output_clean_root = "D:/DeepVoice/Audio_Cleaned"
log_file_path = "D:/DeepVoice/error_log.txt"

# Tham số
target_sr = 16000
target_duration = 5.0  # giây
target_length = int(target_sr * target_duration)

# Xóa file log cũ nếu có
if os.path.exists(log_file_path):
    os.remove(log_file_path)

error_logs = []
labels = ["real", "fake"]

def noise_reduction(audio, sr):
    """
    Giảm noise bằng thư viện noisereduce nếu có.
    """
    try:
        import noisereduce as nr
        return nr.reduce_noise(y=audio, sr=sr)
    except ImportError:
        print("Chưa cài thư viện noisereduce, bỏ qua bước tách noise.")
        return audio

for label in labels:
    input_folder = os.path.join(input_root, label)
    output_clean_folder = os.path.join(output_clean_root, label)
    os.makedirs(output_clean_folder, exist_ok=True)

    print(f"Đang xử lý thư mục: {label}")

    for file_name in tqdm(os.listdir(input_folder), desc=f"Xử lý {label}"):
        if file_name.endswith(('.mp3', '.wav', '.flac')):
            file_path = os.path.join(input_folder, file_name)
            try:
                # Load và chuẩn hóa
                audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
                audio = noise_reduction(audio, sr)

                # Cắt hoặc đệm để đủ 5 giây
                if len(audio) < target_length:
                    audio = librosa.util.fix_length(audio, size=target_length)
                else:
                    audio = audio[:target_length]

                # Ghi tạm vào bộ nhớ để normalize bằng pydub
                temp_path = os.path.join(output_clean_folder, "temp.wav")
                sf.write(temp_path, audio, sr)

                # Normalize
                segment = AudioSegment.from_wav(temp_path)
                normalized = effects.normalize(segment)

                # Ghi kết quả cuối
                output_path = os.path.join(output_clean_folder, file_name.rsplit('.', 1)[0] + ".wav")
                normalized.export(output_path, format="wav")

                # Xoá file tạm
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            except Exception as e:
                reason = f"[PROCESS] {file_name} ({label}): {e}"
                print(reason)
                error_logs.append(reason)

# Ghi log lỗi
if error_logs:
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(error_logs))
    print(f"\nCó {len(error_logs)} lỗi. Xem chi tiết tại: {log_file_path}")
else:
    print("\nKhông có lỗi nào xảy ra!")

print("\nHoàn tất xử lý toàn bộ âm thanh.")
