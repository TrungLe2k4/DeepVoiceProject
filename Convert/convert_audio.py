import os
import librosa
import soundfile as sf
from pydub import AudioSegment, effects
import numpy as np
from tqdm import tqdm

# ========== Cấu hình ===========
input_root = "D:/DeepVoice/Audio"
output_root = "D:/DeepVoice/Audio_Cleaned"
target_sr = 16000
duration = 5  # giây
target_length = target_sr * duration
labels = ["real", "fake"]

# ========== Hàm xử lý âm thanh ==========
def process_audio(input_path, output_path):
    try:
        # Chuyển âm thanh thành mono
        audio = AudioSegment.from_file(input_path).set_channels(1)
        
        # Nếu âm thanh ngắn hơn 5 giây, bỏ qua
        if audio.duration_seconds < 5:
            return None, f"Quá ngắn: {audio.duration_seconds:.2f}s"
        
        # Chuẩn hóa âm lượng
        normalized = effects.normalize(audio)
        samples = np.array(normalized.get_array_of_samples()).astype(np.float32) / 32768.0
        if np.abs(samples).mean() < 0.01:  # Kiểm tra âm lượng
            return None, "Âm lượng quá nhỏ"

        # Lưu lại tạm thời sau khi chuẩn hóa
        temp_wav = "temp.wav"
        normalized.export(temp_wav, format="wav")

        # Cắt tĩnh trong âm thanh
        y, _ = librosa.load(temp_wav, sr=target_sr)
        y_trimmed, _ = librosa.effects.trim(y, top_db=30)  # Loại bỏ tĩnh

        # Cắt hoặc giữ lại đủ 5 giây
        y_final = librosa.util.fix_length(y_trimmed, size=target_length)
        
        # Lưu âm thanh đã xử lý
        sf.write(output_path, y_final, target_sr)
        return True, ""
    except Exception as e:
        return None, str(e)

# ========== Xử lý dữ liệu ==========
os.makedirs(output_root, exist_ok=True)
error_log = []

for label in labels:
    input_folder = os.path.join(input_root, label)
    output_folder = os.path.join(output_root, label)
    os.makedirs(output_folder, exist_ok=True)

    files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.wav', '.mp3', '.flac'))]
    print(f"📁 {label}: {len(files)} file đầu vào")

    for file_name in tqdm(files, desc=f"Xử lý {label}"):
        input_path = os.path.join(input_folder, file_name)
        output_name = file_name.rsplit('.', 1)[0] + ".wav"
        output_path = os.path.join(output_folder, output_name)

        ok, msg = process_audio(input_path, output_path)
        if not ok:
            error_log.append((file_name, msg))

    print(f"📤 Sau xử lý {label}: {len(os.listdir(output_folder))} file")

# ========== Ghi log lỗi ==========
if error_log:
    log_path = os.path.join(output_root, "convert_audio_error_log.txt")
    with open(log_path, "w", encoding="utf-8") as logf:
        for name, msg in error_log:
            logf.write(f"{name}: {msg}\\n")
    print(f"⚠️ Ghi log lỗi tại: {log_path}")

print("✅ Đã xử lý toàn bộ audio.")
