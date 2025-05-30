import os
import hashlib
from glob import glob

# Đường dẫn thư mục chứa audio cần kiểm tra (real và fake)
audio_root = "D:/DeepVoice/Audio_Cleaned"
output_delete_log = "duplicate_wav_deleted_log.txt"

# Hàm tính hash nhanh cho nội dung file WAV
def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# Quét và phát hiện file bị trùng nội dung
hashes = {}
duplicates = []

for subfolder in ["real", "fake"]:
    folder_path = os.path.join(audio_root, subfolder)
    wav_files = glob(os.path.join(folder_path, "*.wav"))
    
    for wav in wav_files:
        h = file_hash(wav)
        if h in hashes:
            duplicates.append((wav, hashes[h]))  # (bản sao, bản gốc)
        else:
            hashes[h] = wav

# Ghi log và tạo danh sách file trùng để xoá
with open(output_delete_log, "w", encoding="utf-8") as f:
    for dup, original in duplicates:
        f.write(f"XÓA: {dup} (trùng với {original})\n")
        os.remove(dup)  # Xoá file trùng

# Kết quả
output_delete_log, len(duplicates)