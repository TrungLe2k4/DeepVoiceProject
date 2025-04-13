import os
import re

# Thư mục gốc chứa các file âm thanh gốc
audio_root = "D:/DeepVoice/Audio"

# Các nhãn (thư mục con)
labels = ['real', 'fake']

# Regex nhận diện file đã được đặt tên chuẩn
pattern = re.compile(r'^(real|fake)_(\d{4})\.(wav|mp3|flac)$')

# Duyệt qua từng nhãn
for label in labels:
    folder = os.path.join(audio_root, label)
    if not os.path.exists(folder):
        print(f"Bỏ qua vì không tìm thấy: {folder}")
        continue

    files = [f for f in os.listdir(folder) if f.endswith(('.wav', '.mp3', '.flac'))]
    files.sort()

    # Lấy danh sách số thứ tự đã tồn tại
    existing_indices = []
    for f in files:
        match = pattern.match(f)
        if match and match.group(1) == label:
            existing_indices.append(int(match.group(2)))

    next_index = max(existing_indices) + 1 if existing_indices else 1

    renamed_count = 0
    for f in files:
        if pattern.match(f):  # Nếu đã đúng định dạng thì bỏ qua
            continue

        ext = os.path.splitext(f)[1]
        new_name = f"{label}_{next_index:04d}{ext}"
        old_path = os.path.join(folder, f)
        new_path = os.path.join(folder, new_name)

        os.rename(old_path, new_path)
        print(f"Đã đổi tên: {f} ➜ {new_name}")
        next_index += 1
        renamed_count += 1

    print(f" Đổi tên {renamed_count} file mới trong thư mục '{label}'\n")
