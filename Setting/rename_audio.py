import os

# Thư mục gốc
audio_root = "D:/DeepVoice/Audio"
labels = ['real', 'fake']

for label in labels:
    folder = os.path.join(audio_root, label)
    if not os.path.exists(folder):
        print(f"Bỏ qua vì không tìm thấy: {folder}")
        continue

    files = [f for f in os.listdir(folder) if f.endswith(('.wav', '.mp3', '.flac'))]
    files.sort()

    # Đổi tên tạm
    for i, f in enumerate(files):
        old_path = os.path.join(folder, f)
        ext = os.path.splitext(f)[1]
        tmp_name = f"tmp_{i:04d}{ext}"
        tmp_path = os.path.join(folder, tmp_name)
        os.rename(old_path, tmp_path)

    # Lấy lại danh sách file tạm
    tmp_files = [f for f in os.listdir(folder) if f.startswith('tmp_')]
    tmp_files.sort()

    # Rename lại đúng nhãn real hoặc fake
    index = 1
    for f in tmp_files:
        ext = os.path.splitext(f)[1]
        new_name = f"{label}_{index:04d}{ext}"   # Chỗ này lấy đúng label hiện tại
        old_path = os.path.join(folder, f)
        new_path = os.path.join(folder, new_name)
        os.rename(old_path, new_path)
        print(f"Đã đổi tên: {f} ➔ {new_name}")
        index += 1

    print(f" Hoàn tất đổi tên {index-1} file trong thư mục '{label}'\n")
