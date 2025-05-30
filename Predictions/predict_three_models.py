import os
import torch
import joblib
import torchaudio
import numpy as np
import librosa
import librosa.display
import pandas as pd
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg')

sample_rate = 16000
duration = 5
target_length = sample_rate * duration
model_root = "D:/DeepVoice/Models"
temp_img = "temp_spectrogram.png"
test_folder = "D:/DeepVoice/uploads"
output_csv = "D:/DeepVoice/Predictions/batch_predictions.csv"

def load_and_fix_length(file_path, target_length=16000*5):
    audio, sr = librosa.load(file_path, sr=sample_rate)
    if len(audio) > target_length:
        audio = audio[:target_length]
    else:
        padding = target_length - len(audio)
        audio = np.pad(audio, (0, padding), 'constant')
    return audio, sr

def extract_mfcc_features(file_path):
    audio, sr = load_and_fix_length(file_path, target_length)
    fft = np.abs(np.fft.rfft(audio))
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))

    # Bổ sung đặc trưng phụ
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_mean = np.mean(chroma, axis=1).tolist()  # 12 chiều

    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    spectral_contrast_mean = np.mean(spectral_contrast, axis=1).tolist()  # 7 chiều

    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
    tonnetz_mean = np.mean(tonnetz, axis=1).tolist()  # 6 chiều

    return np.array([np.mean(fft), np.std(fft), np.mean(mel_spec), np.std(mel_spec),
                    spec_centroid, spec_bw, zcr] +
                   np.mean(mfcc, axis=1).tolist() +
                   np.mean(delta, axis=1).tolist() +
                   np.mean(delta2, axis=1).tolist() +
                   chroma_mean +
                   spectral_contrast_mean +
                   tonnetz_mean)

def save_spectrogram_image(file_path, output_path):
    y, sr = librosa.load(file_path, sr=16000)
    duration = 5
    target_length = sr * duration
    if len(y) > target_length:
        y = y[:target_length]
    else:
        y = np.pad(y, (0, target_length - len(y)), 'constant')

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(6,6))
    plt.axis('off')
    librosa.display.specshow(S_dB, sr=sr, cmap='magma')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def extract_logmel_features(file_path):
    waveform, sr = torchaudio.load(file_path)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    waveform = waveform[:, :target_length]
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=1024, hop_length=160, win_length=400, n_mels=128
    )(waveform)
    logmel = torchaudio.transforms.AmplitudeToDB(top_db=80)(mel).squeeze().numpy()
    return np.concatenate([np.mean(logmel, axis=1), np.std(logmel, axis=1)])

def extract_vgg_features(file_path):
    save_spectrogram_image(file_path, temp_img)
    img = Image.open(temp_img).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image_tensor = transform(img).unsqueeze(0)
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    vgg.classifier = torch.nn.Identity()
    vgg.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
    vgg.eval()
    with torch.no_grad():
        features = vgg(image_tensor).squeeze().numpy()
    os.remove(temp_img)
    return features

def predict_from_feature(feature, model_dir, model_name):
    scaler_path = os.path.join(model_dir, f"{model_name}_scaler.pkl")
    model_path = os.path.join(model_dir, f"{model_name}_model.pkl")
    labelencoder_path = os.path.join(model_dir, f"{model_name}_labelencoder.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(labelencoder_path):
        return None, None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    le = joblib.load(labelencoder_path)

    X_scaled = scaler.transform([feature])
    label = model.predict(X_scaled)[0]
    proba_all = model.predict_proba(X_scaled)[0]  # xác suất từng lớp

    label_str = le.inverse_transform([label])[0]
    proba = proba_all[label]  # lấy xác suất đúng lớp dự đoán

    return label, label_str, proba


def show_mel_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=sample_rate)
    if len(y) > target_length:
        y = y[:target_length]
    else:
        y = np.pad(y, (0, target_length - len(y)), 'constant')

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(6, 6))
    librosa.display.specshow(S_dB, sr=sr, cmap='magma')
    plt.title("Mel Spectrogram")
    plt.colorbar(format="%+2.0f dB")
    plt.show()

def predict_file(file_path):
    # Show ảnh Mel Spectrogram ra màn hình mỗi lần dự đoán file
    show_mel_spectrogram(file_path)

    results = {}
    for model_name in ["SVM", "RandomForest", "XGBoost"]:
        best_feat, best_proba, best_label, best_label_str = None, -1, None, None
        for feat_type, extractor in zip(["FFT_MFCC", "LogMel", "VGG16"],
                                        [extract_mfcc_features, extract_logmel_features, extract_vgg_features]):
            feature = extractor(file_path)
            label, label_str, proba = predict_from_feature(feature, os.path.join(model_root, feat_type), model_name)
            if label is not None and proba > best_proba:
                best_feat, best_proba, best_label, best_label_str = feat_type, proba, label, label_str
        results[model_name] = (best_feat, best_label, best_label_str, best_proba)
    return results

def main():
    print(f"\n📁 Dự đoán tất cả file trong thư mục: {test_folder}\n")
    files = []
    for root, _, filenames in os.walk(test_folder):
        for f in filenames:
            if f.lower().endswith(".wav"):
                files.append(os.path.join(root, f))

    output = []
    for full_path in files:
        fname = os.path.basename(full_path)
        print(f"\n===== Dự đoán file: {fname} =====")
        results = predict_file(full_path)
        for model_name, (best_feat, best_label, best_label_str, best_proba) in results.items():
            print(f"  {model_name}: {best_label_str.upper()} via {best_feat} (Probability: {best_proba:.2f})")
            output.append({
                "File": fname,
                "Model": model_name,
                "Best_Feature": best_feat,
                "Label": best_label_str.upper(),
                "Probability": round(best_proba, 4)
            })

    if output:
        df = pd.DataFrame(output)
        df.to_csv(output_csv, index=False)
        print(f"\n📄 Kết quả đã được lưu tại: {output_csv}")

if __name__ == "__main__":
    if os.path.exists(test_folder):
        main()
    else:
        print("❌ Không tìm thấy thư mục Audio_Cleaned!")
