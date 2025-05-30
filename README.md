#  Dự án DeepVoice – Phát hiện tấn công giọng nói Deepfake  
#  DeepVoice Project – Detecting Deepfake Voice Attacks

---

##  Mô tả dự án  
##  Project Description

**VI:**  
Đây là đồ án môn học nhằm phát hiện các tệp âm thanh bị tấn công bằng công nghệ Deepfake. Dự án sử dụng các đặc trưng âm thanh như FFT, MFCC, Log-Mel Spectrogram, và ảnh Mel Spectrogram kết hợp mô hình học máy (SVM, Random Forest, XGBoost) để phân biệt giọng thật (real) và giả (fake).

**EN:**  
This is a course project focused on detecting audio files manipulated by Deepfake technology. The system uses audio features such as FFT, MFCC, Log-Mel Spectrogram, and Mel Spectrogram images along with machine learning models (SVM, Random Forest, XGBoost) to classify real and fake voices.

---

##  Các đặc trưng & mô hình sử dụng  
##  Features & Models Used

- **FFT + MelSpectrogram + MFCC** (17 chiều / 17-dimensional)
- **Log-Mel Spectrogram** (128 chiều / 128-dimensional)
- **VGG16 Features** từ ảnh spectrogram (512 chiều / 512-dimensional)

 Mô hình học máy sử dụng:
- SVM
- Random Forest
- XGBoost

---

##  Cấu trúc thư mục  
##  Project Structure
DeepVoice/ 
├── Audio/ # Âm thanh thô (raw audio) 
    ├──real  
    ├──fake
├── Audio_Cleaned/ # Âm thanh đã xử lý (.wav) 
    ├──real  
    ├──fake
├── Spectrograms/ # Ảnh Mel Spectrogram 
    ├──real  
    ├──fake
├── Dataset/ # File CSV chứa đặc trưng 
    ├──fft_mfcc.py
    ├──logmel_features.py
    ├──vgg512_features.py
├── Models/ # Mô hình đã huấn luyện
    ├──FFT_MFCC
    ├──LogMel
    ├──VGG16
    ├── train_models.py # Huấn luyện mô hình 
├── Predictions/ # Kết quả dự đoán 
    ├──FFT_MFCC
    ├──LogMel
    ├──VGG16
    └── predict_*.py # Script dự đoán
├── FeatureExtract/ # Script trích xuất đặc trưng 
    ├──extract_mfcc.py
    ├──extract_logmel.py
    ├──extract_vgg.py
├── convert_audio.py # Xử lý âm thanh đầu vào 
├── convert_to_spectrogram.py# Tạo ảnh spectrogram

# Liên hệ
# Contact
 Nếu bạn có câu hỏi hoặc muốn đóng góp, vui lòng liên hệ qua GitHub hoặc email tle15072004@gmail.com.
 If you have any questions or would like to contribute, feel free to reach out via GitHub or email tle15072004@gmail.com.

# 🇻🇳 Đây là đồ án phục vụ học phần "Phát hiện tấn công Deepfake Audio"
# 🇬🇧 This project is part of the course: "Deepfake Audio Attack Detection"

