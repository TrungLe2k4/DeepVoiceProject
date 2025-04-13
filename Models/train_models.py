import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import joblib

# Thư mục chứa file đặc trưng CSV
FEATURE_DIR = "D:/DeepVoice/Dataset"

# Thư mục gốc để lưu các mô hình
BASE_MODEL_DIR = "D:/DeepVoice/Models"

# Map tên feature sang thư mục tương ứng
FEATURE_FOLDER_MAP = {
    "fft_mfcc": "FFT_MFCC",
    "vgg512": "VGG16",
    "logmel": "LogMel"
}

def train_and_save_models(feature_name, df):
    print(f"\n=== Huấn luyện mô hình cho: {feature_name} ===")

    # Tạo thư mục lưu kết quả cho đặc trưng này
    model_dir = os.path.join(BASE_MODEL_DIR, FEATURE_FOLDER_MAP[feature_name])
    os.makedirs(model_dir, exist_ok=True)

    # ==== Xác định cột nhãn ====
    label_column = None
    for col in ["label", "Lable", "target", "class"]:
        if col in df.columns:
            label_column = col
            break
    if label_column is None:
        raise ValueError("Không tìm thấy cột nhãn trong dữ liệu.")

    y = df[label_column]

    # Nếu label là chuỗi 'real'/'fake' thì chuyển thành số
    if y.dtype == object or y.dtype == "string":
        y = y.map({"real": 0, "fake": 1})

    # ==== Tạo X và y ====
    drop_cols = [col for col in ["file", "file_name", label_column] if col in df.columns]
    X = df.drop(columns=drop_cols)

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Lưu scaler
    scaler_path = os.path.join(model_dir, f"{feature_name}_scaler.pkl")
    joblib.dump(scaler, scaler_path)

    # Tách train/test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # ==== SVM ====
    svm_model = SVC(kernel="linear", probability=True)
    svm_model.fit(X_train, y_train)
    svm_path = os.path.join(model_dir, f"{feature_name}_svm.pkl")
    joblib.dump(svm_model, svm_path)

    # Cross-validation SVM
    svm_scores = cross_val_score(svm_model, X_train, y_train, cv=5)
    print(f"[SVM] Độ chính xác trung bình (CV): {svm_scores.mean():.4f}")

    # Confusion matrix SVM
    y_pred_svm = svm_model.predict(X_test)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    disp_svm = ConfusionMatrixDisplay(cm_svm, display_labels=["Real", "Fake"])
    disp_svm.plot()
    plt.title(f"{feature_name} - SVM - Accuracy: {acc_svm:.4f}")
    plt.savefig(os.path.join(model_dir, f"{feature_name}_svm_cm.png"))
    plt.close()

    # ==== Random Forest ====
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_path = os.path.join(model_dir, f"{feature_name}_rf.pkl")
    joblib.dump(rf_model, rf_path)

    # Cross-validation RF
    rf_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
    print(f"[RandomForest] Độ chính xác trung bình (CV): {rf_scores.mean():.4f}")

    # Confusion matrix RF
    y_pred_rf = rf_model.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    disp_rf = ConfusionMatrixDisplay(cm_rf, display_labels=["Real", "Fake"])
    disp_rf.plot()
    plt.title(f"{feature_name} - RF - Accuracy: {acc_rf:.4f}")
    plt.savefig(os.path.join(model_dir, f"{feature_name}_rf_cm.png"))
    plt.close()
    # ==== XGBoost ====
    xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_path = os.path.join(model_dir, f"{feature_name}_xgb.pkl")
    joblib.dump(xgb_model, xgb_path)

    # Cross-validation XGBoost
    xgb_scores = cross_val_score(xgb_model, X_train, y_train, cv=5)
    print(f"[XGBoost] Độ chính xác trung bình (CV): {xgb_scores.mean():.4f}")

    # Confusion matrix XGBoost
    y_pred_xgb = xgb_model.predict(X_test)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    disp_xgb = ConfusionMatrixDisplay(cm_xgb, display_labels=["Real", "Fake"])
    disp_xgb.plot()
    plt.title(f"{feature_name} - XGBoost - Accuracy: {acc_xgb:.4f}")
    plt.savefig(os.path.join(model_dir, f"{feature_name}_xgb_cm.png"))
    plt.close()


# ==== Danh sách các file đặc trưng ====
csv_files = {
    "fft_mfcc": os.path.join(FEATURE_DIR, "fft_mfcc.csv"),
    "vgg512": os.path.join(FEATURE_DIR, "vgg512_features.csv"),
    "logmel": os.path.join(FEATURE_DIR, "logmel_features.csv")
}

# ==== Huấn luyện cho từng file ====
for name, path in csv_files.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        train_and_save_models(name, df)
    else:
        print(f"Không tìm thấy file đặc trưng: {path}")
