import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder

# ==================== Đường dẫn ====================
model_folder = "D:/DeepVoice/Models"
dataset_folder = "D:/DeepVoice/Dataset"

# ==================== Danh sách đặc trưng và mô hình ====================
test_files = [
    ("fft_mfcc_test.csv", "FFT_MFCC"),
    ("logmel_test.csv", "LogMel"),
    ("vgg16_test.csv", "VGG16")
]

models = ["SVM", "RandomForest", "XGBoost"]

all_results = []

for test_file, feature_name in test_files:
    print(f"\n========== ĐANG TEST {feature_name} ==========")
    data_path = os.path.join(dataset_folder, test_file)
    if not os.path.exists(data_path):
        print(f"❌ Không tìm thấy file test: {data_path}. Bỏ qua.")
        continue

    data = pd.read_csv(data_path)
    X_test = data.drop(["file", "label"], axis=1)
    y_test = LabelEncoder().fit_transform(data["label"])

    feature_results = []

    for model_name in models:
        print(f"\n=== Đang kiểm tra {model_name} với {feature_name} ===")
        scaler_path = os.path.join(model_folder, feature_name, f"{model_name}_scaler.pkl")
        model_path = os.path.join(model_folder, feature_name, f"{model_name}_model.pkl")

        if not os.path.exists(scaler_path) or not os.path.exists(model_path):
            print(f"❌ Không tìm thấy mô hình hoặc scaler cho {model_name} với {feature_name}. Bỏ qua.")
            continue

        scaler = joblib.load(scaler_path)
        model = joblib.load(model_path)
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        feature_results.append([feature_name, model_name, acc*100, prec*100, rec*100, f1*100, auc*100])

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {model_name} ({feature_name})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        cm_filename = os.path.join(model_folder, feature_name, f"{model_name}_test_confusion_matrix.png")
        plt.savefig(cm_filename)
        plt.close()

    # Ghi kết quả từng feature
    summary_df = pd.DataFrame(feature_results, columns=["Feature_Set", "Model", "Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"])
    summary_path = os.path.join(model_folder, f"{feature_name}_test_results.csv")
    summary_df.to_csv(summary_path, index=False)
    all_results.extend(feature_results)

    # Vẽ biểu đồ cho từng đặc trưng
    plt.figure(figsize=(10, 6))
    for metric in ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]:
        plt.plot(summary_df["Model"], summary_df[metric], marker="o", label=metric)

    plt.title(f"So sánh hiệu suất các mô hình ({feature_name})")
    plt.ylabel("Giá trị (%)")
    plt.xlabel("Tên mô hình")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    chart_path = os.path.join(model_folder, feature_name, "overall_comparison.png")
    plt.savefig(chart_path)
    plt.close()

# Tổng hợp tất cả kết quả
all_df = pd.DataFrame(all_results, columns=["Feature_Set", "Model", "Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"])
final_summary_path = os.path.join(model_folder, "all_features_test_results.csv")
all_df.to_csv(final_summary_path, index=False)
print(f"\n✅ Đã lưu bảng tổng hợp toàn bộ mô hình vào: {final_summary_path}")
