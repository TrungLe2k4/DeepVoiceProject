import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Chuyển matplotlib sang backend 'Agg' để tránh lỗi GUI
matplotlib.use('Agg')

dataset_folder = "D:/DeepVoice/Dataset"
output_folder = "D:/DeepVoice/Models"
os.makedirs(output_folder, exist_ok=True)

features = [
    ("fft_mfcc_train.csv", "FFT_MFCC"),
    ("logmel_train.csv", "LogMel"),
    ("vgg16_train.csv", "VGG16")
]

def save_model_and_results(model, scaler, le, cm, feature_name, model_name):
    model_folder = os.path.join(output_folder, feature_name)
    os.makedirs(model_folder, exist_ok=True)

    cm_filename = os.path.join(model_folder, f"{model_name}_confusion_matrix.png")
    model_filename = os.path.join(model_folder, f"{model_name}_model.pkl")
    scaler_filename = os.path.join(model_folder, f"{model_name}_scaler.pkl")
    labelencoder_filename = os.path.join(model_folder, f"{model_name}_labelencoder.pkl")

    # Save confusion matrix
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"Confusion Matrix - {model_name} ({feature_name})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(cm_filename)
    plt.close()

    # Save model, scaler, and label encoder
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)
    joblib.dump(le, labelencoder_filename)
    print(f"Đã lưu model, scaler, label encoder, confusion matrix for {model_name} ({feature_name}).")

def grid_search_svm(X, y):
    param_grid = {
        'C': [0.1, 0.5, 1, 5, 10, 50],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
    }
    svc = SVC(probability=True)
    grid = GridSearchCV(svc, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=2)
    grid.fit(X, y)
    print("SVM best params:", grid.best_params_)
    return grid.best_estimator_

def grid_search_rf(X, y):
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
    rf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=2)
    grid.fit(X, y)
    print("RF best params:", grid.best_params_)
    return grid.best_estimator_

results = []

for feature_file, feature_name in features:
    print(f"\n=== Training with feature {feature_name} ===")
    data = pd.read_csv(os.path.join(dataset_folder, feature_file))
    X = data.drop(["file", "label"], axis=1)
    y = data["label"]

    le = LabelEncoder()
    y = le.fit_transform(y)
    print(f"Classes: {le.classes_}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = ["SVM", "RandomForest", "XGBoost"]

    for model_name in models:
        print(f"\nTraining model {model_name} with feature {feature_name}...")

        if model_name == "SVM":
            best_model = grid_search_svm(X_scaled, y)
        elif model_name == "RandomForest":
            best_model = grid_search_rf(X_scaled, y)
        else:
            best_model = XGBClassifier(eval_metric='logloss', max_depth=3, reg_lambda=2.0, subsample=0.8, random_state=42)
            best_model.fit(X_scaled, y)

        model = best_model

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_accuracies, cv_precisions, cv_recalls, cv_f1_scores, cv_roc_aucs = [], [], [], [], []

        for train_idx, test_idx in kf.split(X_scaled, y):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            cv_accuracies.append(accuracy_score(y_test, y_pred))
            cv_precisions.append(precision_score(y_test, y_pred))
            cv_recalls.append(recall_score(y_test, y_pred))
            cv_f1_scores.append(f1_score(y_test, y_pred))
            cv_roc_aucs.append(roc_auc_score(y_test, y_prob))

        avg_acc = np.mean(cv_accuracies)
        avg_prec = np.mean(cv_precisions)
        avg_rec = np.mean(cv_recalls)
        avg_f1 = np.mean(cv_f1_scores)
        avg_auc = np.mean(cv_roc_aucs)

        print(f"Model: {model_name} ({feature_name})")
        print(f"  Accuracy: {avg_acc*100:.2f}%")
        print(f"  Precision: {avg_prec*100:.2f}%")
        print(f"  Recall: {avg_rec*100:.2f}%")
        print(f"  F1-Score: {avg_f1*100:.2f}%")
        print(f"  ROC-AUC: {avg_auc*100:.2f}%")

        results.append([feature_name, model_name, avg_acc*100, avg_prec*100, avg_rec*100, avg_f1*100, avg_auc*100])

        cm = confusion_matrix(y_test, y_pred)
        save_model_and_results(model, scaler, le, cm, feature_name, model_name)

summary_df = pd.DataFrame(results, columns=["Feature_Set", "Model", "Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"])
summary_csv_path = os.path.join(output_folder, "summary_metrics.csv")
summary_df.to_csv(summary_csv_path, index=False)

print("\nSaved all metrics summary at:", summary_csv_path)
print("\nTraining completed for all models and features.")
