# ==========================================
# train_model_LGBM_v4.py
# 功能：自動偵測 GPU、轉換日期欄位、訓練 LightGBM 模型
# ==========================================

import pandas as pd
import lightgbm as lgb
import joblib
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix
)

# ======================
# 路徑設定
# ======================
train_file = "train_data_v4/train_data.csv"
output_dir = "train_data_v4"
os.makedirs(output_dir, exist_ok=True)

model_file = os.path.join(output_dir, "lightgbm_model.pkl")
feature_importance_file = os.path.join(output_dir, "feature_importance.csv")
threshold_file = os.path.join(output_dir, "best_threshold.json")

# ======================
# 1️⃣ 讀取資料
# ======================
print(f"📂 讀取訓練資料：{train_file}")
df = pd.read_csv(train_file)

# ======================
# 2️⃣ 修正日期欄位
# ======================
for col in ["first_txn", "last_txn"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df[col] = (df[col] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
        df[col] = df[col].fillna(0).astype("int64")
        print(f"🕒 已將欄位 {col} 轉換為 UNIX 秒數格式")

# ======================
# 3️⃣ 準備資料
# ======================
y = df["alert_flag"]
X = df.drop(columns=["acct_id", "alert_flag"], errors="ignore")

# 檢查非數值欄位
non_numeric = [c for c in X.columns if X[c].dtype not in ["int64", "float64", "bool"]]
if non_numeric:
    print(f"⚠️ 移除非數值欄位：{non_numeric}")
    X = X.drop(columns=non_numeric)

print(f"✅ 最終特徵數量：{X.shape[1]}")

# ======================
# 4️⃣ 分割訓練 / 驗證集
# ======================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ======================
# 5️⃣ 嘗試使用 GPU
# ======================
device_type = "cpu"
try:
    test_params = {"device": "gpu"}
    dtest = lgb.Dataset([[0, 1], [1, 0]], label=[0, 1])
    lgb.train(test_params, dtest, num_boost_round=1)
    device_type = "gpu"
    print("⚡ GPU 可用，將使用 GPU 加速訓練")
except Exception:
    print("💡 未偵測到 GPU，改用 CPU 模式")

# ======================
# 6️⃣ LightGBM 參數設定
# ======================
params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    "device": device_type
}

# ======================
# 7️⃣ 模型訓練
# ======================
print(f"🚀 開始訓練 LightGBM 模型 ({device_type.upper()}) ...")
start = time.time()

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

callbacks = [
    lgb.early_stopping(stopping_rounds=100),
    lgb.log_evaluation(period=200)
]

model = lgb.train(
    params=params,
    train_set=train_data,
    num_boost_round=2000,
    valid_sets=[train_data, val_data],
    valid_names=["train", "valid"],
    callbacks=callbacks
)

end = time.time()
print(f"⏱️ 訓練完成，耗時 {end - start:.2f} 秒")

# ======================
# 8️⃣ 模型評估
# ======================
print("📊 模型評估中 ...")
y_pred_prob = model.predict(X_val, num_iteration=model.best_iteration)

thresholds = [i / 100 for i in range(30, 71, 5)]
best_f1, best_th = 0, 0.5
for th in thresholds:
    y_pred = (y_pred_prob >= th).astype(int)
    f1 = f1_score(y_val, y_pred)
    if f1 > best_f1:
        best_f1, best_th = f1, th

y_pred = (y_pred_prob >= best_th).astype(int)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
acc = accuracy_score(y_val, y_pred)
cm = confusion_matrix(y_val, y_pred)

print("\n✅ 評估結果：")
print(f"Best Threshold : {best_th:.3f}")
print(f"F1 Score       : {best_f1:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"Accuracy       : {acc:.4f}")
print("Confusion Matrix:")
print(cm)

# ======================
# 9️⃣ 儲存模型與特徵重要度
# ======================
joblib.dump(model, model_file)
print(f"\n💾 模型已儲存至：{model_file}")

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importance()
}).sort_values(by="importance", ascending=False)
feature_importance.to_csv(feature_importance_file, index=False)
print(f"📊 特徵重要度已輸出至：{feature_importance_file}")

# 儲存最佳門檻
import json
with open(threshold_file, "w", encoding="utf-8") as f:
    json.dump({"best_threshold": best_th}, f, indent=2)
print(f"💾 最佳閾值已儲存：{threshold_file}")


