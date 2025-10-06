# ===============================================================
#  predict_accounts_v4.py  (AI CUP 自動防呆 + 完整統計版)
# ===============================================================

import pandas as pd
import joblib
import os
import numpy as np
import json

# ==========================
# 1️⃣ 路徑設定
# ==========================
feature_file = "feature_data_v4/account_features.csv"
predict_file = "dataset/acct_predict.csv"
model_file = "train_data_v4/lightgbm_model.pkl"
best_threshold_file = "train_data_v4/best_threshold.json"
output_dir = "results_v4"
os.makedirs(output_dir, exist_ok=True)

output_full = os.path.join(output_dir, "predict_full_v4.csv")
submit_file = os.path.join(output_dir, "predict_for_submit_v4.csv")

# ==========================
# 2️⃣ 載入資料與模型
# ==========================
print(f"📂 載入特徵表：{feature_file}")
features = pd.read_csv(feature_file)

print(f"📂 載入預測帳戶：{predict_file}")
predict_accts = pd.read_csv(predict_file)

print(f"📦 載入 LightGBM 模型：{model_file}")
model = joblib.load(model_file)

# ==========================
# 3️⃣ 統一帳號欄位名稱
# ==========================
acct_col = None
for col in predict_accts.columns:
    if col.strip().lower() in ["acct", "acct_id"]:
        acct_col = col
        break
if acct_col is None:
    raise ValueError("❌ acct_predict.csv 中沒有找到帳號欄位 (acct 或 acct_id)")

predict_accts.rename(columns={acct_col: "acct_id"}, inplace=True)
features["acct_id"] = features["acct_id"].astype(str).str.strip()
predict_accts["acct_id"] = predict_accts["acct_id"].astype(str).str.strip()

# ==========================
# 4️⃣ 轉換時間欄位
# ==========================
for col in ["first_txn", "last_txn"]:
    if col in features.columns:
        features[col] = pd.to_datetime(features[col], errors="coerce")
        features[col] = (features[col] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
        print(f"🕒 已將欄位 {col} 轉換為 UNIX 秒數格式")

# ==========================
# 5️⃣ 對齊模型特徵欄位
# ==========================
train_features = model.feature_name()
missing_cols = [c for c in train_features if c not in features.columns]
extra_cols = [c for c in features.columns if c not in train_features]

if missing_cols:
    print(f"⚠️ 缺少欄位: {missing_cols}")
if extra_cols:
    print(f"⚠️ 多出欄位: {extra_cols}")

for c in missing_cols:
    features[c] = 0

predict_features = features[features["acct_id"].isin(predict_accts["acct_id"])].copy()

# 若有未匹配帳號 → 自動補上平均特徵
missing_accts = set(predict_accts["acct_id"]) - set(predict_features["acct_id"])
if len(missing_accts) > 0:
    print(f"⚠️ 有 {len(missing_accts)} 筆帳號沒有特徵，將自動補平均值特徵")
    mean_row = features[train_features].mean().to_dict()
    missing_df = pd.DataFrame([mean_row] * len(missing_accts))
    missing_df["acct_id"] = list(missing_accts)
    predict_features = pd.concat([predict_features, missing_df], ignore_index=True)

X_pred = predict_features.reindex(columns=train_features, fill_value=0)
print(f"🔹 使用特徵數量: {X_pred.shape[1]}，預測筆數: {len(X_pred)}")

# ==========================
# 6️⃣ 載入閾值 & GPU 模式偵測
# ==========================
if os.path.exists(best_threshold_file):
    best_th = json.load(open(best_threshold_file, "r")).get("best_threshold", 0.5)
    print(f"🧠 自動載入訓練最佳閾值: {best_th:.3f}")
else:
    best_th = 0.5
    print("⚠️ 未偵測到 best_threshold.json，使用預設 0.5")

try:
    if model.params.get("device", "") == "gpu":
        print("🚀 使用 GPU 進行預測中 ...")
    else:
        print("🚀 使用 CPU 進行預測中 ...")
except Exception:
    print("💡 使用 CPU 推論（無 GPU 模式資訊）")

# ==========================
# 7️⃣ 預測階段
# ==========================
y_pred_prob = model.predict(X_pred, num_iteration=getattr(model, "best_iteration", None))
predict_features["probability"] = y_pred_prob
predict_features["label"] = (y_pred_prob >= best_th).astype(int)

print(f"✅ 已完成模型預測，共 {len(predict_features)} 筆。")

# ==========================
# 8️⃣ 合併與輸出
# ==========================
submission = pd.merge(predict_accts, predict_features[["acct_id", "label", "probability"]],
                      on="acct_id", how="left")

# 🧩 修正重複欄位（label_x, label_y）
if "label_x" in submission.columns and "label_y" in submission.columns:
    print("⚙️ 偵測到 label_x / label_y，進行合併 ...")
    submission["label"] = submission["label_y"].combine_first(submission["label_x"])
    submission.drop(columns=["label_x", "label_y"], inplace=True)

# 🧩 若仍無 label 欄位 → 補上 0
if "label" not in submission.columns:
    print("❌ 警告：合併後完全沒有 label 欄位，將補上全 0。")
    submission["label"] = 0

# 填補缺值
submission["label"] = submission["label"].fillna(0).astype(int)
submission["probability"] = submission["probability"].fillna(0)

# ==========================
# 9️⃣ 結果統計
# ==========================
print(f"💾 已輸出完整結果：{output_full}")
print(f"🏁 已輸出比賽用結果（acct,label 格式）：{submit_file}")
print("✅ label 唯一值 =", submission["label"].unique())
print("✅ acct 數量 =", len(submission))

# ==========================
# 🔟 輸出結果
# ==========================
submission.rename(columns={"acct_id": "acct"}, inplace=True)
submission["acct"] = submission["acct"].astype(str)
submission["label"] = submission["label"].astype(int)

submission.to_csv(output_full, index=False)
submission[["acct", "label"]].to_csv(submit_file, index=False)

prop = submission["label"].value_counts(normalize=True).rename("proportion")
print("📊 label 分布：")
print(prop)

# 額外輸出機率統計
print("\n🔍 機率統計分布：")
print(submission["probability"].describe())

for t in [0.5, 0.4, 0.3, 0.2]:
    abnormal = (submission["probability"] >= t).sum()
    print(f"門檻 {t:.2f} → {abnormal} 筆被判為異常 ({abnormal/len(submission)*100:.2f}%)")

print("✅ 完成，可直接上傳 AI CUP！")
