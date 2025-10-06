# ===============================================================
# build_features_v4_behavior_progress.py
# 功能：進階帳號特徵建立（含日期修正 + 進度顯示）
# ===============================================================

import os
import numpy as np
import pandas as pd
from scipy.stats import entropy

print("📂 讀取交易資料：dataset/acct_transaction.csv")
df = pd.read_csv("dataset/acct_transaction.csv")

print("🧹 資料清理中 ...")
df["txn_amt"] = pd.to_numeric(df["txn_amt"], errors="coerce").fillna(0)

# 將以「第1天為1」的日序轉為日期
base_date = pd.Timestamp("2020-01-01")
df["txn_date"] = pd.to_datetime(base_date + pd.to_timedelta(df["txn_date"].astype(float) - 1, unit="D"), errors="coerce")

# 將時間文字轉換為秒數
if df["txn_time"].dtype == "object":
    t = pd.to_datetime(df["txn_time"], errors="coerce", format="%H:%M:%S")
    df["txn_time"] = t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second
else:
    df["txn_time"] = pd.to_numeric(df["txn_time"], errors="coerce").fillna(0)

df["hour"] = (df["txn_time"] // 3600).astype(int)
df["is_self_txn"] = df["is_self_txn"].fillna("UNK")

print("📊 彙整帳號特徵中 ...")
g = df.groupby("from_acct")
features = g.agg({
    "txn_amt": ["count", "sum", "mean", "max", "min", "std"],
    "to_acct": "nunique",
    "channel_type": "nunique",
    "txn_date": ["min", "max"]
})
features.columns = [
    "txn_count", "total_amt", "avg_amt", "max_amt", "min_amt", "std_amt",
    "to_acct_nunique", "channel_nunique", "first_txn", "last_txn"
]
features["std_amt"] = features["std_amt"].fillna(0)

# === 進階特徵 ===
print("🧠 建立進階行為特徵 ...")
features["active_days"] = (features["last_txn"] - features["first_txn"]).dt.days + 1
features["txn_per_day"] = features["txn_count"] / features["active_days"].replace(0, 1)

print("🧩 計算 night_txn_ratio ...")
night_ratio = df[(df["hour"] >= 0) & (df["hour"] <= 6)].groupby("from_acct").size() / g.size()
features["night_txn_ratio"] = night_ratio.fillna(0)

print("🧩 計算 self_txn_ratio ...")
self_ratio = (df["is_self_txn"] == "Y").groupby(df["from_acct"]).mean()
features["self_txn_ratio"] = self_ratio.fillna(0)

print("🧩 計算 weekday_ratio ...")
features["weekday_ratio"] = g["txn_date"].apply(lambda x: np.mean(x.dt.dayofweek < 5) if len(x) > 0 else 0)

print("🧩 計算 peak_hour_ratio ...")
features["peak_hour_ratio"] = g["hour"].apply(lambda x: np.mean((x >= 9) & (x <= 17)) if len(x) > 0 else 0)

print("🧩 計算 txn_hour_entropy ...")
features["txn_hour_entropy"] = g["hour"].apply(lambda x: entropy(x.value_counts(normalize=True), base=2))

print("🧩 計算 channel_entropy ...")
features["channel_entropy"] = g["channel_type"].apply(lambda x: entropy(x.value_counts(normalize=True), base=2))

print("🧩 計算 q90_over_meann & median_over_mean ...")
features["q90_over_meann"] = g["txn_amt"].apply(lambda x: np.percentile(x, 90) / (x.mean() + 1e-6))
features["median_over_mean"] = g["txn_amt"].apply(lambda x: x.median() / (x.mean() + 1e-6))

print("🧩 計算 most_used_channel_ratio ...")
features["most_used_channel_ratio"] = g["channel_type"].apply(lambda x: x.value_counts(normalize=True).max() if len(x) > 0 else 0)

# 輸出
os.makedirs("feature_data_v4", exist_ok=True)
features.reset_index(inplace=True)
features.rename(columns={"from_acct": "acct_id"}, inplace=True)
features.to_csv("feature_data_v4/account_features.csv", index=False)

print("✅ 已輸出帳號特徵檔：feature_data_v4/account_features.csv")
print(f"📊 特徵總數：{features.shape[1]}")
