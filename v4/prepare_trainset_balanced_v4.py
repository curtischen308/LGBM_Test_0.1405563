# ===============================================
# prepare_trainset_balanced_v4.py
# 功能：
#   - 從特徵表(account_features.csv) 與 警示帳戶(acct_alert.csv)
#     建立 1:1 欠採樣訓練集
#   - 支援自動欄位偵測、異常比例分析
#   - 可調整倍數 (alert : normal = 1 : N)
# ===============================================

import pandas as pd
import os

def prepare_trainset_v4(
    feature_file="feature_data_v4/account_features.csv",
    alert_file="dataset/acct_alert.csv",
    output_dir="train_data_v4",
    ratio=2  # 欠採樣比例，1 代表 1:1
):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "train_data.csv")

    print(f"📂 讀取特徵檔案：{feature_file}")
    features = pd.read_csv(feature_file)
    print(f"📂 讀取警示帳戶：{alert_file}")
    alerts = pd.read_csv(alert_file)

    # ======================
    # 欄位偵測
    # ======================
    alert_col = None
    for c in alerts.columns:
        if c.strip().lower() in ["acct", "acct_id", "alert_key"]:
            alert_col = c
            break
    if alert_col is None:
        raise ValueError("❌ acct_alert.csv 缺少帳號欄位（acct / acct_id / alert_key）")

    alerts.rename(columns={alert_col: "acct_id"}, inplace=True)
    features["acct_id"] = features["acct_id"].astype(str).str.strip()
    alerts["acct_id"] = alerts["acct_id"].astype(str).str.strip()

    # ======================
    # 建立 label 欄位
    # ======================
    print("⚙️ 建立標籤欄位 ...")
    alert_accounts = set(alerts["acct_id"])
    features["alert_flag"] = features["acct_id"].isin(alert_accounts).astype(int)

    # ======================
    # 分組統計
    # ======================
    alert_df = features[features["alert_flag"] == 1]
    normal_df = features[features["alert_flag"] == 0]
    print(f"🚨 警示帳戶：{len(alert_df)}")
    print(f"🟢 正常帳戶：{len(normal_df)}")

    # ======================
    # 欠採樣
    # ======================
    if len(alert_df) == 0:
        raise ValueError("❌ 警示帳戶為空，請檢查 acct_alert.csv")

    sample_size = min(len(normal_df), int(len(alert_df) * ratio))
    print(f"⚖️ 欠採樣正常帳戶：{sample_size} 筆 (比例 {ratio}:1)")
    normal_sample = normal_df.sample(n=sample_size, random_state=42)

    train_data = pd.concat([alert_df, normal_sample]).sample(frac=1, random_state=42)

    # ======================
    # 防呆檢查
    # ======================
    if train_data["alert_flag"].sum() == 0:
        raise ValueError("❌ 所有樣本標籤皆為 0，請確認 alert 合併是否正確。")

    # ======================
    # 輸出結果
    # ======================
    train_data.to_csv(output_file, index=False)
    print(f"✅ 完成！輸出訓練資料：{output_file}")
    print(f"📊 樣本統計：")
    print(train_data["alert_flag"].value_counts(normalize=True).rename("比例"))

if __name__ == "__main__":
    prepare_trainset_v4()
