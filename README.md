## LGBM_Test_0.1405563
## 📦 專案說明

本版本 (`v4`) 為針對「帳戶警示預測」任務的 **進階行為特徵強化版**，  

本版本在 `v3` 基礎上進行系統性優化，  
包含特徵工程、欠採樣平衡、閾值調整與訓練曲線紀錄功能。

---

## 🚀 專案流程架構
flowchart TD
    A[dataset/acct_transaction.csv] --> B[build_features_v4_behavior.py]
    B --> C[feature_data_v4/account_features.csv]
    C --> D[prepare_trainset_balanced_v4.py]
    D --> E[train_data_v4/train_data.csv]
    E --> F[train_model_LGBM_v4.py]
    F --> G[train_data_v4/lightgbm_model.pkl]
    G --> H[predict_accounts_v4.py]
    H --> I[results_v4/predict_for_submit_v4.csv]


## 🧩 特徵工程 (Feature Engineering)
🏗️ 檔案：build_features_v4_behavior.py

從 acct_transaction.csv 建立帳號級特徵。
相較於 v3，本版新增 9 項行為熵與時間特徵：

類別	特徵名稱	說明
交易統計	txn_count, total_amt, avg_amt, max_amt, min_amt, std_amt	基本交易統計
期間特徵	first_txn, last_txn, active_days, txn_per_day	交易活躍時間
行為比例	night_txn_ratio, self_txn_ratio	夜間、自轉比例
📆 進階行為	weekday_ratio, peak_hour_ratio	平日 / 尖峰時段比例
📊 熵特徵	txn_hour_entropy, channel_entropy	時間與通道分布熵
💰 金額分布	q90_over_meann, median_over_mean	高金額與分布平衡性
🔁 使用習慣	most_used_channel_ratio	最常用通道佔比

## 🧠模型優化建

多比例欠採樣測試

可測 1:2, 1:3, 1:5 觀察 Recall 提升效果

# GPU 加速訓練

安裝 RAPIDS (cuDF + LightGBM-GPU)，可加速百倍

# 閾值微調

測試 0.3~0.7 閾值對 precision/recall 平衡影響


