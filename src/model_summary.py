# -*- coding: utf-8 -*-
"""
📊 AŞAMA 6 – Model Sonuç Özeti (Forecast vs Actual)
🎯 Amaç:
Tahmin edilen satışlar ile gerçek satışları karşılaştırarak
işe yönelik anlamlı özet tablolar üretmek.
"""

import pandas as pd
import os

# --- Dosya yolları ---
data_path = "outputs/train_featured.csv"
results_path = "outputs/model_results.csv"
best_model_path = "outputs/best_model_summary.txt"

# --- Veri yükleme ---
print("📂 Veriler yükleniyor...")
train_df = pd.read_csv(data_path, parse_dates=["date"])
print("✅ train_featured yüklendi:", train_df.shape)

# Model çıktısı (tahminler)
# Burada, test verisinden gerçek satışlar (y_test) ve tahminler (y_pred) birleştiriliyor
from joblib import load

if os.path.exists("outputs/best_model.joblib"):
    best_model = load("outputs/best_model.joblib")
    print("✅ En iyi model yüklendi (LightGBM - Tuned)")

    # Özellikler
    features = [
        "store_nbr", "onpromotion", "year", "month", "day", "day_of_week",
        "is_holiday", "dcoilwtico", "rolling_sales_mean_7",
        "sales_lag_7", "sales_lag_14", "is_weekend", "family_encoded"
    ]
    
    X = train_df[features]
    y_true = train_df["sales"]
    y_pred = best_model.predict(X)
    
    # Tahmin-sonuç karşılaştırma tablosu
    summary = train_df[["date", "store_nbr", "family"]].copy()
    summary["actual_sales"] = y_true
    summary["predicted_sales"] = y_pred
    summary["error"] = summary["predicted_sales"] - summary["actual_sales"]
    summary["error_percent"] = (summary["error"] / summary["actual_sales"]) * 100

    # Mağaza ve ürün bazlı özet
    business_summary = (
        summary.groupby(["store_nbr", "family"])
        .agg(
            actual_mean=("actual_sales", "mean"),
            predicted_mean=("predicted_sales", "mean"),
            mean_error_percent=("error_percent", "mean")
        )
        .reset_index()
        .sort_values("mean_error_percent")
    )

    # Sonuçları kaydet
    os.makedirs("outputs", exist_ok=True)
    output_path = "outputs/business_summary.csv"
    business_summary.to_csv(output_path, index=False)
    print(f"💾 Özet tablo kaydedildi: {output_path}")

else:
    print("⚠️ Uyarı: best_model.joblib bulunamadı. Önce model_tuning.py çalıştırılmalı.")
