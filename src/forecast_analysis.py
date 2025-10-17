# -*- coding: utf-8 -*-
"""
 AŞAMA 6 – Forecast Analizi
 Amaç:
Forecast sonuçlarını iş açısından anlamlı hale getirmek:
- En çok artış / düşüş gösteren kategoriler
- Mağaza bazında satış tahmin ortalamaları
- Genel satış trendi (ortalama tahmin)
"""

import pandas as pd
import numpy as np
import os

# --- Dosya yolu ---
forecast_path = "outputs/forecast_results.csv"
print(" Tahmin verisi yükleniyor...")

df = pd.read_csv(forecast_path, parse_dates=["date"])
print(f" Veri yüklendi: {df.shape}")

# --- Genel ortalama satış ---
overall_avg = df["predicted_sales"].mean()
print(f"\n Genel Ortalama Tahmini Satış: {overall_avg:.2f}")

# --- Kategori bazlı ortalama satış ---
family_summary = (
    df.groupby("family")["predicted_sales"]
    .agg(["mean", "max", "min"])
    .sort_values("mean", ascending=False)
    .reset_index()
)
family_summary.rename(columns={"mean": "avg_pred_sales"}, inplace=True)

# --- Mağaza bazlı ortalama satış ---
store_summary = (
    df.groupby("store_nbr")["predicted_sales"]
    .agg(["mean", "max", "min"])
    .sort_values("mean", ascending=False)
    .reset_index()
)
store_summary.rename(columns={"mean": "avg_pred_sales"}, inplace=True)

# --- En çok artış beklenen kategoriler ---
print("\n En Yüksek Ortalama Satış Beklenen İlk 5 Kategori:")
print(family_summary.head(5))

# --- En az satış beklenen kategoriler ---
print("\n En Düşük Ortalama Satış Beklenen 5 Kategori:")
print(family_summary.tail(5))

# --- Dosya olarak kaydet ---
os.makedirs("outputs", exist_ok=True)
family_summary.to_csv("outputs/family_forecast_summary.csv", index=False)
store_summary.to_csv("outputs/store_forecast_summary.csv", index=False)

print("\n Raporlar kaydedildi:")
print(" - outputs/family_forecast_summary.csv")
print(" - outputs/store_forecast_summary.csv")
