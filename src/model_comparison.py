# -*- coding: utf-8 -*-
"""
🚀 AŞAMA 4.2 – Model Comparison & Selection
🎯 Amaç:
Üç farklı modelin performansını karşılaştırmak ve en iyi modeli belirlemek.
"""

# --- Kütüphaneler ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Veri Yükleme ---
print("📂 Model sonuçları yükleniyor...")
results_path = "outputs/model_results.csv"
results_df = pd.read_csv(results_path)

print("\n📊 Model performans sonuçları:")
print(results_df)

# --- Grafik Ayarları ---
sns.set(style="whitegrid", palette="pastel")
plt.figure(figsize=(10, 6))

# --- 1. Grafik: RMSE Karşılaştırması ---
plt.subplot(1, 2, 1)
sns.barplot(x="Model", y="RMSE", data=results_df)
plt.title("Model Bazında RMSE Karşılaştırması")
plt.xticks(rotation=20)

# --- 2. Grafik: WMAPE Karşılaştırması ---
plt.subplot(1, 2, 2)
sns.barplot(x="Model", y="WMAPE", data=results_df)
plt.title("Model Bazında WMAPE Karşılaştırması")
plt.xticks(rotation=20)

plt.tight_layout()
plt.show()

# --- En İyi Modeli Belirleme ---
best_model = results_df.loc[results_df["WMAPE"].idxmin(), "Model"]
best_rmse = results_df.loc[results_df["WMAPE"].idxmin(), "RMSE"]
best_r2 = results_df.loc[results_df["WMAPE"].idxmin(), "R²"]

print(f"\n🏆 En iyi model: {best_model}")
print(f"   → RMSE: {best_rmse:.2f}, R²: {best_r2:.3f}")

# --- Kaydetme ---
summary_path = "outputs/best_model_summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("🏆 En İyi Model Özeti\n")
    f.write("====================\n")
    f.write(f"Model: {best_model}\n")
    f.write(f"RMSE: {best_rmse:.2f}\n")
    f.write(f"R²: {best_r2:.3f}\n")

print(f"\n💾 Özet dosyası kaydedildi: {summary_path}")
