# -*- coding: utf-8 -*-
"""
 AŞAMA 6.2 – Forecast Görselleştirme
 Amaç:
Forecast özet dosyalarından (family & store) anlamlı grafikler oluşturmak.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Dosya yolları ---
family_path = "outputs/family_forecast_summary.csv"
store_path = "outputs/store_forecast_summary.csv"

print(" Özet dosyalar yükleniyor...")
family_df = pd.read_csv(family_path)
store_df = pd.read_csv(store_path)
print(" Veriler yüklendi.")

# --- 1 En çok satılacak 10 kategori ---
top10_families = family_df.nlargest(10, "avg_pred_sales")

plt.figure(figsize=(10, 5))
plt.barh(top10_families["family"], top10_families["avg_pred_sales"], color="skyblue")
plt.xlabel("Ortalama Tahmini Satış")
plt.ylabel("Ürün Kategorisi")
plt.title(" En Yüksek Ortalama Satış Beklenen 10 Kategori")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("outputs/top10_families.png")
plt.show()

# --- 2 En az satılacak 10 kategori ---
bottom10_families = family_df.nsmallest(10, "avg_pred_sales")

plt.figure(figsize=(10, 5))
plt.barh(bottom10_families["family"], bottom10_families["avg_pred_sales"], color="salmon")
plt.xlabel("Ortalama Tahmini Satış")
plt.ylabel("Ürün Kategorisi")
plt.title(" En Düşük Ortalama Satış Beklenen 10 Kategori")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("outputs/bottom10_families.png")
plt.show()

# --- 3 En çok satış beklenen 10 mağaza ---
top10_stores = store_df.nlargest(10, "avg_pred_sales")

plt.figure(figsize=(8, 5))
plt.bar(top10_stores["store_nbr"].astype(str), top10_stores["avg_pred_sales"], color="mediumseagreen")
plt.xlabel("Mağaza No")
plt.ylabel("Ortalama Tahmini Satış")
plt.title(" En Yüksek Satış Beklenen 10 Mağaza")
plt.tight_layout()
plt.savefig("outputs/top10_stores.png")
plt.show()

print("\n Grafikler kaydedildi:")
print(" - outputs/top10_families.png")
print(" - outputs/bottom10_families.png")
print(" - outputs/top10_stores.png")
