# -*- coding: utf-8 -*-
"""
 AŞAMA 7 – Yönetim Özeti Tablosu
 Amaç:
Business_summary.csv verisini sadeleştirerek yönetim dilinde yorumlanabilir hale getirmek.
"""

import pandas as pd
import os

# --- Veri yükleme ---
file_path = "outputs/business_summary.csv"
df = pd.read_csv(file_path)

# --- Yorum sütunu oluştur ---
def yorumla(error):
    if error > 5:
        return "Tahmin fazla (overpredict)"
    elif error < -5:
        return "Tahmin düşük (underpredict)"
    else:
        return "Başarılı tahmin"

df["Yorum"] = df["mean_error_percent"].apply(yorumla)

# --- Sütunları yeniden adlandır ---
df = df.rename(columns={
    "store_nbr": "Mağaza",
    "family": "Ürün Grubu",
    "actual_mean": "Gerçek Satış Ortalaması",
    "predicted_mean": "Tahmin Ortalaması",
    "mean_error_percent": "Ortalama Hata (%)"
})

# --- En iyi & en kötü 10 sonucu çıkar ---
top10 = df.sort_values("Ortalama Hata (%)").head(10)
bottom10 = df.sort_values("Ortalama Hata (%)", ascending=False).head(10)

# --- Kaydet ---
os.makedirs("outputs", exist_ok=True)
output_path = "outputs/management_report.csv"
df.to_csv(output_path, index=False)
print(f" Yönetim özeti kaydedildi: {output_path}")

print("\n En iyi 5 tahmin:")
print(top10.head(5)[["Mağaza", "Ürün Grubu", "Ortalama Hata (%)", "Yorum"]])
