# -*- coding: utf-8 -*-
"""
🧭 AŞAMA 5 – Gelecek Talep Tahmini (Forecast Generation)
🎯 Amaç:
- best_model.joblib yüklenir
- Son tarihten itibaren 5 gün ileriye her mağaza×ürün için tahmin üretilir (test modu)
- Lag ve rolling özellikleri iteratif olarak güncellenir
- Çıktı: outputs/forecast_results.csv
"""

import os
import numpy as np
import pandas as pd
from joblib import load

# -------------------------------
# 🔧 Kullanıcı parametreleri (Test Modu)
# -------------------------------
FORECAST_DAYS = 5          # ⚡ hızlı test için sadece 5 günlük tahmin
PROMO_SCENARIO = 0         # Gelecek günler için kampanya yok varsayımı
OIL_FFILL = True           # Petrol fiyatını ileri doldur (gelecek için aynı değeri kullan)
# -------------------------------

DATA_DIR = "data"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

train_feat_path = os.path.join(OUT_DIR, "train_featured.csv")
best_model_path = os.path.join(OUT_DIR, "best_model.joblib")
holidays_path = os.path.join(DATA_DIR, "holidays_events.csv")
oil_path = os.path.join(DATA_DIR, "oil.csv")

FEATURES = [
    "store_nbr", "onpromotion",
    "year", "month", "day", "day_of_week", "is_weekend",
    "is_holiday", "dcoilwtico",
    "rolling_sales_mean_7", "sales_lag_7", "sales_lag_14",
    "family_encoded"
]

print("📦 Veri ve model yükleniyor...")
if not os.path.exists(train_feat_path):
    raise FileNotFoundError(f"Bulunamadı: {train_feat_path}")

if not os.path.exists(best_model_path):
    raise FileNotFoundError(f"Bulunamadı: {best_model_path}")

df = pd.read_csv(train_feat_path, parse_dates=["date"])
model = load(best_model_path)
print("✅ train_featured ve best_model yüklendi.")
print("   train_featured shape:", df.shape)

# Yardımcı tablolar
oil = pd.read_csv(oil_path, parse_dates=["date"])
holidays = pd.read_csv(holidays_path, parse_dates=["date"])

# Tatil kolonunu esnek yakala
if "type" in holidays.columns:
    hol_temp = holidays[["date", "type"]].rename(columns={"type": "holiday_type"})
elif "description" in holidays.columns:
    hol_temp = holidays[["date", "description"]].rename(columns={"description": "holiday_type"})
else:
    hol_temp = pd.DataFrame(columns=["date", "holiday_type"])

# ---- Temel referanslar
last_date = df["date"].max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                             periods=FORECAST_DAYS, freq="D")

print(f"Son eğitim tarihi: {last_date.date()} → Tahmin aralığı: {future_dates[0].date()} ~ {future_dates[-1].date()}")

# Mağaza×ürün evreni
pairs = df[["store_nbr", "family", "family_encoded"]].drop_duplicates().reset_index(drop=True)
future = pairs.merge(pd.DataFrame({"date": future_dates}), how="cross")
future["onpromotion"] = PROMO_SCENARIO

# Takvim/kategori feature'ları
future["year"] = future["date"].dt.year
future["month"] = future["date"].dt.month
future["day"] = future["date"].dt.day
future["day_of_week"] = future["date"].dt.dayofweek
future["is_weekend"] = future["day_of_week"].isin([5, 6]).astype(int)

# Tatil ve petrol
future = future.merge(hol_temp, on="date", how="left")
future["is_holiday"] = np.where(future["holiday_type"].notnull(), 1, 0)
future.drop(columns=["holiday_type"], inplace=True)

future = future.merge(oil, on="date", how="left")
if OIL_FFILL:
    last_oil = oil["dcoilwtico"].dropna().iloc[-1]
    future["dcoilwtico"].fillna(last_oil, inplace=True)

# -------------------------------
# 🔁 İteratif tahmin (lag & rolling güncelleme)
# -------------------------------
history = (
    df.sort_values("date")
      .groupby(["store_nbr", "family"], group_keys=False)
      .tail(30)[["store_nbr", "family", "date", "sales"]]
)

pred_rows = []
counter = 0

print("Tahmin başlıyor (iteratif güncelleme ile)...")
for (store, fam, fam_enc), g_future in future.groupby(["store_nbr", "family", "family_encoded"]):
    g_hist = history[(history["store_nbr"] == store) & (history["family"] == fam)].copy()
    g_hist = g_hist.sort_values("date")

    if g_hist.empty:
        g_hist = pd.DataFrame({
            "store_nbr": [store]*14,
            "family": [fam]*14,
            "date": pd.date_range(end=last_date, periods=14, freq="D"),
            "sales": [0.0]*14
        })

    for dt in g_future["date"].sort_values():
        last_14 = g_hist.tail(14)["sales"].values
        last_7 = g_hist.tail(7)["sales"].values

        sales_lag_7 = last_7[-1] if len(last_7) >= 7 else 0.0
        sales_lag_14 = last_14[-1] if len(last_14) >= 14 else 0.0
        rolling_7 = np.mean(last_7) if len(last_7) > 0 else 0.0

        base = future[
            (future["store_nbr"] == store) &
            (future["family"] == fam) &
            (future["date"] == dt)
        ].iloc[0]

        row = {
            "store_nbr": int(base["store_nbr"]),
            "onpromotion": int(base["onpromotion"]),
            "year": int(base["year"]),
            "month": int(base["month"]),
            "day": int(base["day"]),
            "day_of_week": int(base["day_of_week"]),
            "is_weekend": int(base["is_weekend"]),
            "is_holiday": int(base["is_holiday"]),
            "dcoilwtico": float(base["dcoilwtico"]) if pd.notna(base["dcoilwtico"]) else 0.0,
            "rolling_sales_mean_7": float(rolling_7),
            "sales_lag_7": float(sales_lag_7),
            "sales_lag_14": float(sales_lag_14),
            "family_encoded": int(fam_enc),
        }

        X_row = pd.DataFrame([row])[FEATURES]
        y_hat = float(model.predict(X_row)[0])

        pred_rows.append({
            "date": dt,
            "store_nbr": store,
            "family": fam,
            "predicted_sales": max(y_hat, 0.0)
        })

        g_hist = pd.concat([
            g_hist,
            pd.DataFrame({"store_nbr":[store], "family":[fam], "date":[dt], "sales":[max(y_hat,0.0)]})
        ], ignore_index=True)

    counter += 1
    if counter % 100 == 0:
        print(f"{counter} mağaza×ürün tahmin edildi...")

# Kaydetme
pred_df = pd.DataFrame(pred_rows).sort_values(["store_nbr", "family", "date"])
out_path = os.path.join(OUT_DIR, "forecast_results.csv")
pred_df.to_csv(out_path, index=False, encoding="utf-8-sig")

print(f"✅ Tahmin tamamlandı. Kaydedildi: {out_path}")
print(pred_df.head(10))
