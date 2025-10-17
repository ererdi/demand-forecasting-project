# src/feature_engineering.py
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# --- Dosya yolları ---
DATA_DIR = "data"
train_path = os.path.join(DATA_DIR, "train.csv")
stores_path = os.path.join(DATA_DIR, "stores.csv")
oil_path = os.path.join(DATA_DIR, "oil.csv")
holidays_path = os.path.join(DATA_DIR, "holidays_events.csv")
transactions_path = os.path.join(DATA_DIR, "transactions.csv")

# --- CSV Dosyalarını Okuma ---
print(" Veriler okunuyor...")
train = pd.read_csv(train_path, parse_dates=["date"])
stores = pd.read_csv(stores_path)
oil = pd.read_csv(oil_path, parse_dates=["date"])
holidays = pd.read_csv(holidays_path, parse_dates=["date"])
transactions = pd.read_csv(transactions_path, parse_dates=["date"])

# --- Tarih bazlı özellikler üretme ---
print(" Tarih bazlı özellikler üretiliyor...")
train["year"] = train["date"].dt.year
train["month"] = train["date"].dt.month
train["day"] = train["date"].dt.day
train["day_of_week"] = train["date"].dt.dayofweek

# --- Mağaza bilgilerini ekleme ---
train = train.merge(stores, on="store_nbr", how="left")

# --- Tatil bilgilerini ekleme (esnek versiyon) ---
print(" Tatil bilgileri ekleniyor...")

if "type" in holidays.columns:
    holidays_temp = holidays[["date", "type"]].rename(columns={"type": "holiday_type"})
elif "description" in holidays.columns:
    holidays_temp = holidays[["date", "description"]].rename(columns={"description": "holiday_type"})
else:
    print(" 'holidays_events.csv' dosyasında 'type' veya 'description' kolonu bulunamadı, tatil verisi eklenmeyecek.")
    holidays_temp = pd.DataFrame(columns=["date", "holiday_type"])

train = train.merge(holidays_temp, on="date", how="left")
train["is_holiday"] = np.where(train["holiday_type"].notnull(), 1, 0)

# --- Petrol fiyatlarını (oil) ekleme ---
train = train.merge(oil, on="date", how="left")

# --- Eksik değerleri doldurma ---
train["dcoilwtico"].fillna(method="ffill", inplace=True)

# --- Rolling mean (hareketli ortalama) özelliği ---
print(" Hareketli ortalama hesaplanıyor...")
train["rolling_sales_mean_7"] = train.groupby(["store_nbr", "family"])["sales"].transform(lambda x: x.rolling(7, 1).mean())

# ---  Yeni Feature Engineering Aşaması ---
print(" Ek feature’lar (lag ve davranışsal özellikler) ekleniyor...")

# 7 ve 14 günlük gecikmeli satış değerleri
train["sales_lag_7"] = train.groupby(["store_nbr", "family"])["sales"].shift(7)
train["sales_lag_14"] = train.groupby(["store_nbr", "family"])["sales"].shift(14)

# Hafta sonu bilgisi
train["is_weekend"] = train["day_of_week"].isin([5, 6]).astype(int)

# Eksik lag değerlerini doldur
train[["sales_lag_7", "sales_lag_14"]] = train[["sales_lag_7", "sales_lag_14"]].fillna(0)

# Ürün ailesi (family) encoding
le = LabelEncoder()
train["family_encoded"] = le.fit_transform(train["family"])

# --- Gereksiz kolonları temizleme ---
train.drop(columns=["id"], inplace=True)

# --- Sonuç ---
print(" Feature engineering tamamlandı!")
print("Yeni veri şekli:", train.shape)
print(train.head())

# --- Kaydetme ---
os.makedirs("outputs", exist_ok=True)
output_path = "outputs/train_featured.csv"
train.to_csv(output_path, index=False)
print(f" Yeni veri dosyası kaydedildi: {output_path}")
