# -*- coding: utf-8 -*-
"""
 FAST MODEL TUNING (GridSearch'süz)
 Amaç:
GridSearch sonucunda önceden belirlenmiş en iyi LightGBM parametreleriyle modeli hızlıca eğitmek
ve best_model.joblib olarak kaydetmek.
"""

# --- Kütüphaneler ---
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from lightgbm import LGBMRegressor
from joblib import dump

# --- Veri Yükleme ---
print(" Veri yükleniyor...")
data_path = "outputs/train_featured.csv"
df = pd.read_csv(data_path, parse_dates=["date"])
print(" Veri başarıyla yüklendi. Boyut:", df.shape)

# --- Eksik değerleri doldurma ---
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# --- Özellikler / Hedef ---
target = "sales"
features = [
    "store_nbr", "onpromotion", "year", "month", "day", "day_of_week",
    "is_holiday", "dcoilwtico", "rolling_sales_mean_7",
    "sales_lag_7", "sales_lag_14", "is_weekend", "family_encoded"
]

X = df[features]
y = df[target]

# --- Train/Test bölme ---
print(" Veri train/test olarak ayrılıyor...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Önceden belirlenen en iyi parametreler ---
best_params = {
    'learning_rate': 0.1,
    'max_depth': 12,
    'n_estimators': 500,
    'num_leaves': 70,
    'random_state': 42
}

# --- Model oluşturma ve eğitme ---
print(" En iyi parametrelerle LightGBM modeli eğitiliyor...")
best_model = LGBMRegressor(**best_params)
best_model.fit(X_train, y_train)

# --- Test performansı ---
y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n Model Performansı:")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}")
print(f"R²: {r2:.3f}")

# --- Modeli kaydet ---
os.makedirs("outputs", exist_ok=True)
dump(best_model, "outputs/best_model.joblib")
print(" Model kaydedildi: outputs/best_model.joblib")

# --- Özet bilgiyi .txt olarak da kaydet ---
summary_path = "outputs/best_params_lightgbm.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(" Best LightGBM Parameters (Fast Mode):\n")
    for k, v in best_params.items():
        f.write(f"{k}: {v}\n")
    f.write(f"\nRMSE: {rmse:.2f}\nMAPE: {mape:.2f}\nR²: {r2:.3f}\n")
print(f" Özet kaydedildi: {summary_path}")
