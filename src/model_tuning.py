# -*- coding: utf-8 -*-
"""
 AŞAMA 4.3 – Hyperparameter Tuning (LightGBM)
 Amaç:
LightGBM modelinin performansını optimize etmek için Grid Search yöntemiyle 
en iyi parametre kombinasyonunu bulmak.
"""

# --- Kütüphaneler ---
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from lightgbm import LGBMRegressor

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

# --- Veri bölme ---
print(" Veri train/test olarak ayrılıyor...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- LightGBM model tanımı ---
model = LGBMRegressor(random_state=42)

# --- Grid Search parametre ızgarası ---
param_grid = {
    'num_leaves': [31, 50, 70],
    'max_depth': [5, 8, 12],
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [100, 300, 500]
}

# --- Grid Search başlat ---
print("\n Grid Search başlatılıyor... (bu işlem birkaç dakika sürebilir)")
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,                  # 3-fold cross validation
    scoring='neg_root_mean_squared_error',
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# --- En iyi parametreleri göster ---
best_params = grid_search.best_params_
print("\n En iyi parametre kombinasyonu bulundu:")
for k, v in best_params.items():
    print(f"   {k}: {v}")

# --- En iyi modelle test seti üzerinde tahmin ---
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n Optimize Edilmiş LightGBM Sonuçları:")
print(f"   RMSE: {rmse:.2f}")
print(f"   MAPE: {mape:.2f}")
print(f"   R²:   {r2:.3f}")

# --- Kaydetme ---
os.makedirs("outputs", exist_ok=True)
summary_path = "outputs/best_params_lightgbm.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(" Best LightGBM Parameters:\n")
    for k, v in best_params.items():
        f.write(f"{k}: {v}\n")
    f.write(f"\nRMSE: {rmse:.2f}\nMAPE: {mape:.2f}\nR²: {r2:.3f}\n")

print(f"\n En iyi parametreler kaydedildi: {summary_path}")

# --- En iyi modeli kaydet ---
from joblib import dump
dump(grid_search.best_estimator_, "outputs/best_model.joblib")
print(" En iyi model kaydedildi: outputs/best_model.joblib")
