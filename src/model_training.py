# -*- coding: utf-8 -*-
"""
 AŞAMA 4 – Model Training & Evaluation
 Amaç:
Hazırlanan veriyi (train_featured.csv) kullanarak farklı regresyon modelleriyle satış tahmini yapmak
ve model performanslarını (RMSE, MAPE, SMAPE, WMAPE, R²) karşılaştırmak.
"""

# --- Gerekli kütüphaneler ---
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor

# --- Özel metrik fonksiyonları ---
def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    denom = (np.abs(y_true) + np.abs(y_pred)) + 1e-8
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / denom)

def wmape(y_true, y_pred):
    """Weighted Mean Absolute Percentage Error"""
    return 100 * np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true) + 1e-8)

# --- Veri Yükleme ---
print(" Veri yükleniyor...")
data_path = "outputs/train_featured.csv"
df = pd.read_csv(data_path, parse_dates=["date"])

print(" Veri başarıyla yüklendi. Boyut:", df.shape)

# --- Eksik değerleri doldurma (mean/median yöntemi) ---
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    median_value = df[col].median()
    df[col] = df[col].fillna(median_value)

# --- Model için gerekli kolonlar ---
target = "sales"
features = [
    "store_nbr", "onpromotion", "year", "month", "day", "day_of_week",
    "is_holiday", "dcoilwtico", "rolling_sales_mean_7",
    "sales_lag_7", "sales_lag_14", "is_weekend", "family_encoded"
]

X = df[features]
y = df[target]

# --- Train / Test Split ---
print(" Veri train ve test olarak ayrılıyor...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train set:", X_train.shape, "Test set:", X_test.shape)

# --- Modelleri Tanımlama ---
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=10),
    "LightGBM": LGBMRegressor(random_state=42, n_estimators=300, learning_rate=0.1)
}

# --- Model Eğitimi ve Değerlendirme ---
results = []

for name, model in models.items():
    print(f"\n {name} modeli eğitiliyor...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # MAPE – sıfır satışlar maskeleniyor
    mask = y_test != 0
    mape = mean_absolute_percentage_error(y_test[mask], y_pred[mask])

    # SMAPE ve WMAPE
    smape_val = smape(y_test, y_pred)
    wmape_val = wmape(y_test, y_pred)

    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "RMSE": rmse,
        "MAPE": mape,
        "SMAPE": smape_val,
        "WMAPE": wmape_val,
        "R²": r2
    })

    print(f"{name} sonuçları -> RMSE: {rmse:.2f}, MAPE: {mape:.2f}, SMAPE: {smape_val:.2f}, WMAPE: {wmape_val:.2f}, R²: {r2:.3f}")

# --- Sonuçları Kaydetme ---
results_df = pd.DataFrame(results)
os.makedirs("outputs", exist_ok=True)
results_path = "outputs/model_results.csv"
results_df.to_csv(results_path, index=False)

print("\n Tüm model sonuçları:")
print(results_df)
print(f"\n Sonuç dosyası kaydedildi: {results_path}")
