# ------------------------------
# OptiStock - Data Exploration (EDA)
# ------------------------------

# 1 Gerekli kütüphaneleri içe aktar
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 2 CSV dosyasını oku
# Not: dosya yolu proje yapına göre ../data/train.csv şeklinde
data_path = "data/train.csv"

df = pd.read_csv(data_path)

# 3 İlk 5 satırı göster
print(" İlk 5 Satır:")
print(df.head())

# 4 Veri bilgisi
print("\n Veri Bilgisi:")
print(df.info())

# 5 Sayısal kolonların özet istatistikleri
print("\n İstatistiksel Özeti:")
print(df.describe())

# 6 Eksik değer kontrolü
print("\n Eksik Değer Sayısı:")
print(df.isna().sum())

# 7 Basit satış trendi grafiği
print("\n Günlük toplam satış grafiği oluşturuluyor...")

# Tarih formatına dönüştür
df['date'] = pd.to_datetime(df['date'])

# Tarih bazında satış toplamı
daily_sales = df.groupby('date')['sales'].sum().reset_index()

# Grafik
plt.figure(figsize=(12,5))
plt.plot(daily_sales['date'], daily_sales['sales'], color='steelblue')
plt.title("Günlük Toplam Satış Trendleri")
plt.xlabel("Tarih")
plt.ylabel("Toplam Satış")
plt.grid(True)
plt.tight_layout()
plt.show()
