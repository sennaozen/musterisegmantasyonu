# ============================================================
#                MÜŞTERİ SEGMENTASYONU PROJESİ
# ============================================================

# ---------------- KÜTÜPHANELER ----------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# ---------------- VERİYİ YÜKLE ----------------
df = pd.read_csv("Mall_Customers.csv")

print("\n--- Veri Önizleme ---")
print(df.head())
print("\n--- Veri Bilgisi ---")
print(df.info())


# ---------------- KORELASYON ISI HARİTASI ----------------
num_cols = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
corr = df[num_cols].corr()

plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Korelasyon Isı Haritası (-1 < r < 1)")
plt.show()


# ---------------- REGRESYON ANALİZİ + SCATTER ----------------
X = df[["Annual Income (k$)"]]
y = df["Spending Score (1-100)"]

lin_reg = LinearRegression()
lin_reg.fit(X, y)

print("\n--- Regresyon Sonuçları ---")
print("Beta1 Katsayısı:", lin_reg.coef_[0])
print("Beta0 Sabit:", lin_reg.intercept_)
print("R2 Skoru:", lin_reg.score(X, y))

plt.figure(figsize=(7, 5))
plt.scatter(X, y, alpha=0.7, label="Veri Noktaları")
plt.plot(X, lin_reg.predict(X), color="red", linewidth=2, label="Regresyon Doğrusu")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Gelir - Harcama İlişkisi (Regresyon Analizi)")
plt.legend()
plt.show()


# ---------------- 2D YOĞUNLUK ISI HARİTASI ----------------
plt.figure(figsize=(6, 5))
plt.hist2d(df["Annual Income (k$)"], df["Spending Score (1-100)"], bins=20, cmap="plasma")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Gelir - Harcama Yoğunluk Isı Haritası (2D Histogram)")
plt.colorbar(label="Yoğunluk")
plt.show()


# ---------------- K-MEANS KÜMELEME (Algoritma 1) ----------------
features = df[["Annual Income (k$)", "Spending Score (1-100)"]]

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=5, random_state=42, n_init="auto")
df["KMeans_Cluster"] = kmeans.fit_predict(features_scaled)

plt.figure(figsize=(7, 5))
sns.scatterplot(
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="KMeans_Cluster",
    data=df,
    palette="Set1"
)
plt.title("K-Means ile Müşteri Segmentasyonu")
plt.legend(title="Küme")
plt.show()


# ---------------- HİYERARŞİK KÜMELEME (Algoritma 2) ----------------
agg = AgglomerativeClustering(n_clusters=5)
df["Agglo_Cluster"] = agg.fit_predict(features_scaled)

plt.figure(figsize=(7, 5))
sns.scatterplot(
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Agglo_Cluster",
    data=df,
    palette="Set2"
)
plt.title("Hiyerarşik Kümeleme ile Müşteri Segmentasyonu")
plt.legend(title="Küme")
plt.show()


# ---------------- SEGMENT ÖZET TABLOSU ----------------
summary = df.groupby("KMeans_Cluster")[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].mean()

print("\n--- Segment Özeti (K-Means) ---")
print(summary)


print("\n----- PROJE TAMAMLANDI -----")
