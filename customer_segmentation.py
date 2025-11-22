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

sns.set(style="whitegrid", font_scale=1.1)

# ---------------- VERIYI YUKLE ----------------
df = pd.read_csv("Mall_Customers.csv")

print("\n--- Veri Onizleme ---")
print(df.head())
print("\n--- Veri Bilgisi ---")
print(df.info())

# ============================================================
# 1) KORELASYON ISI HARITASI
# ============================================================
num_cols = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
corr = df[num_cols].corr()

plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Korelasyon Isı Haritasi (-1 < r < 1)")
plt.show()

# ============================================================
# 2) REGRESYON ANALIZI + SCATTER (Gelir -> Harcama)
# ============================================================
X = df[["Annual Income (k$)"]]
y = df["Spending Score (1-100)"]

lin_reg = LinearRegression()
lin_reg.fit(X, y)

print("\n--- Regresyon Sonuclari ---")
print("Beta1 Katsayisi:", lin_reg.coef_[0])
print("Beta0 Sabit:", lin_reg.intercept_)
print("R^2 Skoru:", lin_reg.score(X, y))

plt.figure(figsize=(7, 5))
plt.scatter(X, y, alpha=0.7, label="Veri Noktalari")
plt.plot(X, lin_reg.predict(X), color="red", linewidth=2, label="Regresyon Dogrusu")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Gelir - Harcama Iliskisi (Regresyon Analizi)")
plt.legend()
plt.show()

# ============================================================
# 3) 2D YOGUNLUK ISI HARITASI (Gelir - Harcama)
# ============================================================
plt.figure(figsize=(6, 5))
plt.hist2d(df["Annual Income (k$)"], df["Spending Score (1-100)"],
           bins=20, cmap="plasma")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Gelir - Harcama Yogunluk Isı Haritasi (2D Histogram)")
plt.colorbar(label="Yogunluk")
plt.show()

# ============================================================
# 4) ILISKISEL EK GRAFIK 1: PAIRPLOT (Age, Income, Score)
# ============================================================
plt.figure()  # bos fig, bazen pairplot ile çakismasin diye
sns.pairplot(df[num_cols])
plt.suptitle("Yas - Gelir - Harcama Pairplot", y=1.02)
plt.show()

# ============================================================
# 5) ILISKISEL EK GRAFIK 2: CINSIYETE GORE HARCAMA BOXPLOT
# ============================================================
plt.figure(figsize=(6, 4))
sns.boxplot(x="Genre", y="Spending Score (1-100)", data=df)
plt.title("Cinsiyete Gore Harcama Dagilimi")
plt.xlabel("Cinsiyet")
plt.ylabel("Spending Score (1-100)")
plt.show()

# ============================================================
# 6) ILISKISEL EK GRAFIK 3: YAS GRUBU vs ORTALAMA HARCAMA
# ============================================================
# Yas araliklari olusturalim
bins = [15, 25, 35, 45, 55, 70]
labels = ["15-25", "25-35", "35-45", "45-55", "55-70"]
df["Age_Group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)

age_group_summary = df.groupby("Age_Group")["Spending Score (1-100)"].mean().reset_index()

plt.figure(figsize=(7, 4))
sns.barplot(x="Age_Group", y="Spending Score (1-100)", data=age_group_summary)
plt.title("Yas Gruplarina Gore Ortalama Harcama Skoru")
plt.xlabel("Yas Grubu")
plt.ylabel("Ortalama Spending Score")
plt.show()

# ============================================================
# 7) K-MEANS KUMELEME (Algoritma 1)
# ============================================================
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
plt.title("K-Means ile Musteri Segmentasyonu")
plt.legend(title="Kume")
plt.show()

# ============================================================
# 8) ILISKISEL EK GRAFIK 4: KUMELERE GORE GELIR-HARCAMA- YAS
# ============================================================
plt.figure(figsize=(7, 5))
sns.scatterplot(
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    size="Age",
    hue="KMeans_Cluster",
    data=df,
    palette="Set1",
    sizes=(20, 200),
    alpha=0.8
)
plt.title("K-Means Kumeleri: Gelir-Harcama (Nokta Boyutu: Yas)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(title="Kume", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# ============================================================
# 9) HIYERARSİK KUMELEME (Algoritma 2)
# ============================================================
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
plt.title("Hiyerarsik Kumeleme ile Musteri Segmentasyonu")
plt.legend(title="Kume")
plt.show()

# ============================================================
# 10) ILISKISEL EK GRAFIK 5: KUME BUYUKLUKLERI BAR GRAFIGI
# ============================================================
cluster_counts = df["KMeans_Cluster"].value_counts().sort_index()

plt.figure(figsize=(6, 4))
sns.barplot(x=cluster_counts.index, y=cluster_counts.values)
plt.title("K-Means Kumelerinin Musteri Sayilari")
plt.xlabel("Kume")
plt.ylabel("Musteri Sayisi")
plt.show()

# ============================================================
# 11) SEGMENT OZET TABLOSU
# ============================================================
summary = df.groupby("KMeans_Cluster")[["Age",
                                        "Annual Income (k$)",
                                        "Spending Score (1-100)"]].mean()

print("\n--- Segment Ozeti (K-Means) ---")
print(summary)

print("\n----- PROJE TAMAMLANDI -----")
