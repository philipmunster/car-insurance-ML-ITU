import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import pairplot
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv('claims_train.csv')

# Drop Exposure > 1, IDpol
df = df[df['Exposure'] <= 1]
df = df.drop(columns=['IDpol'])

# Drop the categorical variables
df = df.drop(columns = ['VehBrand', 'Region'])

df = df.drop(columns=["Area", "VehGas"])

# Log-Transforming Skewed Features
df['log_Density'] = np.log1p(df['Density'])
df['log_BonusMalus'] = np.log1p(df['BonusMalus'])

# Implement Target Variable, drop ClaimNb, Exposure
df['cpy'] = df['ClaimNb'] / df['Exposure']

# Standard Scaling Numerical Features (not the target variable cpy)
clustering_features = [
    'Exposure',
    'VehAge',
    'DrivAge',
    'VehPower',
    'log_Density',
    'log_BonusMalus',
]

X = df[clustering_features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(
    X_scaled,
    columns=X.columns,
    index=X.index
)

X_np = X_scaled.to_numpy(dtype=np.float64, copy=False)

k_values = range(2, 7)
results = []

for k in k_values:
    km = MiniBatchKMeans(
        n_clusters=k,
        n_init=10,
        batch_size=5000,
        max_iter=200,
        random_state=42
    )
    labels = km.fit_predict(X_np)

    sil = silhouette_score(
        X_np, labels,
        sample_size=10000,
        random_state=42
    )

    results.append({"k": k, "silhouette": sil})

results_df = pd.DataFrame(results)
print(results_df)

best_k = results_df.loc[results_df["silhouette"].idxmax(), "k"]
print("Chosen k:", best_k)

final_km = MiniBatchKMeans(
    n_clusters=int(best_k),
    n_init=10,
    batch_size=5000,
    max_iter=200,
    random_state=42
)
final_labels = final_km.fit_predict(X_np)

# sample for visualization
rng = np.random.default_rng(1)
sample_size = 20000
idx = rng.choice(len(X_np), size=sample_size, replace=False)

X_sample = X_np[idx]
labels_sample = final_labels[idx]

# PCA for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_sample)

# figure
fig, ax = plt.subplots(figsize=(6, 5))

scatter = ax.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=labels_sample,
    cmap="coolwarm",
    s=5,
    alpha=0.4,
    linewidths=0
)

ax.set_title("Clusters in PCA Space")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")

plt.tight_layout()
plt.show()