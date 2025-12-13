import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import pairplot
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


df = pd.read_csv('claims_train.csv')

# Removing wrong entries with Exposure > 1
df = df[df['Exposure'] <= 1]

# Creating target variable 'CPY' (Claims per Year)
df['CPY'] = df['ClaimNb'] / df['Exposure']

# Remove ID, ClaimNb and Exposure columns
df = df.drop(columns=['IDpol', 'ClaimNb', 'Exposure'])


# Select numeric column for PCA
numeric_cols = ['VehAge', 'DrivAge', 'BonusMalus', 'Density', 'VehPower']

# Scale the numeric columns because PCA is affected by the scale of the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[numeric_cols])
df_scaled = pd.DataFrame(df_scaled, columns=numeric_cols)

# Apply PCA
pca = PCA(n_components=2)
pca_features = pca.fit_transform(df_scaled)

# PCA Plot
pca_df = pd.DataFrame(data=pca_features, columns=['PC1', 'PC2'])
plt.figure(figsize=(8,6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.6, s=10, color='steelblue')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Projection')
plt.grid(True)
plt.show()