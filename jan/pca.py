import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import pairplot
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv('../claims_train.csv')

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
df = df.drop(columns=['ClaimNb'])

# Standard Scaling Numerical Features (not the target variable cpy)
pca_features = [
    'Exposure',
    'VehAge',
    'DrivAge',
    'VehPower',
    'log_Density',
    'log_BonusMalus',
]

X = df[pca_features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(
    X_scaled,
    columns=X.columns,
    index=X.index
)

#################
# PCA, scree Plot
#################
pca = PCA()
pca.fit(X_scaled)

explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

components = np.arange(1, len(explained_var) + 1)

plt.figure(figsize=(8, 5))

# Individual explained variance (bars)
plt.bar(
    components,
    explained_var,
    alpha=0.7,
    label='Individual Explained Variance'
)

# Cumulative explained variance (line)
plt.plot(
    components,
    cumulative_var,
    marker='o',
    linestyle='--',
    label='Cumulative Explained Variance'
)

plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA Scree Plot')
plt.xticks(components)
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()

###########################
# PCA two compontents plot
###########################

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create DataFrame with PC scores
pca_df = pd.DataFrame(
    X_pca,
    columns=['PC1', 'PC2'],
    index=X_scaled.index
)

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(
    pca_df['PC1'],
    pca_df['PC2'],
    alpha=0.6,
    s=10
)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Projection (PC1 vs PC2)')
plt.grid(True, alpha=0.3)
plt.show()


#############################################
# PCA two compontents + target variable plot
#############################################

pca_df['cpy_bin'] = pd.cut(
    df['cpy'],
    bins=[-0.01, 0, 1, 2, np.inf],
    labels=['0', '(0, 1]', '(1, 2]', '>2']
)


plt.figure(figsize=(8, 6))
for label in pca_df['cpy_bin'].cat.categories:
    mask = pca_df['cpy_bin'] == label
    plt.scatter(
        pca_df.loc[mask, 'PC1'],
        pca_df.loc[mask, 'PC2'],
        label=f'cpy {label}',
        alpha=0.5,
        s=10
    )

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Projection by Claim Frequency Bins')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


