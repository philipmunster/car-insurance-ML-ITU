import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import pairplot
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv('claims_train.csv')

# Drop Exposure > 1, IDpol
df = df[df['Exposure'] <= 1]
df = df.drop(columns=['IDpol'])

# Drop the categorical variables
df = df.drop(columns = ['VehBrand', 'Region'])
df = df.drop(columns=["Area", "VehGas"])

# Implement Target Variable, drop ClaimNb, Exposure
df['cpy'] = df['ClaimNb'] / df['Exposure']
df = df.drop(columns=['ClaimNb'])

numerical_features = [
    'VehAge',
    'DrivAge',
    'VehPower',
    'Density',
    'BonusMalus',
    'cpy'
]

# Select numerical features
df_num = df[numerical_features]

# Compute correlation matrix
corr = df_num.corr()

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.5,
    annot_kws={"size": 15},
    cbar_kws={"shrink": 0.8}
)

plt.title("Correlation Heatmap of Numerical Features", fontsize=14)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

plt.tight_layout()
plt.show()


