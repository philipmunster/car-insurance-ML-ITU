import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('claims_train.csv')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Density
axes[0].hist(df['Density'], bins=50)
axes[0].set_title("Distribution of Density")
axes[0].set_xlabel("Density")
axes[0].set_ylabel("Frequency")

# BonusMalus
axes[1].hist(df['BonusMalus'], bins=50)
axes[1].set_title("Distribution of BonusMalus")
axes[1].set_xlabel("BonusMalus")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()
