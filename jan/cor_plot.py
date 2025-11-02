import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import pairplot

train_data = pd.read_csv('claims_train.csv')

# pairplot(train_data[
#     ['Area', 'VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 
#      'VehBrand', 'VehGas', 'Density', 'Region']
# ], diag_kind='kde')

train_data['Claims_per_year'] = train_data['ClaimNb'] / train_data['Exposure']

corr = train_data[['VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Density', 'Claims_per_year']].corr()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    annot_kws={"size": 12},
    cbar_kws={"shrink": 0.8},
    square=True
)
plt.title('Correlation Heatmap', fontsize=14, pad=15)
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()
plt.show()
