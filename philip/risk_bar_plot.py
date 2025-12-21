# bin by predicted CPY, compare mean actual vs mean predicted

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_train = # load training set
df_test = # load test set

y_test_cpy = df_test['CPY']
w_test = df_test['Exposure']
y_train_count = df_train['ClaimNb']
w_train = df_train['Exposure']

test_pred = # insert predictions on the test set here

n_bins = 10
pred_bins = pd.qcut(test_pred, q=n_bins, duplicates='drop')

calibration_df = pd.DataFrame({
    'actual': y_test_cpy,
    'predicted': test_pred,
    'exposure': w_test,
    'bin': pred_bins
})

# Calculate exposure-weighted means per bin
calibration = calibration_df.groupby('bin', observed=True).apply(
    lambda g: pd.Series({
        'mean_predicted': np.average(g['predicted'], weights=g['exposure']),
        'mean_actual': np.average(g['actual'], weights=g['exposure']),
        'count': len(g)
    })
).reset_index()

# Baseline rate (weighted mean CPY)
baseline_rate = y_train_count.sum() / w_train.sum()

fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(calibration))
width = 0.35
bars_pred = ax.bar([i - width/2 for i in x], calibration['mean_predicted'], width, label='Mean Predicted CPY', color='steelblue')
bars_actual = ax.bar([i + width/2 for i in x], calibration['mean_actual'], width, label='Mean Actual CPY', color='coral')

# Add baseline horizontal line
ax.axhline(y=baseline_rate, color='black', linestyle='--', linewidth=2, label=f'Baseline (weighted mean): {baseline_rate:.4f}')

# Add value labels on top of bars
for bar in bars_pred:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', ha='center', va='bottom', fontsize=9)

for bar in bars_actual:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Prediction bin (Low to high risk)')
ax.set_ylabel('Claims Per Year (CPY)')
# ax.set_title('Calibration: Predicted vs Actual CPY by Risk Decile')
ax.set_xticks(x)
ax.set_xticklabels([f'{i+1}' for i in x])
ax.legend()
plt.tight_layout()
plt.show()

print(calibration[['mean_predicted', 'mean_actual', 'count']].to_string())