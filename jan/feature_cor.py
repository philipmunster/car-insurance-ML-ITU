import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import pairplot

df = pd.read_csv('claims_train.csv')


import numpy as np
import pandas as pd
from scipy import stats
import scipy.stats as ss

numerical = ["VehAge", "DrivAge", "Density", "BonusMalus"]
ordinal   = ["VehPower", "Area"]
nominal   = ["VehBrand", "VehGas", "Region"]

all_vars = numerical + ordinal + nominal

df['Area'] = df['Area'].map({
    'A': 1,
    'B': 2,
    'C': 3,
    'D': 4,
    'E': 5,
    'F': 6
})

# 1. Cramér’s V  (nominal vs nominal)
def cramers_v(x, y):
    confusion = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion)[0]
    n = confusion.sum().sum()
    phi2 = chi2 / n
    r, k = confusion.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1)) / (n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


# 2. Correlation ratio η (nominal → numeric)
def correlation_ratio(categories, values):
    categories = np.array(categories)
    values = np.array(values)
    grand_mean = values.mean()
    ss_between = 0
    for cat in np.unique(categories):
        mask = categories == cat
        ss_between += mask.sum() * (values[mask].mean() - grand_mean) ** 2
    ss_total = ((values - grand_mean)**2).sum()
    return np.sqrt(ss_between / ss_total)


def print_all_correlations(df):
    for i in range(len(all_vars)):
        for j in range(i+1, len(all_vars)):
            v1 = all_vars[i]
            v2 = all_vars[j]

            # numeric - numeric → Pearson
            if v1 in numerical and v2 in numerical:
                corr = df[v1].corr(df[v2], method="pearson")
                method = "Pearson (numeric-numeric)"

            # ordinal - numeric → Spearman
            elif (v1 in ordinal and v2 in numerical) or (v2 in ordinal and v1 in numerical):
                corr = df[v1].corr(df[v2], method="spearman")
                method = "Spearman (ordinal-numeric)"

            # ordinal - ordinal → Spearman
            elif v1 in ordinal and v2 in ordinal:
                corr = df[v1].corr(df[v2], method="spearman")
                method = "Spearman (ordinal-ordinal)"

            # nominal - nominal → Cramér’s V
            elif v1 in nominal and v2 in nominal:
                corr = cramers_v(df[v1], df[v2])
                method = "Cramér’s V (nominal-nominal)"

            # nominal → numeric → η
            elif v1 in nominal and v2 in numerical:
                corr = correlation_ratio(df[v1], df[v2])
                method = "Correlation Ratio η (nominal → numeric)"

            elif v2 in nominal and v1 in numerical:
                corr = correlation_ratio(df[v2], df[v1])
                method = "Correlation Ratio η (nominal → numeric)"

            # print result
            print(f"{v1}  vs  {v2}  | {method}:  {corr:.4f}")


print_all_correlations(df)