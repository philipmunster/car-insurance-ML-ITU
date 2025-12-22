import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('claims_train.csv')
df_test = pd.read_csv('claims_test.csv')

# Check for missing values
df.isnull().any(axis=1).sum()

# check IDpol is in fact unique
len(df) == len(df['IDpol'].unique())

# check values outside of realistic range
df.describe()

# check exposures longer than 1 year
len(df[df['Exposure'] > 1])

# remove policies with exposures more than 1 year
df = df[df['Exposure'] <= 1]

# check categories for each categorical variable
sorted(df['Region'].unique())
sorted(df['VehGas'].unique())
sorted(df['VehBrand'].unique())
sorted(df['Area'].unique())

# Check cardinality of categorical features
len(df['Region'].unique())
len(df['VehGas'].unique())
len(df['VehBrand'].unique())
len(df['Area'].unique())

# cleaned training and test dataset
# remove policies with exposures more than 1 year
df = df[df['Exposure'] <= 1]
df_test = df_test[df_test['Exposure'] <= 1]

# remove polID
df = df.drop(columns=['IDpol'])
df_test = df_test.drop(columns=['IDpol'])

# add target variable
df['CPY'] = df['ClaimNb'] / df['Exposure']
df_test['CPY'] = df_test['ClaimNb'] / df_test['Exposure']
df['has_claim'] = (df['CPY'] > 0).astype(int)
df_test['has_claim'] = (df['CPY'] > 0).astype(int)

# map area from letters A, B, C... to 0, 1, 2...
order = ['A', 'B', 'C', 'D', 'E', 'F']
df['Area'] = pd.Categorical(df['Area'], categories=order, ordered=True)
df['Area'] = df['Area'].cat.codes
df_test['Area'] = pd.Categorical(df_test['Area'], categories=order, ordered=True)
df_test['Area'] = df_test['Area'].cat.codes

# log transform
df['Density'] = np.log1p(df['Density'])
df_test['Density'] = np.log1p(df_test['Density'])

feature_cols = ['Area', 'VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'VehBrand', 'VehGas', 'Density', 'Region']
target_cols = ['ClaimNb', 'CPY', 'has_claim', 'Exposure']
categorical_cols = ['VehBrand', 'VehGas', 'Region']

# one-hot encoding 'VehBrand', 'VehGas', 'Region'

# on test data
df_test_onehot = df_test.copy()
df_test_onehot = pd.get_dummies(df_test_onehot, columns=['VehBrand', 'VehGas', 'Region'], drop_first=True, dtype=int)

# on training data
df_onehot = df.copy()
df_onehot = pd.get_dummies(df_onehot, columns=['VehBrand', 'VehGas', 'Region'], drop_first=True, dtype=int)
dummy_cols = ['VehBrand_B10', 'VehBrand_B11', 'VehBrand_B12', 'VehBrand_B13', 'VehBrand_B14', 'VehBrand_B2', 'VehBrand_B3', 'VehBrand_B4', 'VehBrand_B5', 'VehBrand_B6', 'VehGas_Regular', 'Region_R21', 'Region_R22', 'Region_R23', 'Region_R24', 'Region_R25', 'Region_R26', 'Region_R31', 'Region_R41', 'Region_R42', 'Region_R43', 'Region_R52', 'Region_R53', 'Region_R54', 'Region_R72', 'Region_R73', 'Region_R74', 'Region_R82', 'Region_R83', 'Region_R91', 'Region_R93', 'Region_R94']
variances = df_onehot[dummy_cols].mean()

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_poisson_deviance


model = DecisionTreeRegressor(max_depth=7, min_samples_leaf=400, random_state=42, criterion="poisson")

X_train = df_onehot.drop(columns=target_cols)
y_train_count = df_onehot['ClaimNb']
y_train_rate = df_onehot['CPY']
w_train = df_onehot['Exposure']

X_test = df_test_onehot.drop(columns=target_cols)
y_test_count = df_test_onehot['ClaimNb']
y_test_rate = df_test_onehot['CPY']
w_test = df_test_onehot['Exposure']

model.fit(X_train, y_train_rate, sample_weight=w_train)
train_pred = model.predict(X_train)
train_mean_poisson_deviance_rate = mean_poisson_deviance(y_train_rate, train_pred, sample_weight=w_train)
# train_mean_poisson_deviance_count = mean_poisson_deviance(y_train_count, train_pred, sample_weight=w_train)
print(f"Train Poisson deviance on rate: {train_mean_poisson_deviance_rate}")
# print(f"Train Poisson deviance on count: {train_mean_poisson_deviance_count}")

poisson_test_pred = model.predict(X_test)
test_mean_poisson_deviance_rate = mean_poisson_deviance(y_test_rate, poisson_test_pred, sample_weight=w_test)
# test_mean_poisson_deviance_count = mean_poisson_deviance(y_test_count, test_pred, sample_weight=w_test)
print(f"Test Poisson deviance on rate: {test_mean_poisson_deviance_rate}")
# print(f"Test Poisson deviance on count: {test_mean_poisson_deviance_count}")