import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('claims_train.csv')
test = pd.read_csv('claims_test.csv')

# Check for missing values
print(train.isnull().sum())
print(test.isnull().sum())

# Check for reasonable ranges in numeric columns
print(train.describe())
print(test.describe())