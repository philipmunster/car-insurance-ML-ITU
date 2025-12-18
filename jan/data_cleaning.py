import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('claims_train.csv')

# Drop Exposure > 1, IDpol
df = df[df['Exposure'] <= 1]
df = df.drop(columns=['IDpol'])

# Encode the categorical variables

# Ordinal encoding for Area
area_order = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5
}
df["Area_enc"] = df["Area"].map(area_order)

# One-hot encoding for VehBrand and Region
vehbrand_categories = [
    "B1", "B2", "B6", "B13", "B11", "B5",
    "B12", "B3", "B10", "B4", "B14"
]
df = pd.get_dummies(
    df,
    columns=["VehBrand"],
    prefix="VehBrand",
    dtype=int
)

region_categories = [
    "R24", "R25", "R82", "R53", "R54", "R11",
    "R94", "R93", "R91", "R52", "R72", "R31",
    "R73", "R23", "R22", "R41", "R42", "R83",
    "R21", "R26", "R74", "R43"
]
df = pd.get_dummies(
    df,
    columns=["Region"],
    prefix="Region",
    dtype=int
)

# Binary encoding for VehGas
df["VehGas_enc"] = df["VehGas"].map({
    "Regular": 0,
    "Diesel": 1
})

df = df.drop(columns=["Area", "VehGas"])

# Log-Transforming Skewed Features
df['log_Density'] = round(np.log1p(df['Density']),2)
df['log_BonusMalus'] = round(np.log1p(df['BonusMalus']),2)


# Implement Target Variable, drop ClaimNb, Exposure
df['cpy'] = df['ClaimNb'] / df['Exposure']
df = df.drop(columns=['ClaimNb', 'Exposure'])

