from sklearn.preprocessing import OneHotEncoder
import pandas as pd

data = pd.read_csv('data/claims_train.csv')

# Filter bad data and correlated features
data = data[data['Exposure'] <= 1]
data.drop(['Area', 'DrivAge', 'IDpol'], axis=1, inplace=True)

# One-hot encode
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded = encoder.fit_transform(data[['VehBrand', 'VehGas', 'Region']])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())
df_final = pd.concat([data.drop(['VehBrand', 'VehGas', 'Region'], axis=1), encoded_df], axis=1)

df_encoded = df_final.drop(['Density', 'BonusMalus', 'ClaimNb', 'Exposure', 'VehAge', 'VehPower'], axis=1)

# Remove minority/majority classes
for col in df_encoded.columns:
    prop = df_encoded[col].mean()
    if prop < 0.05 or prop > 0.95:
        print(f"Drop {col}: {prop:.2%} prevalence")
        df_final.drop(col, axis=1, inplace=True)

df_final.dropna(inplace=True)