from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd

# Step 1: Feature engineering (target-independent)
class SmoothedTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column, alpha=10):
        self.column = column
        self.alpha = alpha
        self.encoding_dict = {}
        self.global_mean = None

    def fit(self, X, y):
        if y is None:
            raise ValueError("y cannot be None for SmoothedTargetEncoder")

        self.global_mean = y.mean()

        # Compute statistics per category
        df = pd.DataFrame({'category': X[self.column], 'target': y})
        stats = df.groupby('category')['target'].agg(['mean', 'count'])

        # Apply smoothing formula
        stats['encoded'] = (
                (stats['count'] * stats['mean'] + self.alpha * self.global_mean) /
                (stats['count'] + self.alpha)
        )

        self.encoding_dict = stats['encoded'].to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        X[self.column] = X[self.column].map(self.encoding_dict).fillna(self.global_mean)
        return X


def apply_onehot_with_names(X_train, X_val, onehot_cols):
    """
    Apply one-hot encoding while preserving column names

    Parameters:
    -----------
    X_train : pd.DataFrame
        Training data
    X_val : pd.DataFrame
        Validation data
    onehot_cols : list
        Columns to one-hot encode

    Returns:
    --------
    X_train_encoded, X_val_encoded : pd.DataFrame
        Encoded dataframes with proper column names
    encoder : ColumnTransformer
        Fitted encoder for future use
    """
    encoder = ColumnTransformer([
        ('onehot', OneHotEncoder(sparse_output=False, drop='first'), onehot_cols)
    ], remainder='passthrough')

    # Fit and transform
    X_train_array = encoder.fit_transform(X_train)
    X_val_array = encoder.transform(X_val)

    # Get feature names
    try:
        # Try sklearn 1.0+ method
        feature_names = encoder.get_feature_names_out()
    except AttributeError:
        # Fallback for older sklearn versions
        onehot_features = encoder.named_transformers_['onehot'].get_feature_names_out(onehot_cols)
        passthrough_features = [col for col in X_train.columns if col not in onehot_cols]
        feature_names = list(onehot_features) + passthrough_features

    # Convert to DataFrame
    X_train_encoded = pd.DataFrame(
        X_train_array,
        columns=feature_names,
        index=X_train.index
    )
    X_val_encoded = pd.DataFrame(
        X_val_array,
        columns=feature_names,
        index=X_val.index
    )

    return X_train_encoded, X_val_encoded, encoder


class FeatureEngineeringPipeline(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.area_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
        self.popular_brands = ['B12', 'B2', 'B1']
        self.popular_regions = ['R24', 'R82', 'R93', 'R11', 'R53', 'R52', 'R91', 'R72', 'R31']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X[X['Exposure'] < 1]
        X = X.drop(['IDpol'], axis=1, errors='ignore')

        # Create target (if ClaimNb exists)
        if 'ClaimNb' in X.columns:
            X['cpy'] = X['ClaimNb'] / X['Exposure']

        # Log transforms
        X['Density'] = np.log10(X['Density'])
        X['BonusMalus'] = np.log10(X['BonusMalus'])

        # Ordinal encoding
        X['Area'] = X['Area'].map(self.area_mapping)

        # Group rare categories
        X['VehBrand'] = X['VehBrand'].apply(
            lambda x: x if x in self.popular_brands else 'Other'
        )
        X['Region'] = X['Region'].apply(
            lambda x: x if x in self.popular_regions else 'Other'
        )

        return X


def create_full_pipeline(use_target_encoding=False, alpha=10):
    if use_target_encoding:
        # Pipeline with target encoding
        return Pipeline([
            ('feature_engineering', FeatureEngineeringPipeline()),
            ('target_encode_region', SmoothedTargetEncoder(column='Region', alpha=alpha)),
            ('onehot', ColumnTransformer([
                ('onehot_encoder', OneHotEncoder(sparse_output=False, drop='first'),
                 ['VehBrand', 'VehGas'])
            ], remainder='passthrough'))
        ])
    else:
        # Pipeline with one-hot encoding
        return Pipeline([
            ('feature_engineering', FeatureEngineeringPipeline()),
            ('onehot', ColumnTransformer([
                ('onehot_encoder', OneHotEncoder(sparse_output=False, drop='first'),
                 ['VehBrand', 'VehGas', 'Region'])
            ], remainder='passthrough'))
        ])


    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    data = pd.read_csv('data/claims_train.csv')

    pipeline_target = create_full_pipeline(use_target_encoding=True, alpha=10)

    feat_engineer = FeatureEngineeringPipeline()
    X_engineered = feat_engineer.fit_transform(data)

    # Split
    X = X_engineered.drop(['ClaimNb', 'Exposure', 'cpy'], axis=1, errors='ignore')
    y = X_engineered['cpy']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Apply target encoding
    target_encoder = SmoothedTargetEncoder(column='Region', alpha=10)
    X_train_encoded = target_encoder.fit_transform(X_train, y_train)
    X_val_encoded = target_encoder.transform(X_test)

    # Then one-hot encode remaining
    onehot = ColumnTransformer([
        ('onehot_encoder', OneHotEncoder(sparse_output=False, drop='first'),
         ['VehBrand', 'VehGas'])
    ], remainder='passthrough')

    X_train = onehot.fit_transform(X_train_encoded)
    X_test = onehot.transform(X_val_encoded)

    X_train, X_test, onehot_encoder = apply_onehot_with_names(
        X_train_encoded,
        X_val_encoded,
        ['VehBrand', 'VehGas']
    )

    # Fix column names (one-hot encoder adds prefixes)
    X_train.columns = X_train.columns.str.replace('onehot__', '')
    X_test.columns = X_test.columns.str.replace('onehot__', '')
    X_train.columns = X_train.columns.str.replace('remainder__', '')
    X_test.columns = X_test.columns.str.replace('remainder__', '')