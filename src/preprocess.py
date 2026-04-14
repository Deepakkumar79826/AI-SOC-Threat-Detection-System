import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_preprocessor(df: pd.DataFrame):
    feature_df = df.drop(columns=["label", "difficulty", "target"])

    categorical_cols = ["protocol_type", "service", "flag"]
    numerical_cols = [col for col in feature_df.columns if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    return preprocessor, feature_df.columns.tolist()

def prepare_features(df: pd.DataFrame):
    X = df.drop(columns=["label", "difficulty", "target"])
    y = df["target"]
    return X, y