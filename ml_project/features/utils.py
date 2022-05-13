from typing import List
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def process_categorical_features(categorical_df: pd.DataFrame) -> pd.DataFrame:

    categorical_pipeline = build_categorical_pipeline()
    return pd.DataFrame(categorical_pipeline.fit_transform(categorical_df).toarray())


def build_categorical_pipeline(use_ohe_for_categorical: bool) -> Pipeline:
    if use_ohe_for_categorical:
        categorical_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
                ("ohe", OneHotEncoder()),
            ]
        )
    else:
        categorical_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
            ]
        )
    return categorical_pipeline


def process_numerical_features(numerical_df: pd.DataFrame) -> pd.DataFrame:
    num_pipeline = build_numerical_pipeline()
    return pd.DataFrame(num_pipeline.fit_transform(numerical_df))


def build_numerical_pipeline() -> Pipeline:
    num_pipeline = Pipeline(
        [("impute", SimpleImputer(missing_values=np.nan, strategy="mean")), ]
    )
    return num_pipeline


def make_features(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    return transformer.transform(df)


def build_transformer(
    categorical_features: List[str],
    numerical_features: List[str],
    use_ohe_for_categorical: bool
) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                "categorical_pipeline",
                build_categorical_pipeline(use_ohe_for_categorical),
                categorical_features,
            ),
            (
                "numerical_pipeline",
                build_numerical_pipeline(),
                numerical_features,
            ),
        ]
    )
    return transformer


def extract_target(
    df: pd.DataFrame, target_col: str
) -> pd.Series:
    target = df[target_col]
    return target
