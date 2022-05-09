import pickle
from typing import Dict, Union

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.pipeline import Pipeline

Classifier = Union[
    CatBoostClassifier,
    LogisticRegression,
    LinearRegression
]


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    cfg: DictConfig
) -> Classifier:
    """
        Function get features and return trained model.

        Input:
            features: pd.DataFrame,
            target: pd.Series,
            cfg: DictConfig
        Returns:
            model: cfg.model
    """
    model = instantiate(cfg.model)
    model.fit(features, target)
    return model


def predict_model(
    model: Pipeline,
    features: pd.DataFrame,
    predict_proba: bool = True
) -> np.ndarray:
    """
        Function get model pipeline and make prediction
        for provided features.

        Input:
            model: Pipeline,
            features: pd.DataFrame,
            predict_proba: bool = True
        Returns:
            predicts: np.ndarray
    """
    predicts = model.predict(features)
    if predict_proba:
        predicts = model.predict_proba(features)
    return predicts


def evaluate_model(
    predict_probas: np.ndarray,
    target: pd.Series
) -> Dict[str, float]:
    """
        Function calculated metric between prediction
        probas and truth target.

        Input:
            predict_probas: np.ndarray,
            target: pd.Series
        Returns:
            metrics: Dict[str, float]
    """
    predict_labels = np.argmax(predict_probas, axis=1)
    return {
        "roc_auc": roc_auc_score(target, predict_probas[:, 1]),
        "accuracy": accuracy_score(target, predict_labels),
        "f1_score": f1_score(target, predict_labels)
    }


def create_inference_pipeline(
    model: Classifier,
    transformer: ColumnTransformer
) -> Pipeline:
    return Pipeline([("feature_part", transformer), ("model_part", model)])


def serialize_model(
    model: object,
    output_path: str
) -> str:
    """
        Function dump model to output path and return this path.

        Input:
            model: object,
            output_path: str
        Returns:
            output_path: str
    """
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    return output_path
