from dataclasses import dataclass
from typing import Any, List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf


@dataclass
class LogregConfig:
    _target_: str = 'sklearn.linear_model.LogisticRegression'
    penalty: str = 'l1'
    solver: str = 'liblinear'
    C: float = 1.0
    random_state: int = 42
    max_iter: int = 42


@dataclass
class CatboostConfig:
    _target_: str = "catboost.CatboostClassifier"
    random_seed: int = 42
    n_estimators: int = 100
    depth: int = 2
    learning_rate: float = 0.02
    iterations: int = 100
    verbose: bool = False
    l2_leaf_reg: int = 2
    cat_features: Optional[str] = None


@dataclass
class PathConfig:
    raw_data_path: str = "data/raw/heart_cleveland_upload.csv"
    train_data_path: str = "data/interim/train.csv"
    test_data_path: str = "data/interim/test.csv"
    output_model_path: str = "models/model.pkl"
    metric_path: str = "models/metrics.json"


@dataclass
class DownloadConfig:
    use: bool = False
    s3_bucket: str = "ml-project-bucket"
    s3_path: str = "heart_cleveland_upload.csv"
    local_output_path: str = "data/raw/"


@dataclass
class CrossValConfig:
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class FeaturesConfig:
    categorical_features: List[str] = None
    numerical_features: List[str] = None
    features_to_drop: List[str] = None
    target_col: str = "condition"
    use_ohe_for_categorical: bool = False


@dataclass
class MlflowConfig:
    use_mlflow: bool = False
    mlflow_uri: str = "http://18.156.5.226/"
    mlflow_experiment: str = "inference_demo"


@dataclass
class GeneralConfig:
    random_state: int = 42


@dataclass
class Config:
    paths: PathConfig = PathConfig()
    model: Any = LogregConfig()
    general: GeneralConfig = GeneralConfig()
    download: DownloadConfig = DownloadConfig()
    features: FeaturesConfig = FeaturesConfig()
    cross_val: CrossValConfig = CrossValConfig()
    mlflow: MlflowConfig = MlflowConfig()
