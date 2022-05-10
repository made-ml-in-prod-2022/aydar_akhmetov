from dataclasses import dataclass
from typing import Any, List, Optional

import hydra
from hydra.core.config_store import ConfigStore
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
    cat_features: Optional[List[str]] = None


@dataclass
class PathConfig:
    raw_data_path: str = "data/raw/heart_cleveland_upload.csv"
    train_data_path: str = "data/interim/train.csv"
    test_data_path: str = "data/interim/test.csv"
    output_model_path: str = "models/artifacts/model.pkl"
    output_metric_path: str = "models/artifacts/metrics.json"


@dataclass
class DownloadConfig:
    use_download: bool = False
    s3_bucket: str = "ml-project-bucket"
    s3_path: str = "raw-data/heart_cleveland_upload.csv"
    local_output_path: str = "data/raw/"


@dataclass
class SplittingConfig:
    test_size: float = 0.2
    random_state: int = 42
    shuffle: bool = True
    save_interim: bool = True


@dataclass
class CrossValConfig:
    test_size: float = 0.2
    random_state: int = 42
    shuffle: bool = True


@dataclass
class FeaturesConfig:
    categorical_features: Optional[List[str]] = None
    numerical_features: Optional[List[str]] = None
    features_to_drop: Optional[List[str]] = None
    target_col: str = "condition"
    use_ohe_for_categorical: bool = False


@dataclass
class MlflowConfig:
    use_mlflow: bool = False
    mlflow_uri: str = "http://127.0.0.1:5000/"
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
    splitting: SplittingConfig = SplittingConfig()
    features: FeaturesConfig = FeaturesConfig()
    cross_val: CrossValConfig = CrossValConfig()
    mlflow: MlflowConfig = MlflowConfig()


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Config)
    cs.store(group="model", name="base_catboost", node=CatboostConfig)
    cs.store(group="model", name="base_logreg", node=LogregConfig)
