from hydra import initialize, compose
from sklearn.utils.validation import check_is_fitted
import pandas as pd

from ml_project.data.utils import read_data, split_train_val_data
from ml_project.models.utils import train_model
from ml_project.features.utils import extract_target


def test_train_model(
    dataset_path: str,
    categorical_features: str,
    numerical_features: str
):
    with initialize(config_path="../../ml_project/conf"):
        cfg = compose(config_name="config",)
        data = read_data(
            dataset_path,
            categorical_features,
            numerical_features
        )
        train, test = split_train_val_data(
            data, random_state=42, test_size=0.2, shuffle=True
        )
        train_target = extract_target(train, cfg.features.target_col)
        test_target = extract_target(test, cfg.features.target_col)
        train = train.drop(cfg.features.target_col, axis=1)
        test = test.drop(cfg.features.target_col, axis=1)

        model = train_model(train, train_target, cfg)
        check_is_fitted(model)

        prediction = model.predict(test)
        assert prediction.shape[0] == test_target.shape[0]


def test_train_model_on_generated_data(
    synthetic_data: pd.DataFrame,
):
    with initialize(config_path="../../ml_project/conf"):
        cfg = compose(config_name="config",)
        train, test = split_train_val_data(
            synthetic_data, random_state=42, test_size=0.2, shuffle=True
        )
        train_target = extract_target(train, cfg.features.target_col)
        test_target = extract_target(test, cfg.features.target_col)
        train = train.drop(cfg.features.target_col, axis=1)
        test = test.drop(cfg.features.target_col, axis=1)

        model = train_model(train, train_target, cfg)
        check_is_fitted(model)

        prediction = model.predict(test)
        assert prediction.shape[0] == test_target.shape[0]
