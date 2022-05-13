
from typing import Tuple

import pandas as pd
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import pickle

import conf.schema
from data.utils import read_data


conf.schema.register_configs()


@hydra.main(config_path='conf', config_name='config')
def predict(cfg: DictConfig) -> Tuple[str, dict]:

    with open(to_absolute_path(cfg.paths.output_model_path), "rb") as f:
        model = pickle.load(f)

    test_data = read_data(
        to_absolute_path(cfg.paths.test_data_path),
        cfg.features.categorical_features,
        cfg.features.numerical_features
    )
    test_data = test_data.drop(cfg.features.target_col, axis=1)
    prediction = model.predict(test_data)
    prediction = pd.Series(prediction)
    prediction.to_csv(
        to_absolute_path(cfg.paths.predict_data_path),
        index=None
    )


if __name__ == "__main__":
    predict()
