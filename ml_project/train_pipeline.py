
from typing import Any, Optional, NoReturn, Tuple
import json
import logging
import os
import sys
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import numpy as np
import mlflow

from data.utils import (
    read_data, split_train_val_data,
    download_data_from_s3
)
from features.utils import (
    make_features, extract_target,
    build_transformer,
)
from models.utils import (
    train_model,
    serialize_model,
    predict_model,
    evaluate_model,
    create_inference_pipeline
)
import conf.schema


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def run_train_pipeline(cfg: DictConfig) -> Tuple[str, dict]:

    logger.info(f"running train pipeline")
    logger.info(f"using data downloading: {cfg.download.use_download}")

    if cfg.download.use_download:
        os.makedirs(cfg.download.local_output_path, exist_ok=True)
        download_data_from_s3(
            cfg.download.s3_bucket,
            cfg.download.s3_path,
            to_absolute_path(os.path.join(
                cfg.download.local_output_path,
                Path(cfg.download.s3_path).name
            )),
        )

    logger.info(f"read data from {cfg.paths.raw_data_path}")
    data = read_data(
        to_absolute_path(cfg.paths.raw_data_path),
        cfg.features.categorical_features,
        cfg.features.numerical_features
    )
    logger.info(f"data shape is {data.shape}")

    train_data, test_data = split_train_val_data(
        data, cfg.splitting.test_size,
        cfg.splitting.random_state, cfg.splitting.shuffle
    )

    logger.info(f"using saving interim data: {cfg.splitting.save_interim}")

    if cfg.splitting.save_interim:
        logger.info(f"start saving interim data")
        train_data.to_csv(to_absolute_path(cfg.paths.train_data_path), index=None)
        test_data.to_csv(to_absolute_path(cfg.paths.test_data_path), index=None)

    test_target = extract_target(test_data, cfg.features.target_col)
    train_target = extract_target(train_data, cfg.features.target_col)
    train_data = train_data.drop(cfg.features.target_col, axis=1)
    test_data = test_data.drop(cfg.features.target_col, axis=1)

    logger.info(f"train_data shape is {train_data.shape}")
    logger.info(f"test_data shape is {test_data.shape}")

    transformer = build_transformer(
        train_data.select_dtypes(include="category").columns,
        train_data.select_dtypes(include="number").columns,
        cfg.features.use_ohe_for_categorical
    )

    transformer.fit(train_data)
    train_features = make_features(transformer, train_data)
    logger.info(f"train_features shape is {train_features.shape}")

    model = train_model(train_features, train_target, cfg)

    inference_pipeline = create_inference_pipeline(model, transformer)
    predicts = predict_model(inference_pipeline, test_data)
    metrics = evaluate_model(predicts, test_target)

    with open(to_absolute_path(cfg.paths.output_metric_path), "w") as metric_file:
        json.dump(metrics, metric_file)

    logger.info(f"metrics is {metrics}")

    model_path = serialize_model(
        inference_pipeline,
        to_absolute_path(cfg.paths.output_model_path)
    )

    return model, model_path, metrics


conf.schema.register_configs()


@hydra.main(config_path='conf', config_name='config')
def train_pipeline(cfg: DictConfig) -> Tuple[str, dict]:

    logger.info(f"running main pipeline")
    logger.info(f"using mlflow: {cfg.mlflow.use_mlflow}")

    if cfg.mlflow.use_mlflow:
        mlflow.set_tracking_uri(cfg.mlflow.mlflow_uri)
        mlflow.set_experiment(cfg.mlflow.mlflow_experiment)
        with mlflow.start_run():
            mlflow.log_params(cfg)
            model, model_path, metrics = run_train_pipeline(cfg)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(os.path.dirname(model_path))
            mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
    else:
        return run_train_pipeline(cfg)


if __name__ == "__main__":
    train_pipeline()
