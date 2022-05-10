import os
import sys
from typing import List

from py._path.local import LocalPath
from hydra import initialize, compose

from ml_project.train_pipeline import run_train_pipeline


def test_config_setup() -> None:
    with initialize(config_path="../ml_project/conf"):
        cfg = compose(config_name="config",)
        assert "model" in cfg
        assert "paths" in cfg


def test_train_e2e(
    dataset_path: str,
    artifacts_path: str
):
    with initialize(config_path="../ml_project/conf"):
        cfg = compose(config_name="config",)

        # change paths for testing
        cfg.paths.raw_data_path = dataset_path
        cfg.splitting.save_interim = False
        cfg.paths.output_model_path = os.path.join(artifacts_path, 'model.pkl')
        cfg.paths.output_metric_path = os.path.join(artifacts_path, 'metrics.json')

        #from pdb import set_trace; set_trace()

        _, model_path, metrics = run_train_pipeline(cfg)
        assert metrics["roc_auc"] > 0.5
        assert os.path.exists(model_path)
