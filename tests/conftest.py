import os

import pytest
from typing import List


@pytest.fixture()
def dataset_path():
    curdir = os.path.dirname(__file__)
    return os.path.join(curdir, "train_sample.csv")


@pytest.fixture()
def artifacts_path():
    curdir = os.path.dirname(__file__)
    return os.path.join(curdir, "artifacts")


@pytest.fixture()
def target_col():
    return "condition"


@pytest.fixture()
def categorical_features() -> List[str]:
    return [
        "sex", "cp", "fbs", "restecg",
        "exang", "slope", "ca", "thal"
    ]


@pytest.fixture
def numerical_features() -> List[str]:
    return [
        "age", "trestbps",
        "chol", "thalach", "oldpeak"
    ]
