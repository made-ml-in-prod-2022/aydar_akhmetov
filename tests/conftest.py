import os
from typing import List
from collections import defaultdict

import pytest
import numpy as np
import pandas as pd


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


@pytest.fixture()
def numerical_features() -> List[str]:
    return [
        "age", "trestbps",
        "chol", "thalach", "oldpeak"
    ]


@pytest.fixture()
def generated_data() -> List[str]:
    return [
        "age", "trestbps",
        "chol", "thalach", "oldpeak"
    ]


@pytest.fixture()
def synthetic_data():
    np.random.seed(42)

    synt_data = defaultdict(list)

    for _ in range(1000):
        row = {
            "age": int(9 * np.random.randn() + 54),
            "sex": np.random.choice([0, 1], p=[0.33, 0.67]),
            "cp": np.random.choice([0, 1, 2, 3], p=[0.08, 0.16, 0.28, 0.48]),
            "trestbps": int(17 * np.random.randn() + 131),
            "chol": int(51 * np.random.randn() + 247),
            "fbs": np.random.choice([0, 1], p=[0.85, 0.15]),
            "restecg": np.random.choice([0, 1, 2], p=[0.48, 0.04, 0.48]),
            "thalach": int(22 * np.random.randn() + 149),
            "exang": np.random.choice([0, 1], p=[0.67, 0.33]),
            "oldpeak": max(1.16 * np.random.randn() + 1.05, 0),
            "slope": np.random.choice([0, 1, 2], p=[0.46, 0.46, 0.08]),
            "ca": np.random.choice([0, 1, 2, 3], p=[0.58, 0.22, 0.13, 0.07]),
            "thal": np.random.choice([0, 1, 2], p=[0.55, 0.39, 0.06]),
            "condition": np.random.choice([0, 1], p=[0.53, 0.47]),
        }
        for column, value in row.items():
            synt_data[column].append(value)

    synt_data = pd.DataFrame(synt_data)
    return synt_data
