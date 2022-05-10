import pandas as pd
from ml_project.data.utils import read_data, split_train_val_data


def test_load_dataset(
    dataset_path: str, target_col: str,
    categorical_features: str,
    numerical_features: str
) -> None:
    data = read_data(
        dataset_path,
        categorical_features,
        numerical_features
    )
    assert len(data) > 10
    assert target_col in data.keys()
    assert isinstance(data, pd.DataFrame)


def test_split_dataset(
    dataset_path: str,
    categorical_features: str,
    numerical_features: str
):
    data = read_data(
        dataset_path,
        categorical_features,
        numerical_features
    )
    train, test = split_train_val_data(
        data, random_state=42, test_size=0.2, shuffle=True
    )
    data_shape = data.shape[0]
    assert train.shape[0] > 0.7 * data_shape
    assert test.shape[0] < 0.3 * data_shape
    assert isinstance(train, pd.DataFrame)
