import os
from typing import Tuple, NoReturn

import pandas as pd
import numpy as np
from boto3.session import Session
from sklearn.model_selection import train_test_split


def download_data_from_s3(
    s3_bucket: str, s3_path: str,
    local_output_path: str
) -> NoReturn:
    session = Session()
    s3 = session.client("s3", endpoint_url='https://storage.yandexcloud.net')
    print(s3_bucket, s3_path, local_output_path)
    s3.download_file(s3_bucket, s3_path, local_output_path)

def read_data(
    path: str,
    categorical_features,
    numerical_features
) -> pd.DataFrame:

    data = pd.read_csv(path)
    
    for column in categorical_features:
        data[column] = data[column].astype('category')
    
    for column in numerical_features:
        data[column] = data[column].astype(np.float32)

    return data


def split_train_val_data(
    data: pd.DataFrame, test_size: float,
    random_state: float, shuffle: bool
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data, test_data = train_test_split(
        data, test_size=test_size,
        random_state=random_state,
        shuffle=shuffle
    )
    return train_data, test_data
