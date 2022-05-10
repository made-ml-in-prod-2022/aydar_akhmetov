ml_project
==============================

Description:
~~~
This is an example of ml project, including the following features:
- training pipeline with module structuring
- all external dependencies are tracked by poetry
- also poetry provides building package via command poetry build
- hydra configs with saving training artifacts
- mlflow with model registry, mlflow server can work local or remote
- dvc with data tracking using yandex s3 object storage
- loggers using in main pipeline
- all train artifacts and logs save for reproducibility 
- tests for end2end training and for individual modules
~~~

Installation:
~~~
in process
~~~

Usage:
~~~
cd ml_project/
python train_pipeline.py
or if you want change some paramater via command line:
python train_pipeline.py general.random_state=43
~~~

Run mlflow server:
~~~
mlflow server \
    --backend-store-uri sqlite:///mydb.sqlite \
    --default-artifact-root $(pwd)/ml_project/mlruns/ \
    --host 0.0.0.0 --port 5001
~~~

Test:
~~~
pytest tests/
~~~

Project Organization
------------

├── dist                <- Package successufully built to this directory
├── .dvc                <- If you want data tracking for reproducibility
├── .git                <- You see this text because of this guy
├── README.md           <- The top-level README for developers using this project.
├── ml_project          <- Main code there
│   ├── conf            <- Pretty config files
│   │   ├── cross_val
│   │   ├── download
│   │   ├── features
│   │   ├── general
│   │   ├── mlflow
│   │   ├── model
│   │   ├── paths
│   │   └── splitting
│   ├── data            <- Data for data science magic
│   │   ├── interim
│   │   ├── predictions
│   │   └── raw
│   ├── features        <- Feature utils
│   ├── mlruns          <- Mlflow artifacts, you need this for reproducibility
│   ├── models          <- Model utils
│   │   └──artifacts    <- Saving model artifacts, why not
│   └── outputs         <- Hydra artifacts, you need this for reproducibility
├── notebooks           <- Very beatiful EDA with seaborn there
├── notes               <- This is my markdown notes for every feature in my project
└── tests               <- Testing end2end and modules 
    ├── artifacts
    ├── data_utils
    └── model_utils

--------